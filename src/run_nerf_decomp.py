# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(5)
#

import numpy as np
import torch
from torch.distributions.categorical import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

from config_parser import export_config
from nerf_models.nerf_decomp_renderer import render_decomp, render_decomp_path
from nerf_models.nerf_decomp import create_NeRFDecomp

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils.label_utils import *
from config_parser import recursive_config_parser
from dataset.dataset_interface import load_dataset
from miscellaneous.test_dataset_speed import *

from utils.generator_utils import *
from utils.timing_utils import *
import cv2
from torch.nn.functional import normalize

def test():
    raise NotImplementedError


def test_parser():
    parser = recursive_config_parser()
    args = parser.parse_args()

def test_autograd():
    a = torch.tensor([[2., 3.], [4., 5.]])
    b = torch.tensor([[1., 1.], [1., 1.]])
    L = nn.Linear(2, 1)
    L.weight.data.fill_(1)
    L.bias.data.fill_(0)

    a.requires_grad = True
    Q = L(a*b)**2
    print("Q", Q)
    Q.backward(torch.ones_like(Q))
    print(a.grad)

    a = torch.tensor([[2., 5.], [3., 4.]])
    a.requires_grad = True
    Q = L(a*b) ** 2
    print("Q", Q)
    Q.backward(torch.ones_like(Q))
    print(a.grad)
    #print(b.grad)

def test_base_color():
    parser = recursive_config_parser()
    args = parser.parse_args()
    args.device = device
    # (1) Load dataset
    with time_measure("[1] Data load"):
        def load_dataset_split(split="train", do_logging=True, **kwargs):
            # create dataset config
            target_dataset = load_dataset(args.dataset_type, args.datadir, split=split, **kwargs)
            # real data load using multiprocessing(torch DataLoader) --> load all at once
            # TODO : if dataset is too large, it may not be loaded at once.
            target_dataset.load_all_data(num_of_workers=10)
            return target_dataset

        # load train and validation dataset
        dataset = load_dataset_split("train", sample_length=args.sample_length, image_scale=args.image_scale)

    # (2) Load dataset
    with time_measure("[2] Base color evaluation"):
        dataset.get_base_color(visualize=True)


def train():
    parser = recursive_config_parser()
    args = parser.parse_args()
    args.device = device

    if args.render_only:
        test()
        return

    # (0) Print train phase overview
    logger_dataset = load_logger("Dataset Info")
    logger_export = load_logger("Export Logger")
    use_instance_mask = args.instance_mask
    logger_dataset.info("Instance mask: " + str(use_instance_mask))
    logger_dataset.info("Instance mask encoding: " + str(args.instance_label_encoding))
    logger_dataset.info("Infer normal: " + str(args.infer_normal))
    logger_dataset.info("Learn normal from oracle: " + str(args.learn_normal_from_oracle))
    logger_dataset.info("Learn albedo from oracle: " + str(args.learn_albedo_from_oracle))


    # (1) Load dataset
    with time_measure("[1] Data load"):
        def load_dataset_split(split="train", do_logging=True, **kwargs):
            # create dataset config
            target_dataset = load_dataset(args.dataset_type, args.datadir, split=split, **kwargs)
            target_dataset.load_instance_label_mask = use_instance_mask

            # real data load using multiprocessing(torch DataLoader) --> load all at once
            # TODO : if dataset is too large, it may not be loaded at once.
            target_dataset.load_all_data(num_of_workers=10)
            if do_logging:
                logger_dataset.info(target_dataset)
            return target_dataset

        # load train and validation dataset
        load_params = {
            "image_scale": args.image_scale,
            "load_normal": True, #args.learn_normal_from_oracle,
            "load_albedo": args.learn_albedo_from_oracle,
            "sample_length": args.sample_length
        }
        dataset = load_dataset_split("train", **load_params)
        dataset_val = load_dataset_split("test", skip=10, **load_params)

        # calculate base color
        dataset.get_base_color(
            learn_from_gt_albedo_map=args.learn_albedo_from_oracle,
            cluster_image_number=args.cluster_image_number,
            cluster_image_resize=args.cluster_image_resize,
            cluster_init_number=args.cluster_init_number,
            cluster_merge_threshold=args.cluster_merge_threshold,
            cluster_number_lower_bound=args.cluster_number_lower_bound
        )

        os.makedirs(os.path.join(args.basedir, args.expname), exist_ok=True)
        if os.path.isfile(os.path.join(args.basedir, args.expname, 'init_basecolor.txt')):
            dataset.init_basecolor = np.loadtxt(os.path.join(args.basedir, args.expname, 'init_basecolor.txt'))
        else:
            np.savetxt(os.path.join(args.basedir, args.expname, 'init_basecolor.txt'), dataset.init_basecolor)
        init_basecolor = dataset.init_basecolor
        dataset_val.init_basecolor = init_basecolor

        hwf = [dataset.height, dataset.width, dataset.focal]

        # move data to GPU side
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        dataset.to_tensor(args.device)
        dataset_val.to_tensor(args.device)

        # Load BRDF LUT
        brdf_lut_path = "../data/ibl_brdf_lut.png"
        brdf_lut = cv2.imread(brdf_lut_path)
        brdf_lut = cv2.cvtColor(brdf_lut, cv2.COLOR_BGR2RGB)
        brdf_lut = brdf_lut.astype(np.float32)
        brdf_lut /= 255.0
        brdf_lut = torch.tensor(brdf_lut).to(args.device)
        brdf_lut = brdf_lut.permute((2, 0, 1))

    # (2) Create log file / folder
    with time_measure("[2] Log file create"):
        # Create log dir and copy the config file
        basedir = args.basedir
        expname = args.expname
        export_config(args)

        # Create Tensorboard writer
        writer = SummaryWriter(log_dir=os.path.join(basedir, expname))

    # (3) Create nerf model
    with time_measure("[3] NeRFDecomp load"):
        # set instance label dimension
        label_encoder = None
        if use_instance_mask:
            label_encoder = get_label_encoder(dataset.instance_color_list, args.instance_label_encoding, args.instance_label_dimension)
            args.instance_label_dimension = label_encoder.get_dimension()
        else:
            args.instance_label_dimension = 0

        # create nerf model
        args.num_cluster = dataset.num_cluster
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_NeRFDecomp(args)
        global_step = start

        # update near / far plane
        bds_dict = dataset.get_near_far_plane()
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        render_kwargs_train['brdf_lut'] = brdf_lut
        render_kwargs_test['brdf_lut'] = brdf_lut

        if use_instance_mask:
            is_instance_label_logit = isinstance(label_encoder, OneHotLabelEncoder) and (args.CE_weight_type != "mse")
            render_kwargs_train["is_instance_label_logit"] = is_instance_label_logit
            render_kwargs_test["is_instance_label_logit"] = is_instance_label_logit
        logger_render_options = load_logger("Render Kwargs")
        logs = ["[Render Kwargs (simple only)]"]
        for k, v in render_kwargs_train.items():
            if isinstance(v, (str, float, int, bool)):
                logs += ["\t-%s : %s" % (k, str(v))]
        logger_render_options.info("\n".join(logs))

    # (4) Create the sample generator
    with time_measure("[4] Sample generator create"):
        batch_size = args.N_rand
        use_batching = not args.no_batching
        start = start + 1
        if use_batching:
            sample_generator = sample_generator_all_image_merged(dataset, batch_size=batch_size)
        else:
            sample_generator = sample_generator_single_image(dataset, batch_size=batch_size,
                                                             precrop_iters=args.precrop_iters,
                                                             precrop_frac=args.precrop_frac, initial_iters=start)

    # (5) Main train loop
    K = dataset.get_focal_matrix()
    N_iters = args.N_iter + 1

    # export ground truth image
    img_gt = dataset_val.images.permute((0, 3, 1, 2))
    writer.add_images('test/gt_rgb', img_gt, 0)

    if use_instance_mask:
        colored_label_gt = label_to_colored_label(dataset_val.masks, label_encoder.label_color_list)
        colored_label_gt = colored_label_gt.permute((0, 3, 1, 2))
        writer.add_images('test/gt_instance_colored', colored_label_gt, 0)

    if True:#args.learn_normal_from_oracle:
        normal_gt = dataset_val.normals.permute((0, 3, 1, 2))
        writer.add_images('test/gt_normal', normal_gt, 0)
    if args.learn_albedo_from_oracle:
        albedo_gt = dataset_val.albedos.permute((0, 3, 1, 2))
        writer.add_images('test/gt_albedo', albedo_gt, 0)

    mse_loss = torch.nn.MSELoss()
    l1_loss = torch.nn.L1Loss()
    render_kwargs_test["calculate_normal_from_sigma_gradient"] = True
    render_kwargs_test["calculate_normal_from_sigma_gradient_surface"] = True
    render_kwargs_test["calculate_normal_from_depth_gradient"] = True

    normal_target_keys = [
        "normal_map_from_sigma_gradient",
        "normal_map_from_sigma_gradient_surface",
        "normal_map_from_depth_gradient"
    ]
    if args.infer_normal:
        assert args.infer_normal_target in normal_target_keys

    for i in trange(start, N_iters):
        # sample rgb and rays from sample generator
        target_info, rays_o, rays_d = next(sample_generator)  # 3 x (N_rand, 3)
        target_rgb = target_info["rgb"]
        if args.learn_albedo_from_oracle:
            target_chromaticity = target_info["albedo"]
        else:
            target_chromaticity = target_rgb / (torch.linalg.norm(target_rgb, dim=-1, keepdim=True) + 1e-10)
        batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)

        #####  Core optimization loop  #####
        calculate_normal_from_depth_gradient = i % args.summary_step == 0
        calculate_normal_from_sigma_gradient = i % args.summary_step == 0
        calculate_normal_from_sigma_gradient_surface = i % args.summary_step == 0
        if i >= args.N_iter_ignore_normal:
            if args.infer_normal_target == "normal_map_from_sigma_gradient":
                calculate_normal_from_sigma_gradient = True
            elif args.infer_normal_target == "normal_map_from_sigma_gradient_surface":
                calculate_normal_from_sigma_gradient_surface = True
            elif args.infer_normal_target == "normal_map_from_depth_gradient":
                calculate_normal_from_depth_gradient = True

        # 1. render sample
        result = render_decomp(
            dataset.height, dataset.width, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,
            init_basecolor=dataset.init_basecolor,
            calculate_normal_from_sigma_gradient=calculate_normal_from_sigma_gradient,
            calculate_normal_from_sigma_gradient_surface=calculate_normal_from_sigma_gradient_surface,
            calculate_normal_from_depth_gradient=calculate_normal_from_depth_gradient,
            **render_kwargs_train
        )

        optimizer.zero_grad()

        # 2. calculate loss
        # 1) rendering loss
        rgb = result['color_map']
        loss_render = mse_loss(rgb, target_rgb)
        if 'color_map0' in result:
            loss_render0 = mse_loss(result['color_map0'], target_rgb)
            loss_render = loss_render + loss_render0

        # 1) radiance loss
        radiance = result['radiance_map']
        loss_render_radiance = mse_loss(radiance, target_rgb)
        if 'radiance_map0' in result:
            loss_render_radiance0 = mse_loss(result['radiance_map0'], target_rgb)
            loss_render_radiance = loss_render_radiance + loss_render_radiance0

        # 2) albedo render loss
        loss_albedo_render = mse_loss(result['albedo_map'], target_chromaticity)
        if 'albedo_map0' in result:
            loss_albedo_render0 = mse_loss(result['albedo_map0'], target_chromaticity)
            loss_albedo_render = loss_albedo_render + loss_albedo_render0

        loss_depth = 0
        if args.infer_depth:
            loss_depth = mse_loss(result['inferred_depth_map'], result['depth_map'].detach())
            if 'depth_map0' in result:
                loss_depth += mse_loss(result['inferred_depth_map'], result['depth_map0'].detach())

        # 3) Normal render loss

        # normal from gt
        result["ground_truth_normal"] = normalize(2 * target_info["normal"] - 1, dim=-1)

        def calculate_normal_loss(key_name, target="ground_truth_normal", loss_fn=mse_loss):
            if key_name not in result:
                print("Key %s not in result" % key_name)
                return 0

            loss_normal_from_target = loss_fn(result[key_name], result[target])
            if key_name+'0' in result:
                if target+'0' in result:
                    loss_normal_from_target += loss_fn(result[key_name + '0'], result[target+'0'])
                else:
                    loss_normal_from_target += loss_fn(result[key_name + '0'], result[target])
            return loss_normal_from_target

        # inferred & gt / sigma
        loss_inferred_normal = 0
        if args.infer_normal and i >= args.N_iter_ignore_normal:
            loss_inferred_normal = calculate_normal_loss("inferred_normal_map", args.infer_normal_target)

        # total_loss = loss_render_radiance
        if i < args.N_iter_ignore_normal:
            total_loss = args.beta_radiance_render * loss_render_radiance
        else:
            total_loss = args.beta_radiance_render * loss_render_radiance \
                         + args.beta_inferred_depth * loss_depth + args.beta_inferred_normal * loss_inferred_normal
                    #args.beta_render * loss_render + args.beta_albedo_render * loss_albedo_render + args.beta_inferred_depth * loss_depth
                      # + args.beta_inferred_normal * loss_inferred_normal + args.beta_radiance_render * loss_render_radiance \

        if i % args.summary_step == 0:
            writer.add_scalar('Loss/Total_Loss', total_loss, i)
            writer.add_scalar('Loss/Loss_render', loss_render, i)

            writer.add_scalar('Loss/Loss_albedo_render', loss_albedo_render, i)
            writer.add_scalar('Loss/Loss_radiance_render', loss_render_radiance, i)

            for normal_key in normal_target_keys:
                loss_from_gt = calculate_normal_loss(normal_key)
                writer.add_scalar('Loss_normal/%s'%normal_key, loss_from_gt, i)

            if args.infer_depth:
                writer.add_scalar('Loss/Loss_depth', loss_depth, i)

            if args.infer_normal:
                writer.add_scalar('Loss/inferred_normal', loss_inferred_normal, i)
                loss_from_gt = calculate_normal_loss("inferred_normal_map")
                writer.add_scalar('inferred_normal_from_gt/%s' % "inferred_normal_map", loss_from_gt, i)

        total_loss.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate

        # Export weight
        if i % args.i_weights == 0:
            path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
            save_target = {
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            if args.infer_depth:
                save_target['depth_mlp'] = render_kwargs_train['depth_mlp'].state_dict()
            if args.infer_normal:
                save_target['normal_mlp'] = render_kwargs_train['normal_mlp'].state_dict()
            torch.save(save_target, path)
            print('Saved checkpoints at', path)

        # export images
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)

            for var in grad_vars:
                var.requires_grad = False

            # with torch.no_grad():
            poses = torch.Tensor(dataset_val.poses).to(device)
            render_decomp_path_results = render_decomp_path(
                poses, hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir,
                render_factor=4, init_basecolor=dataset.init_basecolor
            )

            def add_image_to_writer(key_name):
                if key_name not in render_decomp_path_results:
                    return
                stacked_images = render_decomp_path_results[key_name]
                if len(stacked_images.shape) != 4:
                    stacked_images = np.expand_dims(stacked_images, -1)
                writer.add_images('test/inferred/%s' % key_name, stacked_images.transpose((0, 3, 1, 2)), i)

            # show_result_keys = ["rgb", "radiance", "albedo", "roughness", "specular", "normal", "inferred_normal"]
            show_result_keys = list(render_decomp_path_results.keys())
            for key in show_result_keys:
                add_image_to_writer(key)

            # writer.add_images('test/inferred/rgb', rgbs.transpose((0, 3, 1, 2)), i)
            # disps = np.expand_dims(disps, -1)
            # writer.add_images('test/inferred/disps', disps.transpose((0, 3, 1, 2)), i)
            # writer.add_images('test/inferred/albedo', albedos.transpose((0, 3, 1, 2)), i)
            # writer.add_images('test/inferred/direct_illumination', direct_illuminations.transpose((0, 3, 1, 2)), i)
            # writer.add_images('test/inferred/indirect_illumination', indirect_illuminations.transpose((0, 3, 1, 2)), i)
            # writer.add_images('test/inferred/illumination', illuminations.transpose((0, 3, 1, 2)), i)
            # writer.add_images('test/inferred/normals', normals.transpose((0, 3, 1, 2)), i)
            # roughness = np.expand_dims(roughness, -1)
            # writer.add_images('test/inferred/roughness', roughness.transpose((0, 3, 1, 2)), i)
            # writer.add_images('test/inferred/speculars', speculars.transpose((0, 3, 1, 2)), i)
            # if args.infer_normal:
            #     writer.add_images('test/inferred/inferred_normals', inferred_normals.transpose((0, 3, 1, 2)), i)

            for var in grad_vars:
                var.requires_grad = True

            logger_export.info('Saved test set')

        global_step += 1


if __name__ == '__main__':
    train()
    #test_autograd()
    #test_base_color()
    #test_parser()
