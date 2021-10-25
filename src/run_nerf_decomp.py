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


def test():
    raise NotImplementedError


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
        dataset = load_dataset_split("train", sample_length=args.sample_length, image_scale=args.image_scale)
        dataset_val = load_dataset_split("test", skip=10, sample_length=args.sample_length)

        # calculate base color
        dataset.get_base_color(
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

    for i in trange(start, N_iters):
        # sample rgb and rays from sample generator
        target_rgb, target_label, rays_o, rays_d = next(sample_generator)  # 3 x (N_rand, 3)
        batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)

        #####  Core optimization loop  #####

        # 1. render sample
        result = render_decomp(
            dataset.height, dataset.width, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,
            init_basecolor=dataset.init_basecolor, **render_kwargs_train
        )

        optimizer.zero_grad()

        # 2. calculate loss
        # 1) rendering loss
        rgb = result['color_map']
        loss_render = img2mse(rgb, target_rgb)
        if use_instance_mask:
            instance_loss = label_encoder.error(
                output_encoded_label=result['instance_map'],
                target_label=target_label,
                CE_weight_type=args.CE_weight_type
            )
        else:
            instance_loss = 0

        if 'color_map0' in result:
            loss_render0 = img2mse(result['color_map0'], target_rgb)
            loss_render = loss_render + loss_render0

        if 'instance0' in result and use_instance_mask:
            instance_loss0 = label_encoder.error(
                output_encoded_label=result['instance0'],
                target_label=target_label,
                CE_weight_type=args.CE_weight_type
            )
            instance_loss = instance_loss + instance_loss0

        instance_loss_weight = args.instance_loss_weight
        loss_render = loss_render + instance_loss_weight * instance_loss

        # 2) regularization loss
        albedo = result['albedo']  # (N_rays, N_samples + N_importance, 3)
        log_albedo = torch.log(albedo)
        log_init_basecolor = torch.log(dataset.init_basecolor)
        diff_albedo_cluster = log_albedo[..., None, :] - log_init_basecolor[None, None, ...]  # (N_rays, N_samples, n_cluster, 3)
        diff_albedo_cluster = torch.linalg.norm(diff_albedo_cluster, dim=-1).min(dim=-1).values
        loss_albedo_cluster = diff_albedo_cluster.sum() / torch.numel(diff_albedo_cluster)

        indirect_illumination_weight = result['indirect_illumination_weight']
        l1_indir_illum = torch.linalg.norm(indirect_illumination_weight, ord=1, dim=-1)
        l1_indir_illum = l1_indir_illum.sum() / torch.numel(l1_indir_illum)

        if 'indirect_illumination_weight0' in result:
            indirect_illumination_weight0 = result['indirect_illumination_weight0']
            l1_indir_illum0 = torch.linalg.norm(indirect_illumination_weight0, ord=1, dim=-1)
            l1_indir_illum0 = l1_indir_illum0.sum() / torch.numel(l1_indir_illum0)
            l1_indir_illum = l1_indir_illum + l1_indir_illum0

        loss_reg = args.beta_albedo_cluster * loss_albedo_cluster + args.beta_indirect_sparse * l1_indir_illum

        # 3) smooth prior
        # TODO: implement smooth prior
        smooth_prior_albedo = 0
        smooth_prior_indirect = 0

        loss_smooth = args.beta_smooth_albedo * smooth_prior_albedo + args.beta_smooth_indirect * smooth_prior_indirect

        total_loss = args.beta_render * loss_render + loss_reg + loss_smooth

        if i % args.summary_step == 0:
            writer.add_scalar('Loss/Total_Loss', total_loss, i)
            writer.add_scalar('Loss/Loss_render', args.beta_render * loss_render, i)
            writer.add_scalar('Loss/Loss_reg', loss_reg, i)
            writer.add_scalar('Loss/Loss_reg/albedo_cluster', loss_albedo_cluster * args.beta_albedo_cluster, i)
            writer.add_scalar('Loss/Loss_reg/indirect_sparsity', args.beta_indirect_sparse * l1_indir_illum, i)

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
            torch.save({
                'global_step': global_step,
                'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, path)
            print('Saved checkpoints at', path)

        # export images
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)

            with torch.no_grad():
                poses = torch.Tensor(dataset_val.poses).to(device)
                rgbs, albedos, direct_illuminations, indirect_illuminations, illuminations, disps = render_decomp_path(
                    poses, hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir,
                    render_factor=4, init_basecolor=dataset.init_basecolor
                )
                writer.add_images('test/inferred/rgb', rgbs.transpose((0, 3, 1, 2)), i)
                disps = np.expand_dims(disps, -1)
                writer.add_images('test/inferred/disps', disps.transpose((0, 3, 1, 2)), i)
                writer.add_images('test/inferred/albedo', albedos.transpose((0, 3, 1, 2)), i)
                writer.add_images('test/inferred/direct_illumination', direct_illuminations.transpose((0, 3, 1, 2)), i)
                writer.add_images('test/inferred/indirect_illumination', indirect_illuminations.transpose((0, 3, 1, 2)), i)
                writer.add_images('test/inferred/illumination', illuminations.transpose((0, 3, 1, 2)), i)

            logger_export.info('Saved test set')

        global_step += 1


if __name__ == '__main__':
    train()
    #test_base_color()
