# import os
#
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = str(0)
import matplotlib
matplotlib.use('TkAgg')

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
from utils.math_utils import *


def test():
    raise NotImplementedError


def train():
    parser = recursive_config_parser()
    args = parser.parse_args()
    args.device = device

    if args.expname is None:
        expname = args.config.split("/")[-1]
        expname = expname.split(".")[0]
        args.expname = expname

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
            target_dataset.load_all_data(num_of_workers=1)
            if do_logging:
                logger_dataset.info(target_dataset)
            return target_dataset

        # load train and validation dataset
        load_params = {
            "image_scale": args.image_scale,
            "load_normal": True, #args.learn_normal_from_oracle,
            "load_roughness": True,
            "load_albedo": args.learn_albedo_from_oracle,
            "sample_length": args.sample_length,
            "coarse_radiance_number": args.coarse_radiance_number,
            "load_instance_label_mask": args.instance_mask
        }
        dataset = load_dataset_split("train", **load_params)
        dataset_val = load_dataset_split("test", skip=10, **load_params)
        dataset_test = load_dataset_split("test", skip=1, **load_params)

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
        dataset_test.to_tensor(args.device)

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
        is_instance_label_logit = False
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
        # TODO: sample generator returns image patch
        if use_batching:
            sample_generator = sample_generator_all_image_merged(dataset, batch_size=batch_size)
        else:
            sample_generator = sample_generator_single_image(dataset, batch_size=batch_size,
                                                             precrop_iters=args.precrop_iters,
                                                             precrop_frac=args.precrop_frac, initial_iters=start,
                                                             ray_sample=args.ray_sample)

    # (5) Main train loop
    K = dataset.get_focal_matrix()
    N_iters = args.N_iter + 1

    # export ground truth image
    img_gt = dataset_val.images.permute((0, 3, 1, 2))
    writer.add_images('test/gt_rgb', img_gt, 0)

    for k in range(args.coarse_radiance_number):
        img_gt_k = dataset_val.get_coarse_images(k+1)
        writer.add_images('test/gt_rgb_coarse_%d' % (k+1), img_gt_k, 0)

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

    normal_target_keys = [
        "normal_map_from_sigma_gradient",
        "normal_map_from_sigma_gradient_surface",
        "normal_map_from_depth_gradient",
        "normal_map_from_depth_gradient_direction",
        "normal_map_from_depth_gradient_epsilon",
        "normal_map_from_depth_gradient_direction_epsilon"
    ]

    assert args.calculating_normal_type in normal_target_keys + ["ground_truth"]

    """
    render_kwargs_test["calculate_normal_from_sigma_gradient"] = args.calculate_all_analytic_normals
    render_kwargs_test["calculate_normal_from_sigma_gradient_surface"] = args.calculate_all_analytic_normals
    render_kwargs_test["calculate_normal_from_depth_gradient"] = args.calculate_all_analytic_normals
    render_kwargs_test["calculate_normal_from_depth_gradient_direction"] = args.calculate_all_analytic_normals
    render_kwargs_test["calculate_normal_from_depth_gradient_epsilon"] = args.calculate_all_analytic_normals
    render_kwargs_test["calculate_normal_from_depth_gradient_direction_epsilon"] = args.calculate_all_analytic_normals
    render_kwargs_test["epsilon"] = 0.01
    render_kwargs_test["epsilon_direction"] = 0.005
    render_kwargs_train["epsilon"] = 0.01
    render_kwargs_train["epsilon_direction"] = 0.005
    """
    render_kwargs_test["target_normal_map_for_radiance_calculation"] = args.calculating_normal_type
    render_kwargs_train["target_normal_map_for_radiance_calculation"] = args.calculating_normal_type
    render_kwargs_test["epsilon"] = 0.01
    render_kwargs_test["epsilon_direction"] = 0.005
    render_kwargs_train["epsilon"] = 0.01
    render_kwargs_train["epsilon_direction"] = 0.005

    # we will not infer normal
    # if args.infer_normal:
    #     assert args.infer_normal_target in normal_target_keys + ["ground_truth_normal"]

    hemisphere_samples = get_hemisphere_samples(args.N_hemisphere_sample_sqrt)
    hemisphere_samples = torch.Tensor(hemisphere_samples).to(args.device)

    for i in trange(start, N_iters):
        # sample rgb and rays from sample generator
        # target_info, rays_o, rays_d = next(sample_generator)  # 3 x (N_rand, 3)
        target_info, rays_o, rays_d, neigh_info, rays_o_neigh, rays_d_neigh = next(sample_generator)  # (N_rand, 3), (N_rand, 8, 3)

        target_rgb = target_info["rgb"]
        if args.learn_albedo_from_oracle:
            target_chromaticity = target_info["albedo"]
        else:
            target_chromaticity = target_rgb / (torch.linalg.norm(target_rgb, dim=-1, keepdim=True) + 1e-10)
        batch_rays = torch.stack([rays_o, rays_d], 0)  # (2, N_rand, 3)
        batch_rays_neigh = None
        if args.ray_sample == "patch":
            batch_rays_neigh = torch.stack([rays_o_neigh.reshape((-1, 3)), rays_d_neigh.reshape((-1, 3))], 0)  # (2, N_rand * 8, 3)

        #####  Core optimization loop  #####
        if args.calculate_all_analytic_normals:
            calculate_normal_from_depth_gradient = (i % args.summary_step == 0)
            calculate_normal_from_sigma_gradient = (i % args.summary_step == 0)
            calculate_normal_from_sigma_gradient_surface = (i % args.summary_step == 0)
            calculate_normal_from_depth_gradient_direction = (i % args.summary_step == 0)
            calculate_normal_from_depth_gradient_epsilon = (i % args.summary_step == 0)
            calculate_normal_from_depth_gradient_direction_epsilon = (i % args.summary_step == 0)
        else:
            calculate_normal_from_depth_gradient = False
            calculate_normal_from_sigma_gradient = False
            calculate_normal_from_sigma_gradient_surface = False
            calculate_normal_from_depth_gradient_direction = False
            calculate_normal_from_depth_gradient_epsilon = False
            calculate_normal_from_depth_gradient_direction_epsilon = False

        if i >= args.N_iter_ignore_normal:
            if args.infer_normal_target == "normal_map_from_sigma_gradient":
                calculate_normal_from_sigma_gradient = True
            elif args.infer_normal_target == "normal_map_from_sigma_gradient_surface":
                calculate_normal_from_sigma_gradient_surface = True
            elif args.infer_normal_target == "normal_map_from_depth_gradient":
                calculate_normal_from_depth_gradient = True
            elif args.infer_normal_target == "normal_map_from_depth_gradient_direction":
                calculate_normal_from_depth_gradient_direction = True
            elif args.infer_normal_target == "normal_map_from_depth_gradient_epsilon":
                calculate_normal_from_depth_gradient_epsilon = True
            elif args.infer_normal_target == "normal_map_from_depth_gradient_direction_epsilon":
                calculate_normal_from_depth_gradient_direction_epsilon = True

        # 1. render sample
        result = render_decomp(
            dataset.height, dataset.width, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,
            init_basecolor=dataset.init_basecolor,
            calculate_normal_from_sigma_gradient=calculate_normal_from_sigma_gradient,
            calculate_normal_from_sigma_gradient_surface=calculate_normal_from_sigma_gradient_surface,
            calculate_normal_from_depth_gradient=calculate_normal_from_depth_gradient,
            calculate_normal_from_depth_gradient_direction=calculate_normal_from_depth_gradient_direction,
            calculate_normal_from_depth_gradient_epsilon=calculate_normal_from_depth_gradient_epsilon,
            calculate_normal_from_depth_gradient_direction_epsilon=calculate_normal_from_depth_gradient_direction_epsilon,
            gt_values=target_info,
            hemisphere_samples=hemisphere_samples,
            approximate_radiance=i>=args.N_iter_ignore_approximated_radiance,
            **render_kwargs_train
        )

        if args.ray_sample == "patch":
            with torch.no_grad():
                result_neigh = render_decomp(
                    dataset.height, dataset.width, K, chunk=args.chunk, rays=batch_rays_neigh, verbose=i < 10, retraw=True,
                    init_basecolor=dataset.init_basecolor,
                    is_neighbor=True,
                    approximate_radiance=i >= args.N_iter_ignore_approximated_radiance,
                    **render_kwargs_train
                )


        def calculate_loss(key_name, target="ground_truth_normal", loss_fn=mse_loss):
            if key_name not in result:
                # print("Key %s not in result" % key_name)
                return 0

            if not isinstance(target, str):
                loss_from_target = loss_fn(result[key_name], target)

                if key_name+'0' in result:
                    loss_from_target += loss_fn(result[key_name + '0'], target)
            else:
                loss_from_target = loss_fn(result[key_name], result[target])
                if key_name+'0' in result:
                    if target+'0' in result:
                        loss_from_target += loss_fn(result[key_name + '0'], result[target+'0'])
                    else:
                        loss_from_target += loss_fn(result[key_name + '0'], result[target])
            return loss_from_target

        # normal from gt
        result["ground_truth_normal"] = normalize(2 * target_info["normal"] - 1, dim=-1)

        # 2. calculate loss

        # 0) approximated radiance loss
        loss_render = calculate_loss("color_map", target_rgb)

        # 1) radiance loss
        loss_render_radiance = calculate_loss("radiance_map", target_rgb)

        # 1-A) coarse radiance loss (for prefiltered env-map)
        loss_render_coarse_radiance = []
        for k in range(args.coarse_radiance_number):
            loss_render_radiance_i = calculate_loss("radiance_map_%d" % (k+1), target_info["rgb_%d" % (k+1)])
            loss_render_coarse_radiance.append(loss_render_radiance_i)

        # 2) albedo render loss
        loss_albedo_render = calculate_loss("albedo_map", target_chromaticity)
        # loss_roughness_render = calculate_loss("roughness_map", target_info.get("roughness", 1.0))

        # 3) Depth map if required
        loss_depth = 0
        if args.infer_depth and i >= args.N_iter_ignore_depth:
            # according to NeRV paper
            # 3-1) point from camera
            loss_depth = mse_loss(result['inferred_depth_map'], result['depth_map'].detach())

            # 3-2) point in volume
            # select first hit point & random direction align with normal
            normal_map = result["ground_truth_normal"]

            expected_points = rays_o + rays_d * result['depth_map'][..., None]
            expected_points = expected_points.detach()

            random_direction = 2 * torch.rand(*rays_d.shape) - 1
            normal_dot = torch.sum(random_direction * normal_map, dim=-1)
            random_direction = torch.sign(normal_dot)[..., None] * random_direction
            random_direction = F.normalize(random_direction, dim=-1)

            random_points = torch.stack([expected_points.reshape((-1, 3)), random_direction.reshape((-1, 3))], 0)

            random_points = random_points[:, 0:args.N_depth_random_volume, :]
            # with torch.no_grad():
            result_random_volume = render_decomp(
                dataset.height, dataset.width, K, chunk=args.chunk, rays=random_points, verbose=i < 10, retraw=True,
                init_basecolor=dataset.init_basecolor,
                is_depth_only=True,
                approximate_radiance=False,
                **render_kwargs_train
            )

            loss_depth_random = mse_loss(result_random_volume['inferred_depth_map'], result_random_volume['depth_map'].detach())
            loss_depth += loss_depth_random
            #if 'depth_map0' in result:
            #    loss_depth += mse_loss(result['inferred_depth_map'], result['depth_map0'].detach())

        # 3) Normal render loss


        # inferred & gt / sigma
        loss_inferred_normal = 0
        if args.infer_normal and i >= args.N_iter_ignore_normal:
            loss_inferred_normal = calculate_loss("inferred_normal_map", args.infer_normal_target)

        # 4) instance render loss
        loss_instance = 0
        if args.instance_mask:
            loss_instance = label_encoder.error(
                output_encoded_label=result['instance_map'],
                target_label=target_info['label'],
                CE_weight_type=args.CE_weight_type
            )
            if 'instance_map0' in result:
                loss_instance0 = label_encoder.error(
                    output_encoded_label=result['instance_map0'],
                    target_label=target_info['label'],
                    CE_weight_type=args.CE_weight_type
                )
                loss_instance += loss_instance0

        # 5) smoothness loss
        loss_smooth_roughness = 0
        loss_smooth_albedo = 0
        loss_smooth_irradiance = 0

        if args.ray_sample == "patch":
            if args.smooth_weight_type == "color":
                smooth_weight = torch.exp(-args.smooth_weight_decay * torch.norm(neigh_info['rgb'] - target_info['rgb'].view([-1, 1, 3]), 2, -1))
            elif args.smooth_weight_type == 'chrom':
                smooth_weight = torch.exp(
                    -args.smooth_weight_decay * torch.norm(
                        normalize(neigh_info["rgb"], dim=-1) - normalize(target_info['rgb'].view([-1, 1, 3]), dim=-1), 2, -1
                    )
                )
            elif args.smooth_weight_type == 'normal':
                smooth_weight = torch.exp(-args.smooth_weight_decay * torch.norm(neigh_info['normal'] - target_info['normal'].view([-1, 1, 3]), 2, -1))
            elif args.smooth_weight_type == 'all':
                smooth_weight = torch.exp(-args.smooth_weight_decay * torch.norm(
                    torch.cat([neigh_info['rgb'], neigh_info['normal']], dim=-1) - torch.cat([target_info['rgb'], target_info['normal']], dim=-1).view([-1, 1, 6]), 2, -1
                ))
            else:
                raise ValueError

            def calculate_smooth_loss(key_name, norm_p=1):
                if len(result[key_name].shape) == 1:
                    result_p = result[key_name][..., None]
                    result_p_neighs = result_neigh[key_name][..., None]
                else:
                    result_p = result[key_name]
                    result_p_neighs = result_neigh[key_name]
                result_p_neighs = result_p_neighs.reshape([-1, 8, result_p_neighs.shape[-1]])

                loss_smooth = result_p[:, None, :] - result_p_neighs
                loss_smooth = torch.norm(loss_smooth, norm_p, -1)
                loss_smooth = torch.mean(smooth_weight * loss_smooth)

                if key_name + '0' in result:
                    loss_smooth += calculate_smooth_loss(key_name + '0', norm_p)
                return loss_smooth

            # 5-1 roughness smooth
            if args.roughness_smooth:
                loss_smooth_roughness = calculate_smooth_loss("roughness_map")

            # 5-2 albedo smooth
            if args.albedo_smooth:
                loss_smooth_albedo = calculate_smooth_loss("albedo_map")

            # 5-3 irradiance smooth
            if args.irradiance_smooth:
                loss_smooth_irradiance = calculate_smooth_loss("irradiance_map")

        # 6) instance-wise constant loss (for test)
        loss_instancewise_constant_albedo = 0
        loss_instancewise_constant_irradiance = 0
        if args.instance_mask:
            expected_label = torch.argmax(result['instance_map'], dim=-1)  # (N_rand, )
            for instance_idx in range(args.instance_label_dimension - 1):  # ignore last label (last label is for others)
                instance_mask = expected_label == instance_idx
                if args.albedo_instance_constant:
                    instance_albedos = result['albedo_map'][instance_mask]
                    instancewise_albedo_std = torch.mean(torch.std(instance_albedos, dim=0, unbiased=True))
                    if not torch.isnan(instancewise_albedo_std):
                        loss_instancewise_constant_albedo += instancewise_albedo_std
                if args.irradiance_instance_constant:
                    instance_irradiances = result['irradiance_map'][instance_mask]
                    instancewise_irradiance_std = torch.std(instance_irradiances, dim=0, unbiased=True)
                    if not torch.isnan(instancewise_irradiance_std):
                        loss_instancewise_constant_irradiance += instancewise_irradiance_std
            if 'instance_map0' in result:
                expected_label0 = torch.argmax(result['instance_map0'], dim=-1)
                for instance_idx in range(args.instance_label_dimension - 1):
                    instance_mask0 = expected_label0 == instance_idx
                    if args.albedo_instance_constant:
                        instance_albedos0 = result['albedo_map0'][instance_mask0]
                        instancewise_albedo_std0 = torch.mean(torch.std(instance_albedos0, dim=0, unbiased=True))
                        if not torch.isnan(instancewise_albedo_std0):
                            loss_instancewise_constant_albedo += instancewise_albedo_std0
                    if args.irradiance_instance_constant:
                        instance_irradiances0 = result['irradiance_map0'][instance_mask0]
                        instancewise_irradiance_std0 = torch.std(instance_irradiances0, dim=0, unbiased=True)
                        if not torch.isnan(instancewise_irradiance_std0):
                            loss_instancewise_constant_irradiance += instancewise_irradiance_std0
        loss_instancewise_constant = loss_instancewise_constant_albedo + loss_instancewise_constant_irradiance

        # Final loss
        # (a) radiance loss
        total_loss = args.beta_radiance_render * loss_render_radiance
        for k in range(args.coarse_radiance_number):
            total_loss += args.beta_radiance_render * loss_render_coarse_radiance[k]

        # (b) normal loss
        if i>= args.N_iter_ignore_normal:
            total_loss += args.beta_inferred_normal * loss_inferred_normal
                    #args.beta_render * loss_render + args.beta_albedo_render * loss_albedo_render + args.beta_inferred_depth * loss_depth
                      # + args.beta_inferred_normal * loss_inferred_normal + args.beta_radiance_render * loss_render_radiance \
        if i>= args.N_iter_ignore_approximated_radiance:
            total_loss += args.beta_render * loss_render
            #total_loss += 0.1 * args.beta_albedo_render * loss_albedo_render
        #else:
            #total_loss += args.beta_albedo_render * loss_albedo_render

        # (c) smoothness loss
        if i >= args.N_iter_ignore_smooth:
            total_loss += args.beta_roughness_smooth * loss_smooth_roughness
            total_loss += args.beta_irradiance_smooth * loss_smooth_irradiance
            total_loss += args.beta_albedo_smooth * loss_smooth_albedo

        # (d) instance loss
        if args.instance_mask:
            total_loss += args.beta_instance * loss_instance

        # (e) instance-wise constant loss (for test)
        if i >= args.N_iter_ignore_instancewise_constant:
            total_loss += args.beta_instancewise_constant * loss_instancewise_constant

        # (f) depth loss
        if i >= args.N_iter_ignore_depth:
            total_loss += args.beta_inferred_depth * loss_depth

        if i % args.summary_step == 0:
            writer.add_scalar('Loss/Total_Loss', total_loss, i)
            writer.add_scalar('Loss/Loss_render', loss_render, i)

            writer.add_scalar('Loss/Loss_albedo_render', loss_albedo_render, i)
            # writer.add_scalar('Loss/Loss_roughness_render', loss_roughness_render, i)

            writer.add_scalar('Loss/Loss_radiance_render', loss_render_radiance, i)

            for k in range(args.coarse_radiance_number):
                writer.add_scalar('Loss/Loss_radiance_render_coarse_%d' % (k+1), loss_render_coarse_radiance[k], i)

            if args.calculate_all_analytic_normals:
                for normal_key in normal_target_keys:
                    loss_from_gt = calculate_loss(normal_key)
                    writer.add_scalar('Loss_normal/%s' % normal_key, loss_from_gt, i)

            if args.infer_depth:
                writer.add_scalar('Loss/Loss_depth', loss_depth, i)

            if args.infer_normal:
                writer.add_scalar('Loss_normal/inferred_normal', loss_inferred_normal, i)
                loss_from_gt = calculate_loss("inferred_normal_map", "ground_truth_normal")
                writer.add_scalar('Loss_normal/inferred_normal_from_gt', loss_from_gt, i)

            writer.add_scalar('Loss/Loss_roughness_smooth', loss_smooth_roughness, i)
            writer.add_scalar('Loss/Loss_irradiance_smooth', loss_smooth_irradiance, i)
            writer.add_scalar('Loss/Loss_albedo_smooth', loss_smooth_albedo, i)
            if args.instance_mask:
                writer.add_scalar('Loss/Loss_instance', loss_instance, i)
                writer.add_scalar('Loss/Loss_instancewise_constant', loss_instancewise_constant, i)
                if args.albedo_instance_constant:
                    writer.add_scalar('Loss/Loss_instancewise_constant_albedo', loss_instancewise_constant_albedo, i)
                if args.irradiance_instance_constant:
                    writer.add_scalar('Loss/Loss_instancewise_constant_irradiance', loss_instancewise_constant_irradiance, i)

        # total_loss = args.beta_radiance_render * loss_render_radiance
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000

        def set_lr(name, start_count):
            for param_group in optimizer.param_groups:
                if param_group['name'] == name and global_step > start_count:
                    new_lrate = args.lrate * (decay_rate ** ((global_step - start_count) / decay_steps))
                    param_group['lr'] = new_lrate

        set_lr("coarse", 0)
        set_lr("fine", 0)
        set_lr("depth", args.N_iter_ignore_depth)
        set_lr("normal", args.N_iter_ignore_normal)

        # new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = new_lrate

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
            if args.use_environment_map:
                save_target['env_map'] = render_kwargs_train['env_map'].emission

            torch.save(save_target, path)
            print('Saved checkpoints at', path)

        # export images
        if i % args.i_testset == 0 and i > 0:
            testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
            os.makedirs(testsavedir, exist_ok=True)

            for param_group in optimizer.param_groups:
                for var in param_group['params']:
                    var.requires_grad = False

            # with torch.no_grad():
            # poses = torch.Tensor(dataset_val.poses).to(device)
            if i % 50000 == 0:
                render_decomp_path_results = render_decomp_path(
                    dataset_test, hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir,
                    render_factor=1, init_basecolor=dataset.init_basecolor,
                    calculate_normal_from_depth_map=args.calculate_all_analytic_normals,
                    use_instance=use_instance_mask, label_encoder=label_encoder,
                    hemisphere_samples=hemisphere_samples,
                    approximate_radiance=True
                )
            else:
                render_decomp_path_results = render_decomp_path(
                    dataset_val, hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir,
                    render_factor=4, init_basecolor=dataset.init_basecolor,
                    calculate_normal_from_depth_map=args.calculate_all_analytic_normals,
                    use_instance=use_instance_mask, label_encoder=label_encoder,
                    hemisphere_samples=hemisphere_samples,
                    approximate_radiance=True
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

            for param_group in optimizer.param_groups:
                for var in param_group['params']:
                    var.requires_grad = True

            logger_export.info('Saved test set')

        global_step += 1


if __name__ == '__main__':
    train()
