import numpy as np
import torch

from config_parser import export_config
from nerf_models.nerf_renderer import render, render_path
from nerf_models.nerf import create_nerf

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils.label_utils import *
from config_parser import recursive_config_parser
from dataset.dataset_interface import load_dataset_split
from miscellaneous.test_dataset_speed import *

from utils.generator_utils import *
from utils.timing_utils import *


def train(args):
    # (0) Print train phase overview
    logger_dataset = load_logger("Dataset Info")
    logger_export = load_logger("Export Logger")
    use_instance_mask = args.instance_mask
    logger_dataset.info("Instance mask: " + str(use_instance_mask))
    logger_dataset.info("Instance mask encoding: " + str(args.instance_label_encoding))

    # (1) Load dataset
    with time_measure("[1] Data load"):
        # load train and validation dataset
        dataset = load_dataset_split(args, "train")
        dataset_val = load_dataset_split(args, "val", skip=5)

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
    with time_measure("[3] NeRF load"):
        # set instance label dimension
        label_encoder = None
        if use_instance_mask:
            label_encoder = get_label_encoder(dataset.instance_color_list, args.instance_label_encoding, args.instance_label_dimension)
            args.instance_label_dimension = label_encoder.get_dimension()
        else:
            args.instance_label_dimension = 0

        # create nerf model
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
        global_step = start

        # update near / far plane
        bds_dict = dataset.get_near_far_plane()
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        render_kwargs_test['decompose'] = args.render_decompose
        render_kwargs_test['alpha_th'] = args.alpha_th
        render_kwargs_test['decompose_target'] = [int(item) for item in args.decompose_target.split(',')]
        render_kwargs_test['decompose_mode'] = args.decompose_mode

        if use_instance_mask:
            is_instance_label_logit = isinstance(label_encoder, OneHotLabelEncoder) and (args.CE_weight_type != "mse")
            render_kwargs_train["is_instance_label_logit"] = is_instance_label_logit
            render_kwargs_test["is_instance_label_logit"] = is_instance_label_logit
        logger_render_options = load_logger("Render Kwargs")
        logs = ["[Train Render Kwargs (simple only)]"]
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
        result = render(dataset.height, dataset.width, K,
                                                  chunk=args.chunk, rays=batch_rays,
                                                  verbose=i < 10, retraw=True,
                                                  **render_kwargs_train)

        optimizer.zero_grad()
        rgb = result['rgb_map']
        img_loss = img2mse(rgb, target_rgb)
        if use_instance_mask:
            instance_loss = label_encoder.error(
                output_encoded_label=result['instance_map'],
                target_label=target_label,
                CE_weight_type=args.CE_weight_type
            )
        else:
            instance_loss = 0

        trans = result['raw'][..., -1]
        psnr = mse2psnr(img_loss)

        if 'rgb0' in result:
            img_loss0 = img2mse(result['rgb0'], target_rgb)
            img_loss = img_loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        if 'instance0' in result and use_instance_mask:
            instance_loss0 = label_encoder.error(
                output_encoded_label=result['instance0'],
                target_label=target_label,
                CE_weight_type=args.CE_weight_type
            )
            instance_loss = instance_loss + instance_loss0

        alpha = args.instance_loss_weight
        loss = img_loss + alpha * instance_loss
        if i % 100 == 0:
            # error in decoded space (0, 1, 2, ..., N-1) where N is number of instance
            instance_loss_decoded = label_encoder.error_in_decoded_space(output_encoded_label=result['instance_map'], target_label=target_label)
            writer.add_scalar('Loss/rgb_MSE', img_loss, i)
            writer.add_scalar('Loss/instance_loss', instance_loss, i)
            writer.add_scalar('Loss/total_loss', loss, i)
            writer.add_scalar('Loss/instance_loss_decoded', instance_loss_decoded, i)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

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
                rgbs, disps, instances, instance_colors = render_path(poses,
                                                                      hwf, K, args.chunk, render_kwargs_test,
                                                                      gt_imgs=None, savedir=testsavedir,
                                                                      label_encoder=label_encoder, render_factor=2)
                writer.add_images('test/inferred_rgb', rgbs.transpose((0, 3, 1, 2)), i)
                disps = np.expand_dims(disps, -1)
                writer.add_images('test/inferred_disps', disps.transpose((0, 3, 1, 2)), i)

                if use_instance_mask:
                    writer.add_images('test/inferred_mask', instance_colors.transpose((0, 3, 1, 2)), i)

            logger_export.info('Saved test set')

        global_step += 1