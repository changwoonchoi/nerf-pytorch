import numpy as np
import torch

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

from config_parser import export_config
from nerf_models.nerf_renderer import render, render_path
from nerf_models.nerf import create_nerf

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
from utils.label_utils import *
from utils.mesh_utils import *
from miscellaneous.test_dataset_speed import *


def test():
    parser = config_parser()
    args = parser.parse_args()
    args.device = device

    dataset = load_dataset(args.dataset_type, args.datadir, split='test', skip=args.testskip, sample_length=args.sample_length)
    dataset.load_instance_label_mask = args.instance_mask
    dataset.load_all_data(num_of_workers=8)

    hwf = [dataset.height, dataset.width, dataset.focal]
    K = dataset.get_focal_matrix()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    dataset.to_tensor(args.device)

    basedir = args.basedir
    expname = args.expname

    if args.instance_mask:
        label_encoder = get_label_encoder(dataset.instance_color_list, args.instance_label_encoding)
        args.instance_label_dimension = label_encoder.get_dimension()
    else:
        args.instance_label_dimension = 0

    _, render_kwargs_test, start, _, _ = create_nerf(args)
    bds_dict = dataset.get_near_far_plane()
    render_kwargs_test.update(bds_dict)

    testsavedir = os.path.join(basedir, expname, 'render_only_{:06d}'.format(start))
    os.makedirs(testsavedir, exist_ok=True)

    with torch.no_grad():
        _, _, _, _ = render_path(
            torch.Tensor(dataset.poses).to(device), hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir,
            label_encoder=label_encoder, render_factor=4
        )

    if args.extract_mesh:
        ###################
        # TODO: move parameter to config file
        N = 256
        threshold = 0.001
        bound = 4.5
        ###################
        net_query_fn = render_kwargs_test['network_query_fn']
        net_fn = render_kwargs_test['network_fine']
        sigma = query(N, bound, args.chunk, net_query_fn, net_fn)
        mesh = march_cubes(sigma.cpu().numpy(), grid_num=N, th=threshold)
        mesh.export(os.path.join(testsavedir, 'mesh_bound={}_th={}.obj'.format(bound, threshold)))


def train():
    parser = config_parser()
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
        dataset = load_dataset_split("train", sample_length=args.sample_length)
        dataset_val = load_dataset_split("val", skip=5, sample_length=args.sample_length)

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
            label_encoder = get_label_encoder(dataset.instance_color_list, args.instance_label_encoding)
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
                fixed_CE_weight=args.fixed_CE_weight
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
                fixed_CE_weight=args.fixed_CE_weight
            )
            instance_loss = instance_loss + instance_loss0

        alpha = args.instance_loss_weight
        loss = img_loss + alpha * instance_loss
        if i % 100 == 0:
            writer.add_scalar('Loss/rgb_MSE', img_loss, i)
            writer.add_scalar('Loss/instance_loss', instance_loss, i)
            writer.add_scalar('Loss/total_loss', loss, i)

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
                rgbs, disps, instances, instance_colors = render_path(torch.Tensor(dataset_val.poses).to(device),
                                                                      hwf, K, args.chunk, render_kwargs_test,
                                                                      gt_imgs=None, savedir=testsavedir,
                                                                      label_encoder=label_encoder, render_factor=4)
                writer.add_images('test/inferred_rgb', rgbs.transpose((0, 3, 1, 2)), i)
                disps = np.expand_dims(disps, -1)
                writer.add_images('test/inferred_disps', disps.transpose((0, 3, 1, 2)), i)

                if use_instance_mask:
                    writer.add_images('test/inferred_mask', instance_colors.transpose((0, 3, 1, 2)), i)

            logger_export.info('Saved test set')

        global_step += 1


if __name__ == '__main__':
    train()
