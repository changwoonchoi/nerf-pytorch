import numpy as np
import torch


from config_parser import export_config
from nerf_models.nerf_renderer import render_path
from nerf_models.nerf import create_nerf

from torch.utils.tensorboard import SummaryWriter
from utils.label_utils import *
from dataset.dataset_interface import load_dataset_split
from miscellaneous.test_dataset_speed import *

from utils.generator_utils import *
from utils.timing_utils import *


def nerf_eval(args):
    use_instance_mask = args.instance_mask
    logger_export = load_logger("Export Logger")
    logger_dataset = load_logger("Dataset Info")

    # (1) Load dataset
    with time_measure("[1] Data load"):
        # load train and validation dataset
        dataset_test = load_dataset_split(args, "val", skip=25)
        logger_dataset.info(dataset_test)

        hwf = [dataset_test.height, dataset_test.width, dataset_test.focal]

        # move data to GPU side
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        dataset_test.to_tensor(args.device)

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
            label_encoder = get_label_encoder(dataset_test.instance_color_list, args.instance_label_encoding,
                                              args.instance_label_dimension)
            args.instance_label_dimension = label_encoder.get_dimension()
        else:
            args.instance_label_dimension = 0

        # create nerf model
        _, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)

        # update near / far plane
        bds_dict = dataset_test.get_near_far_plane()
        render_kwargs_test.update(bds_dict)

        render_kwargs_test['decompose'] = args.render_decompose
        render_kwargs_test['alpha_th'] = args.alpha_th
        render_kwargs_test['instance_th'] = args.instance_th
        render_kwargs_test['decompose_target'] = [int(item) for item in args.decompose_target.split(',')]
        render_kwargs_test['decompose_mode'] = args.decompose_mode

        if use_instance_mask:
            is_instance_label_logit = isinstance(label_encoder, OneHotLabelEncoder) and (
                        args.CE_weight_type != "mse")
            render_kwargs_test["is_instance_label_logit"] = is_instance_label_logit
        logger_render_options = load_logger("Render Kwargs")
        logs = ["[Test Render Kwargs (simple only)]"]
        for k, v in render_kwargs_test.items():
            if isinstance(v, (str, float, int, bool)):
                logs += ["\t-%s : %s" % (k, str(v))]
            elif k == "decompose_target":
                logs += ["\t-%s : %s" % (k, str(v))]

        logger_render_options.info("\n".join(logs))

    # (5) Main train loop
    K = dataset_test.get_focal_matrix()
    N_iters = args.N_iter + 1

    testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(0))
    os.makedirs(testsavedir, exist_ok=True)

    with torch.no_grad():
        poses = torch.Tensor(dataset_test.poses).to(device)
        rgbs, disps, instances, instance_colors = render_path(poses,
                                                              hwf, K, args.chunk, render_kwargs_test,
                                                              gt_imgs=None, savedir=testsavedir,
                                                              label_encoder=label_encoder, render_factor=1)
        writer.add_images('test/inferred_rgb', rgbs.transpose((0, 3, 1, 2)), 0)
        disps = np.expand_dims(disps, -1)
        writer.add_images('test/inferred_disps', disps.transpose((0, 3, 1, 2)), 0)

        if use_instance_mask:
            writer.add_images('test/inferred_mask', instance_colors.transpose((0, 3, 1, 2)), 0)

    logger_export.info('Saved test set')
    return
