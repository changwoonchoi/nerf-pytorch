from nerf_models.nerf_renderer import render, render_path
from nerf_models.nerf import create_nerf
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)

from utils.label_utils import *
from config_parser import recursive_config_parser
from miscellaneous.test_dataset_speed import *

from utils.generator_utils import *
from utils.timing_utils import *
from run_nerf_eval import nerf_eval
from run_nerf_train import train


def test_marching_cube():
    parser = recursive_config_parser()
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

    # writer = SummaryWriter(log_dir=os.path.join(basedir, expname))

    if args.instance_mask:
        label_encoder = get_label_encoder(dataset.instance_color_list, args.instance_label_encoding, args.instance_label_dimension)
        args.instance_label_dimension = label_encoder.get_dimension()
    else:
        args.instance_label_dimension = 0

    _, render_kwargs_test, start, _, _ = create_nerf(args)
    bds_dict = dataset.get_near_far_plane()
    render_kwargs_test.update(bds_dict)

    if args.instance_mask:
        is_instance_label_logit = isinstance(label_encoder, OneHotLabelEncoder) and (args.CE_weight_type != "mse")
        render_kwargs_test["is_instance_label_logit"] = is_instance_label_logit
    testsavedir = os.path.join(basedir, expname, 'render_only_{:06d}'.format(start))
    os.makedirs(testsavedir, exist_ok=True)

    with torch.no_grad():
        _, _, _, _ = render_path(
            torch.Tensor(dataset.poses).to(device), hwf, K, args.chunk, render_kwargs_test, savedir=testsavedir,
            label_encoder=label_encoder, render_factor=1
        )

    if args.extract_mesh:
        ###################
        # TODO: move parameter to config file
        N_grid = 256
        threshold = 0.001
        bound = 4.5
        ###################
        net_query_fn = render_kwargs_test['network_query_fn']
        net_fn = render_kwargs_test['network_fine']
        sigma = query(N_grid, bound, args.chunk, net_query_fn, net_fn)
        mesh = march_cubes(sigma.cpu().numpy(), grid_num=N_grid, th=threshold)
        mesh.export(os.path.join(testsavedir, 'mesh_bound={}_th={}.obj'.format(bound, threshold)))


if __name__ == '__main__':
    parser = recursive_config_parser()
    args = parser.parse_args()
    args.device = device

    if args.render_only:
        print("This is [Evaluation]")
        nerf_eval(args)
    else:
        print("This is [Training]")
        train(args)
