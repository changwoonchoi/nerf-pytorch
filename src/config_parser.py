import configargparse
import os
from pathlib import Path


def load_all_include(config_file):
	parser = config_parser()
	args = parser.parse_args("--config %s" % config_file)
	path = Path(config_file)

	include = []
	if args.include:
		include.append(os.path.join(path.parent, args.include))
		return include + load_all_include(os.path.join(path.parent, args.include))
	else:
		return include


def recursive_config_parser():
	parser = config_parser()
	args = parser.parse_args()
	include_files = load_all_include(args.config)

	include_files = list(reversed(include_files))
	parser = config_parser(default_files=include_files)
	return parser


def config_parser(default_files=None):
	if default_files is not None:
		parser = configargparse.ArgumentParser(default_config_files=default_files)
	else:
		parser = configargparse.ArgumentParser()

	parser.add_argument('--config', is_config_file=True, help='config file path')
	parser.add_argument('--include', type=str, default=None, help='config file path')

	parser.add_argument("--expname", type=str, help='experiment name')
	parser.add_argument("--basedir", type=str, default='./logs/', help='where to store ckpts and logs')
	parser.add_argument("--datadir", type=str, default='./data/llff/fern', help='input data directory')

	# training options
	parser.add_argument("--image_scale", type=float, default=1.0, help="image scale ex) 0.5 = half")
	parser.add_argument("--instance_mask", action="store_true", help='NeRF with instance mask')
	parser.add_argument("--instance_loss_weight", type=float, default=0.01, help='Instance loss weight')
	parser.add_argument("--instance_label_encoding", type=str, default="one_hot",
	                    help="how to encode instance label. one of single, one_hot, label_color")
	parser.add_argument("--instance_label_dimension", type=int, default=0, help="instance mask dimension")
	parser.add_argument("--use_instance_feature_layer", action="store_true", help='NeRF with instance_feature_layer(Zhi, 2021)')
	parser.add_argument("--use_basecolor_score_feature_layer", action="store_true", help='NeRF with basecolor score feature layer')
	parser.add_argument("--use_illumination_feature_layer", action="store_true", help='NeRF with illumination feature_layer(Zhi, 2021)')

	parser.add_argument("--N_iter", type=int, default=200000, help="Total iteration num")
	parser.add_argument("--netdepth", type=int, default=8, help='layers in network')
	parser.add_argument("--netwidth", type=int, default=256, help='channels per layer')
	parser.add_argument("--netdepth_fine", type=int, default=8, help='layers in fine network')
	parser.add_argument("--netwidth_fine", type=int, default=256, help='channels per layer in fine network')
	parser.add_argument("--N_rand", type=int, default=32 * 32 * 4,
	                    help='batch size (number of random rays per gradient step)')
	parser.add_argument("--CE_weight_type", type=str, default=None, help='weight type in CE Loss, bg_weakened/adaptive/equal or mse')

	parser.add_argument("--beta_sparse_base", type=float, default=1., help="")
	parser.add_argument("--beta_res", type=float, default=1., help="")
	parser.add_argument("--beta_mod", type=float, default=1., help="")
	parser.add_argument("--beta_indirect", type=float, default=1., help="")
	parser.add_argument("--beta_smooth_albedo", type=float, default=1., help="")
	parser.add_argument("--beta_smooth_indirect", type=float, default=1., help="")
	parser.add_argument("--beta_render", type=float, default=1.)
	parser.add_argument("--beta_albedo_cluster", type=float, default=1.)
	parser.add_argument("--beta_albedo_render", type=float, default=1.)
	parser.add_argument("--beta_indirect_sparse", type=float, default=1.)

	parser.add_argument("--lrate", type=float, default=5e-4, help='learning rate')
	parser.add_argument("--lrate_decay", type=int, default=250,
	                    help='exponential learning rate decay (in 1000 steps)')
	parser.add_argument("--chunk", type=int, default=1024 * 16,
	                    help='number of rays processed in parallel, decrease if running out of memory')
	parser.add_argument("--netchunk", type=int, default=1024 * 64,
	                    help='number of pts sent through network in parallel, decrease if running out of memory')
	parser.add_argument("--no_batching", action='store_true', help='only take random rays from 1 image at a time')
	parser.add_argument("--no_reload", action='store_true', help='do not reload weights from saved ckpt')
	parser.add_argument("--ft_path", type=str, default=None,
	                    help='specific weights npy file to reload for coarse network')

	# rendering options
	parser.add_argument("--N_samples", type=int, default=64, help='number of coarse samples per ray')
	parser.add_argument("--N_importance", type=int, default=0, help='number of additional fine samples per ray')
	parser.add_argument("--perturb", type=float, default=1., help='set to 0. for no jitter, 1. for jitter')
	parser.add_argument("--use_viewdirs", action='store_true', help='use full 5D input instead of 3D')
	parser.add_argument("--i_embed", type=int, default=0,
	                    help='set 0 for default positional encoding, -1 for none')
	parser.add_argument("--multires", type=int, default=10,
	                    help='log2 of max freq for positional encoding (3D location)')
	parser.add_argument("--multires_views", type=int, default=4,
	                    help='log2 of max freq for positional encoding (2D direction)')
	parser.add_argument("--raw_noise_std", type=float, default=0.,
	                    help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

	parser.add_argument("--render_only", action='store_true',
	                    help='do not optimize, reload weights and render out render_poses path')
	parser.add_argument("--render_test", action='store_true',
	                    help='render the test set instead of render_poses path')
	parser.add_argument("--render_factor", type=int, default=0,
	                    help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')
	parser.add_argument("--render_decompose", action='store_true', help="render decomposed instance in test phase")
	parser.add_argument("--alpha_th", type=float, default=.0, help='decompose alpha thredhold')
	parser.add_argument("--instance_th", type=float, default=.0, help='decompose instance thredhold')

	parser.add_argument("--decompose_target", type=str, default="0", help='decompose target instance ids')
	parser.add_argument("--decompose_mode", type=str, default="binary", help='decompose mode one of all or binary')

	# training options
	parser.add_argument("--precrop_iters", type=int, default=0, help='number of steps to train on central crops')
	parser.add_argument("--precrop_frac", type=float, default=.5, help='fraction of img taken for central crops')

	# test options
	parser.add_argument("--extract_mesh", action='store_true', help='extract mesh')

	# dataset options
	parser.add_argument("--dataset_type", type=str, default='llff', help='options: llff / blender / deepvoxels')
	parser.add_argument("--testskip", type=int, default=8,
	                    help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

	# clustering options
	parser.add_argument("--cluster_image_number", type=int, default=-1, help='how many images will be used for clustering? -1 to use all')
	parser.add_argument("--cluster_image_resize", type=float, default=0.5, help='resize image for clustering?')
	parser.add_argument("--cluster_init_number", type=int, default=8, help='initial cluster size')
	parser.add_argument("--cluster_number_lower_bound", type=int, default=4, help='minimum number of cluster')
	parser.add_argument("--cluster_merge_threshold", type=float, default=0.1, help='cluster merge threshold')

	# clevr options
	parser.add_argument("--sample_length", type=float, default=8, help='sampling length along ray')

	## deepvoxels flags
	parser.add_argument("--shape", type=str, default='greek',
	                    help='options : armchair / cube / greek / vase')

	## blender flags
	parser.add_argument("--white_bkgd", action='store_true',
	                    help='set to render synthetic data on a white bkgd (always use for dvoxels)')
	parser.add_argument("--half_res", action='store_true',
	                    help='load blender synthetic data at 400x400 instead of 800x800')

	## llff flags
	parser.add_argument("--factor", type=int, default=8, help='downsample factor for LLFF images')
	parser.add_argument("--no_ndc", action='store_true',
	                    help='do not use normalized device coordinates (set for non-forward facing scenes)')
	parser.add_argument("--lindisp", action='store_true',
	                    help='sampling linearly in disparity rather than depth')
	parser.add_argument("--spherify", action='store_true', help='set for spherical 360 scenes')
	parser.add_argument("--llffhold", type=int, default=8,
	                    help='will take every 1/N images as LLFF test set, paper uses 8')

	# logging/saving options
	parser.add_argument("--summary_step", type=int, default=100)
	parser.add_argument("--i_print", type=int, default=100,
	                    help='frequency of console printout and metric loggin')
	parser.add_argument("--i_img", type=int, default=500, help='frequency of tensorboard image logging')
	parser.add_argument("--i_weights", type=int, default=10000, help='frequency of weight ckpt saving')
	parser.add_argument("--i_testset", type=int, default=50000, help='frequency of testset saving')
	parser.add_argument("--i_video", type=int, default=50000, help='frequency of render_poses video saving')

	return parser


def export_config(args):
	# Create log dir and copy the config file
	basedir = args.basedir
	expname = args.expname

	os.makedirs(os.path.join(basedir, expname), exist_ok=True)
	f = os.path.join(basedir, expname, 'args.txt')
	with open(f, 'w') as file:
		for arg in sorted(vars(args)):
			attr = getattr(args, arg)
			file.write('{} = {}\n'.format(arg, attr))
	if args.config is not None:
		f = os.path.join(basedir, expname, 'config.txt')
		with open(f, 'w') as file:
			file.write(open(args.config, 'r').read())