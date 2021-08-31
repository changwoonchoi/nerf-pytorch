import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import utils.logging_utils
from nerf_models.positional_embedder import get_embedder
import os


# Model
class NeRF(nn.Module):
	def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4],
	             use_viewdirs=False, instance_label_dimension=0, use_instance_feature_layer=False):
		""" 
		"""
		super(NeRF, self).__init__()
		self.D = D
		self.W = W
		self.input_ch = input_ch
		self.input_ch_views = input_ch_views
		self.output_ch = output_ch
		self.skips = skips
		self.use_viewdirs = use_viewdirs
		self.instance_label_dimension = instance_label_dimension
		self.use_instance_feature_layer = use_instance_feature_layer
		
		self.pts_linears = nn.ModuleList(
			[nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
		
		### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
		self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

		### Implementation according to the paper
		# self.views_linears = nn.ModuleList(
		#     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])
		
		if use_viewdirs:
			self.feature_linear = nn.Linear(W, W)
			self.alpha_linear = nn.Linear(W, 1)
			if self.instance_label_dimension > 0:
				if self.use_instance_feature_layer:
					self.instance_feature_linear = nn.Linear(W, W//2)
					self.instance_linear = nn.Linear(W//2, self.instance_label_dimension)
				else:
					self.instance_feature_linear = None
					self.instance_linear = nn.Linear(W, self.instance_label_dimension)
			self.rgb_linear = nn.Linear(W//2, 3)
		else:
			self.output_linear = nn.Linear(W, output_ch)

	def __str__(self):
		logs = ["[NeRF]"]
		logs += ["\t- depth : %s" % str(self.D)]
		logs += ["\t- width : %s" % str(self.W)]
		logs += ["\t- input_ch : %s" % str(self.input_ch)]
		logs += ["\t- output_ch : %s" % str(self.output_ch)]
		logs += ["\t- instance_label_dimension : %s" % str(self.instance_label_dimension)]
		logs += ["\t- use_view_dir : %s" % str(self.use_viewdirs)]
		logs += ["\t- use_instance_feature_layer : %s" % str(self.use_instance_feature_layer)]
		return "\n".join(logs)

	def forward(self, x):
		input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
		h = input_pts
		for i, l in enumerate(self.pts_linears):
			h = self.pts_linears[i](h)
			h = F.relu(h)
			if i in self.skips:
				h = torch.cat([input_pts, h], -1)

		if self.use_viewdirs:
			alpha = self.alpha_linear(h)
			if self.instance_label_dimension > 0:
				if self.use_instance_feature_layer:
					instance = self.instance_linear(self.instance_feature_linear(h))
				else:
					instance = self.instance_linear(h)
					# instance = nn.Sigmoid(instance)  -> activate with softmax function after accumulate along ray direction.
			feature = self.feature_linear(h)
			h = torch.cat([feature, input_views], -1)
		
			for i, l in enumerate(self.views_linears):
				h = self.views_linears[i](h)
				h = F.relu(h)

			rgb = self.rgb_linear(h)
			if self.instance_label_dimension > 0:
				outputs = torch.cat([rgb, alpha, instance], -1)
			else:
				outputs = torch.cat([rgb, alpha], -1)
		else:
			outputs = self.output_linear(h)

		return outputs

	def load_weights_from_keras(self, weights):
		assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
		
		# Load pts_linears
		for i in range(self.D):
			idx_pts_linears = 2 * i
			self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
			self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
		
		# Load feature_linear
		idx_feature_linear = 2 * self.D
		self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
		self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

		# Load views_linears
		idx_views_linears = 2 * self.D + 2
		self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
		self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

		# Load rgb_linear
		idx_rbg_linear = 2 * self.D + 4
		self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
		self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

		# Load alpha_linear
		idx_alpha_linear = 2 * self.D + 6
		self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
		self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

def batchify(fn, chunk):
	"""
	Constructs a version of 'fn' that applies to smaller batches.
	"""
	if chunk is None:
		return fn
	def ret(inputs):
		return torch.cat([fn(inputs[i:i + chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
	return ret
	
def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
	"""
	Prepares inputs and applies network 'fn'.
	"""
	inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
	embedded = embed_fn(inputs_flat)

	if viewdirs is not None:
		input_dirs = viewdirs[:,None].expand(inputs.shape)
		input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
		embedded_dirs = embeddirs_fn(input_dirs_flat)
		embedded = torch.cat([embedded, embedded_dirs], -1)

	outputs_flat = batchify(fn, netchunk)(embedded)
	outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
	return outputs


def create_nerf(args):
	"""
	Instantiate NeRF's MLP model.
	"""
	embed_fn, input_ch = get_embedder(args.multires, args.i_embed)
	logger = utils.logging_utils.load_logger("NeRF Loader")

	input_ch_views = 0
	embeddirs_fn = None
	if args.use_viewdirs:
		embeddirs_fn, input_ch_views = get_embedder(args.multires_views, args.i_embed)
	output_ch = 5 if args.N_importance > 0 else 4
	skips = [4]
	model = NeRF(
		D=args.netdepth, W=args.netwidth, input_ch=input_ch, output_ch=output_ch, skips=skips,
		input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
		instance_label_dimension=args.instance_label_dimension,
		use_instance_feature_layer=args.use_instance_feature_layer
	).to(args.device)
	logger.info(model)

	grad_vars = list(model.parameters())

	model_fine = None
	if args.N_importance > 0:
		model_fine = NeRF(
			D=args.netdepth_fine, W=args.netwidth_fine, input_ch=input_ch, output_ch=output_ch, skips=skips,
			input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs,
			instance_label_dimension=args.instance_label_dimension,
			use_instance_feature_layer=args.use_instance_feature_layer
		).to(args.device)
		logger.info("NeRF fine model")
		logger.info(model)
		grad_vars += list(model_fine.parameters())

	network_query_fn = lambda inputs, viewdirs, network_fn: run_network(
		inputs, viewdirs, network_fn, embed_fn=embed_fn, embeddirs_fn=embeddirs_fn, netchunk=args.netchunk
	)

	# Create optimizer
	optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

	start = 0
	basedir = args.basedir
	expname = args.expname

	##########################

	# Load checkpoints
	if args.ft_path is not None and args.ft_path!='None':
		ckpts = [args.ft_path]
	else:
		ckpts = [os.path.join(basedir, expname, f) \
				 for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]

	logger.info('Found ckpts: %s' % str(ckpts))

	if len(ckpts) > 0 and not args.no_reload:
		ckpt_path = ckpts[-1]
		logger.info('Reloading from %s' % str(ckpt_path))
		ckpt = torch.load(ckpt_path)

		start = ckpt['global_step']
		optimizer.load_state_dict(ckpt['optimizer_state_dict'])

		# Load model
		model.load_state_dict(ckpt['network_fn_state_dict'])
		if model_fine is not None:
			model_fine.load_state_dict(ckpt['network_fine_state_dict'])

	##########################

	render_kwargs_train = {
		'network_query_fn': network_query_fn,
		'perturb': args.perturb,
		'N_importance': args.N_importance,
		'network_fine': model_fine,
		'N_samples': args.N_samples,
		'network_fn': model,
		'use_viewdirs': args.use_viewdirs,
		'white_bkgd': args.white_bkgd,
		'raw_noise_std': args.raw_noise_std,
		'label_encoding': args.instance_label_encoding,
	}

	# NDC only good for LLFF-style forward facing data
	if args.dataset_type != 'llff' or args.no_ndc:
		logger.info('Not ndc!')
		render_kwargs_train['ndc'] = False
		render_kwargs_train['lindisp'] = args.lindisp

	render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
	render_kwargs_test['perturb'] = False
	render_kwargs_test['raw_noise_std'] = 0.

	return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer