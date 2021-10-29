from nerf_models.nerf_renderer_helper import *
import time
from tqdm import tqdm, trange
import os
import imageio
from utils.label_utils import *
from torch.nn.functional import normalize
from torch.autograd import Variable
from nerf_models.microfacet import *

DEBUG = False


# def calculate_normal(network_fn, network_query_fn, pts):
# 	raw = network_query_fn(pts, None, network_fn)
# 	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
# 	sigma = raw2sigma(, dists)



def raw2outputs(raw, z_vals, normal, rays_d, instance_label_dimension=0, raw_noise_std=0.0, white_bkgd=False, pytest=False,
				is_instance_label_logit=True, label_encoder=None, init_basecolor=None, albedo_mod=None,
				calculate_normal=False, infer_normal=False, brdf_lut=None):
	"""Transforms model's predictions to semantically meaningful values.
	Args:
		raw: [num_rays, num_samples along ray, 4]. Prediction from model.
		- instance_label_dimension==0: [num_rays, num_samples along ray, 4]. Prediction from model. (R,G,B,a)
		- instance_label_dimension >0: [num_rays, num_samples along ray, 10]. (R,G,B,a,instance_label_dimension)
		z_vals: [num_rays, num_samples along ray]. Integration time.
		rays_d: [num_rays, 3]. Direction of each ray.
		is_instance_label_logit: export instance label as logit
	Returns:
		rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
		disp_map: [num_rays]. Disparity map. Inverse of depth map.
		acc_map: [num_rays]. Sum of weights along each ray.
		weights: [num_rays, num_samples]. Weights assigned to each sampled color.
		depth_map: [num_rays]. Estimated distance to object.
	"""
	# sigma_th = 0.0
	# sigma_filter = nn.Threshold(sigma_th, 0)

	num_cluster = init_basecolor.shape[0]
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	noise = 0.
	if raw_noise_std > 0.:
		noise = torch.randn(raw[..., 0].shape) * raw_noise_std

		# Overwrite randomly sampled data if pytest
		if pytest:
			np.random.seed(0)
			noise = np.random.rand(*list(raw[..., 0].shape)) * raw_noise_std
			noise = torch.Tensor(noise)

	sigma = raw2sigma(raw[..., 0] + noise, dists)  # [N_rays, N_samples, ]
	# sigma = alpha_filter(sigma)

	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]

	albedo = torch.sigmoid(raw[..., 1:4])
	albedo_map = torch.sum(weights[..., None] * albedo, -2)

	roughness = torch.sigmoid(raw[..., 4])
	roughness_map = torch.sum(weights * roughness, -1)

	indirect_illumination_weight = raw[..., 5:5 + num_cluster]  # [N_rays, N_sample, num_cluster]
	direct_illumination = raw[..., 5 + num_cluster]  # [N_rays, N_sample,]

	direct_illumination_extend = direct_illumination.reshape(*direct_illumination.shape, 1)
	direct_illumination_extend = direct_illumination_extend.expand(*direct_illumination_extend.shape[:-1], 3)
	direct_illumination_map = torch.sum(weights[..., None] * direct_illumination_extend, -2)

	decomp_indirect_illum = indirect_illumination_weight.reshape((*indirect_illumination_weight.shape, 1))
	decomp_indirect_illum = decomp_indirect_illum.expand(*decomp_indirect_illum.shape[:-1], 3)
	decomp_indirect_illum = decomp_indirect_illum * init_basecolor[None, None, ...]
	decomp_indirect_illum_map = torch.sum(weights[..., None, None] * decomp_indirect_illum, 1)

	indirect_illumination = torch.sum(decomp_indirect_illum, dim=-2)
	indirect_illumination_map = torch.sum(weights[..., None] * indirect_illumination, -2)

	illumination = indirect_illumination + direct_illumination.reshape((*direct_illumination.shape, 1))  # [N_rays, N_samples, 3]
	illumination_map = torch.sum(weights[..., None] * illumination, -2)

	normal_map = torch.sum(weights[..., None] * normal, -2)  # [N_rays, 3]
	target_normal = normal
	if infer_normal:
		inferred_normal = raw[..., 4 + num_cluster: 7 + num_cluster]
		target_normal = inferred_normal.clone().detach()
		# (TODO) do normalize..?
		# inferred_normal = normalize(inferred_normal, dim=-1)
		inferred_normal_map = torch.sum(weights[..., None] * inferred_normal, -2)  # [N_rays, 3]
	else:
		inferred_normal = None
		inferred_normal_map = None

	n_dot_v = torch.sum(rays_d.unsqueeze(1) * target_normal, -1)
	n_dot_v = F.relu(n_dot_v)

	BRDF_2D_LUT_uv = torch.stack([n_dot_v, roughness], -1)
	envBRDF = F.grid_sample(brdf_lut[None, ...], BRDF_2D_LUT_uv[None,...], align_corners=True)
	envBRDF = envBRDF.permute((0, 2, 3, 1))
	envBRDF = envBRDF.squeeze(0)

	F0 = torch.tensor([0.04, 0.04, 0.04])
	F0 = F0.repeat(*sigma.shape, 1)
	envBRDF_coefficient1 = envBRDF[..., 0]
	envBRDF_coefficient0 = envBRDF[..., 1]
	envBRDF_coefficient1 = torch.stack(3*[envBRDF_coefficient1], -1)
	fresnel = fresnel_schlick_roughness(n_dot_v, F0, roughness)
	specular = fresnel * envBRDF_coefficient1 + envBRDF_coefficient0[..., None]
	specular_map = torch.sum(weights[..., None] * specular, -2)

	# print(fresnel.shape, "fresnel")
	# rint(specular.shape, "specular")
	K_d = 1 - fresnel
	color = (K_d * albedo + specular) * illumination  # [N_rays, N_samples, 3]
	color_map = torch.sum(weights[..., None] * color, -2)  # [N_rays, 3]

	depth_map = torch.sum(weights * z_vals, -1)
	disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
	acc_map = torch.sum(weights, -1)
	#print(disp_map.shape, "disp map")
	#print(specular_map.shape, "specular_map")
	#print(roughness_map.shape, "roughness_map")

	return color_map, albedo, albedo_map, indirect_illumination_weight, disp_map, acc_map, weights, depth_map, \
		direct_illumination_map, indirect_illumination_map, decomp_indirect_illum_map, illumination_map, normal_map, \
		inferred_normal, inferred_normal_map, roughness_map, specular_map


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, init_basecolor, retraw=False, lindisp=False,
				perturb=0., N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False,
				pytest=False, is_instance_label_logit=False, decompose=False, label_encoder=None, use_viewdirs=None
				,brdf_lut=None):
	"""Volumetric rendering.
	Args:
	  ray_batch: array of shape [batch_size, ...]. All information necessary
		for sampling along a ray, including: ray origin, ray direction, min
		dist, max dist, and unit-magnitude viewing direction.
	  network_fn: function. Model for predicting RGB and density at each point
		in space.
	  network_query_fn: function used for passing queries to network_fn.
	  N_samples: int. Number of different times to sample along each ray.
	  retraw: bool. If True, include model's raw, unprocessed predictions.
	  lindisp: bool. If True, sample linearly in inverse depth rather than in depth.
	  perturb: float, 0 or 1. If non-zero, each ray is sampled at stratified
		random points in time.
	  N_importance: int. Number of additional times to sample along each ray.
		These samples are only passed to network_fine.
	  network_fine: "fine" network with same spec as network_fn.
	  white_bkgd: bool. If True, assume a white background.
	  raw_noise_std: ...
	  verbose: bool. If True, print more debugging info.
	Returns:
	  rgb_map: [num_rays, 3]. Estimated RGB color of a ray. Comes from fine model.
	  disp_map: [num_rays]. Disparity map. 1 / depth.
	  acc_map: [num_rays]. Accumulated opacity along each ray. Comes from fine model.
	  raw: [num_rays, num_samples, 4]. Raw predictions from model.
	  rgb0: See rgb_map. Output for coarse model.
	  disp0: See disp_map. Output for coarse model.
	  acc0: See acc_map. Output for coarse model.
	  z_std: [num_rays]. Standard deviation of distances along ray for each
		sample.
	"""
	N_rays = ray_batch.shape[0]
	rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]  # [N_rays, 3] each
	viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
	bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
	near, far = bounds[..., 0], bounds[..., 1]  # [-1,1]

	t_vals = torch.linspace(0., 1., steps=N_samples)
	if not lindisp:
		z_vals = near * (1. - t_vals) + far * (t_vals)
	else:
		z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

	z_vals = z_vals.expand([N_rays, N_samples])

	if perturb > 0.:
		# get intervals between samples
		mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
		upper = torch.cat([mids, z_vals[..., -1:]], -1)
		lower = torch.cat([z_vals[..., :1], mids], -1)
		# stratified samples in those intervals
		t_rand = torch.rand(z_vals.shape)

		# Pytest, overwrite u with numpy's fixed random numbers
		if pytest:
			np.random.seed(0)
			t_rand = np.random.rand(*list(z_vals.shape))
			t_rand = torch.Tensor(t_rand)

		z_vals = lower + (upper - lower) * t_rand
	pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
	#pts = pts.detach()
	#pts = Variable(pts.data, requires_grad=True)
	#pts.requires_grad = True

	raw = network_query_fn(pts, viewdirs, network_fn)  # raw = [sigma, albedo_res, indirect_illumination, direct_illumination, basecolor_score]

	# Calculate normal
	pts.requires_grad = True
	sigma_x = network_query_fn(pts, None, network_fn)
	sigma_x = F.relu(sigma_x)
	sigma_x.backward(torch.ones_like(sigma_x))
	normal = -normalize(pts.grad, dim=-1)
	normal = normal.detach()
	pts.requires_grad = False

	color_map, albedo, albedo_map, indirect_illumination_weight, disp_map, acc_map, weights, depth_map, \
	direct_illumination_map, indirect_illumination_map, decomp_indirect_illum_map, illumination_map, normal_map,\
	inferred_normal, inferred_normal_map, roughness_map, specular_map = raw2outputs(
		raw, z_vals, normal, rays_d, network_fn.instance_label_dimension, raw_noise_std, white_bkgd, pytest=pytest,
		is_instance_label_logit=is_instance_label_logit, label_encoder=label_encoder, init_basecolor=init_basecolor,
		infer_normal=network_fn.infer_normal, brdf_lut=brdf_lut
	)

	if N_importance > 0:
		color_map_0, disp_map_0, acc_map_0 = color_map, disp_map, acc_map
		albedo_0 = albedo
		roughness_map_0 = roughness_map
		specular_map_0 = specular_map
		weights_0 = weights
		indirect_illumination_weight_0 = indirect_illumination_weight
		albedo_map_0, direct_illumination_map_0, indirect_illumination_map_0 = albedo_map, direct_illumination_map, indirect_illumination_map
		decomp_indirect_illum_map_0, illumination_map_0 = decomp_indirect_illum_map, illumination_map
		normal_0, normal_map_0 = normal, normal_map
		inferred_normal_0, inferred_normal_map_0 = inferred_normal, inferred_normal_map

		z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
		z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
		z_samples = z_samples.detach()

		z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
		pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]
		#pts = Variable(pts.data, requires_grad=True)

		run_fn = network_fn if network_fine is None else network_fine
		#         raw = run_network(pts, fn=run_fn)
		raw = network_query_fn(pts, viewdirs, run_fn)

		# Calculate normal
		pts.requires_grad = True
		sigma_x = network_query_fn(pts, None, run_fn)
		sigma_x = F.relu(sigma_x)
		sigma_x.backward(torch.ones_like(sigma_x))
		normal = -normalize(pts.grad, dim=-1)
		normal = normal.detach()
		pts.requires_grad = False

		color_map, albedo, albedo_map, indirect_illumination_weight, disp_map, acc_map, weights, depth_map, \
		direct_illumination_map, indirect_illumination_map, decomp_indirect_illum_map, illumination_map, normal_map,\
		inferred_normal, inferred_normal_map, roughness_map, specular_map = raw2outputs(
			raw, z_vals, normal, rays_d, run_fn.instance_label_dimension, raw_noise_std, white_bkgd, pytest=pytest,
			is_instance_label_logit=is_instance_label_logit, label_encoder=label_encoder, init_basecolor=init_basecolor,
			infer_normal=network_fn.infer_normal, brdf_lut=brdf_lut
		)

	ret = {
		'color_map': color_map,
		'albedo': albedo,
		'albedo_map': albedo_map,
		'roughness_map': roughness_map,
		'specular_map': specular_map,
		'indirect_illumination_weight': indirect_illumination_weight,
		'disp_map': disp_map,
		'acc_map': acc_map,
		'weights': weights,
		'direct_illumination_map': direct_illumination_map,
		'indirect_illumination_map': indirect_illumination_map,
		'decomp_indirect_illum_map': decomp_indirect_illum_map,
		'illumination_map': illumination_map,
		'normal' : normal,
		'normal_map': normal_map
	}
	if retraw:
		ret['raw'] = raw
	if network_fn.infer_normal:
		ret['inferred_normal'] = inferred_normal
		ret['inferred_normal_map'] = inferred_normal_map
		if N_importance > 0:
			ret['inferred_normal_0'] = inferred_normal_0
			ret['inferred_normal_map_0'] = inferred_normal_map_0
	if N_importance > 0:
		ret['color_map0'] = color_map_0
		ret['albedo0'] = albedo_0
		ret['albedo_map0'] = albedo_map_0
		ret['indirect_illumination_weight0'] = indirect_illumination_weight_0
		ret['disp0'] = disp_map_0
		ret['acc0'] = acc_map_0
		ret['weights0'] = weights_0
		ret['direct_illumination_map0'] = direct_illumination_map_0
		ret['indirect_illumination_map0'] = indirect_illumination_map_0
		ret['decomp_indirect_illum_map0'] = decomp_indirect_illum_map_0
		ret['illumination_map0'] = illumination_map_0
		ret['normal_0'] = normal_0
		ret['normal_map_0'] = normal_map_0

		ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

	for k in ret:
		if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
			print(f"! [Numerical Error] {k} contains nan or inf.")

	return ret


def batchify_rays(rays_flat, chunk=1024 * 32, label_encoder=None, **kwargs):
	"""Render rays in smaller minibatches to avoid OOM.
	"""
	all_ret = {}
	for i in range(0, rays_flat.shape[0], chunk):
		ret = render_rays(rays_flat[i:i + chunk], label_encoder=label_encoder, **kwargs)
		for k in ret:
			if k not in all_ret:
				all_ret[k] = []
			all_ret[k].append(ret[k])

	# all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
	for k in all_ret:
		all_ret[k] = torch.cat(all_ret[k], dim=0)
	return all_ret


def render_decomp(
		H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True, near=0., far=1.,
		c2w_staticcam=None, label_encoder=None, **kwargs
):
	"""Render rays
	Args:
	  H: int. Height of image in pixels.
	  W: int. Width of image in pixels.
	  focal: float. Focal length of pinhole camera.
	  chunk: int. Maximum number of rays to process simultaneously. Used to
		control maximum memory usage. Does not affect final results.
	  rays: array of shape [2, batch_size, 3]. Ray origin and direction for
		each example in batch.
	  c2w: array of shape [3, 4]. Camera-to-world transformation matrix.
	  ndc: bool. If True, represent ray origin, direction in NDC coordinates.
	  near: float or array of shape [batch_size]. Nearest distance for a ray.
	  far: float or array of shape [batch_size]. Farthest distance for a ray.
	  use_viewdirs: bool. If True, use viewing direction of a point in space in model.
	  c2w_staticcam: array of shape [3, 4]. If not None, use this transformation matrix for
	   camera while using other c2w argument for viewing directions.
	Returns:
	  rgb_map: [batch_size, 3]. Predicted RGB values for rays.
	  disp_map: [batch_size]. Disparity map. Inverse of depth.
	  acc_map: [batch_size]. Accumulated opacity (alpha) along a ray.
	  extras: dict with everything returned by render_rays().
	"""
	if c2w is not None:
		# special case to render full image
		rays_o, rays_d = get_rays(H, W, K, c2w)
	else:
		# use provided ray batch
		rays_o, rays_d = rays

	viewdirs = rays_d
	if c2w_staticcam is not None:
		# special case to visualize effect of viewdirs
		rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
	viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
	viewdirs = torch.reshape(viewdirs, [-1, 3]).float()

	sh = rays_d.shape  # [..., 3]
	if ndc:
		# for forward facing scenes
		rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

	# Create ray batch
	rays_o = torch.reshape(rays_o, [-1, 3]).float()
	rays_d = torch.reshape(rays_d, [-1, 3]).float()

	near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
	rays = torch.cat([rays_o, rays_d, near, far], -1)
	rays = torch.cat([rays, viewdirs], -1)

	# Render and reshape
	all_ret = batchify_rays(rays, chunk, label_encoder=label_encoder, **kwargs)
	for k in all_ret:
		k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
		all_ret[k] = torch.reshape(all_ret[k], k_sh)
	return all_ret


# k_extract = ['rgb_map', 'disp_map', 'acc_map']
# ret_list = [all_ret[k] for k in k_extract]
# ret_dict = {k : all_ret[k] for k in all_ret if k not in k_extract}
# return ret_list + [ret_dict]


def render_decomp_path(render_poses, hwf, K, chunk, render_kwargs,
					   savedir=None, render_factor=0, init_basecolor=None):
	H, W, focal = hwf

	if render_factor != 0:
		# Render downsampled for speed
		H = H // render_factor
		W = W // render_factor
		focal = focal / render_factor

	rgbs = []
	albedos = []
	direct_illuminations = []
	indirect_illuminations = []
	decomp_indirect_illuminations = []
	illuminations = []
	disps = []
	normals = []
	inferred_normals = []
	roughnesss = []
	speculars = []

	K = np.array([
		[focal, 0, 0.5 * W],
		[0, focal, 0.5 * H],
		[0, 0, 1]
	]).astype(np.float32)

	t = time.time()
	for i, c2w in enumerate(tqdm(render_poses)):
		t = time.time()
		results = render_decomp(H, W, K, chunk=chunk, c2w=c2w[:3, :4], init_basecolor=init_basecolor, **render_kwargs)
		rgb = results['color_map']  # (chunk, 3)
		albedo = results['albedo_map']  # (chunk, 3)
		direct_illumination = results['direct_illumination_map']  # (chunk, 3)
		indirect_illumination = results['indirect_illumination_map']  # (chunk, 3)
		decomp_indirect_illum_map = results['decomp_indirect_illum_map']  # (chunk, num_cluster, 3)
		illumination_map = results['illumination_map']  # (chunk, 3)
		disp = results['disp_map']
		normal = (results['normal_map'] + 1) * 0.5
		if 'inferred_normal_map' in results:
			inferred_normal = (results['inferred_normal_map'] + 1) * 0.5
			inferred_normals.append(inferred_normal.cpu().numpy())
		roughness = results['roughness_map']
		specular = results['specular_map']

		rgbs.append(rgb.cpu().numpy())
		albedos.append(albedo.cpu().numpy())
		direct_illuminations.append(direct_illumination.cpu().numpy())
		indirect_illuminations.append(indirect_illumination.cpu().numpy())
		illuminations.append(illumination_map.cpu().numpy())
		disps.append(disp.cpu().numpy())
		normals.append(normal.cpu().numpy())
		roughnesss.append(roughness.cpu().numpy())
		speculars.append(specular.cpu().numpy())

		if savedir is not None:
			rgb8 = to8b(rgbs[-1])
			albedo8 = to8b(albedos[-1])
			direct_illumination8 = to8b(direct_illuminations[-1])
			indirect_illumination8 = to8b(indirect_illuminations[-1])
			illumination8 = to8b(illuminations[-1])
			normal8 = to8b(normals[-1])
			roughness8 = to8b(roughnesss[-1])
			disp8 = to8b(disps[-1])
			specular8 = to8b(speculars[-1])

			filename_rgb = os.path.join(savedir, 'rgb_{:03d}.png'.format(i))
			filename_albedo = os.path.join(savedir, 'albedo_{:03d}.png'.format(i))
			filename_direct_illumination = os.path.join(savedir, 'direct_{:03d}.png'.format(i))
			filename_indirect_illumination = os.path.join(savedir, 'indirect_{:03d}.png'.format(i))
			filename_illumination = os.path.join(savedir, 'illumination_{:03d}.png'.format(i))
			filename_normal = os.path.join(savedir, 'normal_{:03d}.png'.format(i))
			filename_roughness = os.path.join(savedir, 'roughness_{:03d}.png'.format(i))
			filename_disp = os.path.join(savedir, 'disp_{:03d}.png'.format(i))
			filename_specular = os.path.join(savedir, 'specular_{:03d}.png'.format(i))

			imageio.imwrite(filename_rgb, rgb8)
			imageio.imwrite(filename_albedo, albedo8)
			imageio.imwrite(filename_direct_illumination, direct_illumination8)
			imageio.imwrite(filename_indirect_illumination, indirect_illumination8)
			imageio.imwrite(filename_illumination, illumination8)
			imageio.imwrite(filename_normal, normal8)
			imageio.imwrite(filename_roughness, roughness8)
			imageio.imwrite(filename_disp, disp8)
			imageio.imwrite(filename_specular, specular8)

			if 'inferred_normal_map' in results:
				inferred_normal8 = to8b(inferred_normals[-1])
				filename_inferred_normal = os.path.join(savedir, 'inferred_normal_{:03d}.png'.format(i))
				imageio.imwrite(filename_inferred_normal, inferred_normal8)

	rgbs = np.stack(rgbs, 0)
	albedos = np.stack(albedos, 0)
	direct_illuminations = np.stack(direct_illuminations, 0)
	indirect_illuminations = np.stack(indirect_illuminations, 0)
	illuminations = np.stack(illuminations, 0)
	disps = np.stack(disps, 0)
	normals = np.stack(normals, 0)
	if len(inferred_normals) > 0:
		inferred_normals = np.stack(inferred_normals, 0)
	roughnesss = np.stack(roughnesss, 0)
	speculars = np.stack(speculars, 0)

	return rgbs, albedos, direct_illuminations, indirect_illuminations, illuminations, disps, \
		   normals, inferred_normals, roughnesss, speculars

