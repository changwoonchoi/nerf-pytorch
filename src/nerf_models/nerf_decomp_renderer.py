from nerf_models.nerf_renderer_helper import *
import time
from tqdm import tqdm, trange
import os
import imageio
from utils.label_utils import *
from torch.nn.functional import normalize
from torch.autograd import Variable
from nerf_models.microfacet import *
from torchvision import transforms
DEBUG = False
from utils.depth_to_normal_utils import depth_to_normal_image_space

from nerf_models.normal_from_depth import *
from nerf_models.normal_from_sigma import *


def raw2outputs_simple(raw, z_vals, rays_d, coarse_radiance_number=3):
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	sigma = raw2sigma(raw[..., 0], dists)
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]

	radiance = torch.sigmoid(raw[..., 6:6 + 3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	# (5)-A additional coarse radiance maps
	N = 9
	coarse_radiance_maps = []
	for i in range(coarse_radiance_number):
		coarse_radiance = torch.sigmoid(raw[..., N:N + 3])
		coarse_radiance_map = torch.sum(weights[..., None] * coarse_radiance, -2)
		coarse_radiance_maps.append(coarse_radiance_map)
		N += 3

	return radiance_map, coarse_radiance_maps


def raw2outputs_neigh(rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, network_fn, raw_noise_std):
	# sample points
	pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
	raw = network_query_fn(pts, rays_d, network_fn)

	# distances
	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	noise = 0.
	if raw_noise_std > 0.:
		noise = torch.randn(raw[..., 0].shape) * raw_noise_std

	# (0) get sigma
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
	sigma = raw2sigma(raw[..., 0] + noise, dists)

	# (1) get weight from sigma
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
	weights_detached = weights.detach()
	roughness = torch.sigmoid(raw[..., 4])
	roughness_map = torch.sum(weights_detached * roughness, -1)
	results = {}
	results["roughness_map"] = roughness_map
	results["weights"] = weights
	return results


def raw2outputs(rays_o, rays_d, z_vals, z_vals_constant,
				network_query_fn, network_fn,
				raw_noise_std=0., pytest=False, neighbor=False,
				calculate_normal_from_sigma_gradient=False,
				calculate_normal_from_sigma_gradient_surface=False,
				calculate_normal_from_depth_gradient=False,
				calculate_normal_from_depth_gradient_direction=False,
				calculate_normal_from_depth_gradient_epsilon=False,
				calculate_normal_from_depth_gradient_direction_epsilon=False,
				infer_normal=False,
				infer_normal_at_surface=False,
				normal_mlp=None,
				brdf_lut=None,
				epsilon=0.01,
				epsilon_direction=0.01,
				gt_values=None,
				target_normal_map_for_radiance_calculation="ground_truth",
				target_albedo_map_for_radiance_calculation="ground_truth",
				target_roughness_map_for_radiance_calculation="ground_truth",
				target_radiance_map_for_radiance_calculation="ground_truth",
				**kwargs):
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
	if neighbor:
		return raw2outputs_neigh(
			rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, network_fn, raw_noise_std
		)

	# sample points
	pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
	raw = network_query_fn(pts, rays_d, network_fn)

	# distances
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

	# (0) get sigma
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
	sigma = raw2sigma(raw[..., 0] + noise, dists)

	# (1) get weight from sigma
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
	weights_detached = weights.detach()

	# (2) get depth / disp / acc map
	depth_map = torch.sum(weights * z_vals, -1)
	disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
	acc_map = torch.sum(weights, -1)

	# (3) get surface point surface_x
	x_surface = rays_o + rays_d * depth_map[..., None]
	x_surface.detach_()

	# (4A) calculate normal from sigma gradient
	# (4A-1) normal from sigma gradient at volume
	normal_map_from_sigma_gradient = None
	if calculate_normal_from_sigma_gradient:
		normal_map_from_sigma_gradient = get_normal_from_sigma_gradient(pts, weights_detached, network_query_fn, network_fn)

	# (4A-2) normal from sigma gradient at surface
	normal_map_from_sigma_gradient_surface = None
	if calculate_normal_from_sigma_gradient_surface:
		normal_map_from_sigma_gradient_surface = get_normal_from_sigma_gradient_surface(x_surface, network_query_fn, network_fn)

	# (4B) calculate normal from depth gradient
	# (4B-1) normal from depth gradient w.r.t position
	normal_map_from_depth_gradient = None
	if calculate_normal_from_depth_gradient:
		normal_map_from_depth_gradient = get_normal_from_depth_gradient(rays_o, rays_d, network_query_fn, network_fn, z_vals)
		normal_map_from_depth_gradient.detach_()

	# (4B-2) normal from numerical depth gradient w.r.t position
	normal_map_from_depth_gradient_epsilon = None
	if calculate_normal_from_depth_gradient_epsilon:
		normal_map_from_depth_gradient_epsilon = get_normal_from_depth_gradient_epsilon(rays_o, rays_d, network_query_fn, network_fn, z_vals, epsilon=epsilon)
		normal_map_from_depth_gradient_epsilon.detach_()

	# (4B-3) normal from depth gradient w.r.t direction
	normal_map_from_depth_gradient_direction = None
	if calculate_normal_from_depth_gradient_direction:
		normal_map_from_depth_gradient_direction = get_normal_from_depth_gradient_direction(rays_o, rays_d, network_query_fn, network_fn, z_vals)
		normal_map_from_depth_gradient_direction.detach_()

	# (4B-4) normal from numerical depth gradient w.r.t direction
	normal_map_from_depth_gradient_direction_epsilon = None
	if calculate_normal_from_depth_gradient_direction_epsilon:
		normal_map_from_depth_gradient_direction_epsilon = get_normal_from_depth_gradient_direction_epsilon(rays_o, rays_d, network_query_fn, network_fn, z_vals, epsilon=epsilon_direction)
		normal_map_from_depth_gradient_direction_epsilon.detach_()

	# (5) other values
	albedo = torch.sigmoid(raw[..., 1:4])
	albedo_map = torch.sum(weights_detached[..., None] * albedo, -2)

	roughness = torch.sigmoid(raw[..., 4])
	roughness_map = torch.sum(weights_detached * roughness, -1)

	irradiance = torch.sigmoid(raw[..., 5])
	irradiance_map = torch.sum(weights_detached * irradiance, -1)

	radiance = torch.sigmoid(raw[..., 6:6 + 3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	# (5)-A additional coarse radiance maps
	N = 9
	coarse_radiance_maps = []
	for i in range(network_fn.coarse_radiance_number):
		coarse_radiance = torch.sigmoid(raw[..., N:N + 3])
		coarse_radiance_map = torch.sum(weights_detached[..., None] * coarse_radiance, -2)
		coarse_radiance_maps.append(coarse_radiance_map)
		N += 3

	# (6) infer normal if necessary
	inferred_normal_map = None
	if infer_normal:
		if infer_normal_at_surface:
			inferred_normal_map = network_query_fn(x_surface[..., None, :], None, normal_mlp)
			inferred_normal_map = 2 * torch.sigmoid(inferred_normal_map) - 1
			inferred_normal_map.squeeze_(-2)
		else:
			inferred_normal_raw = network_query_fn(pts, None, normal_mlp)
			inferred_normal = 2 * torch.sigmoid(inferred_normal_raw) - 1
			inferred_normal_map = torch.sum(weights_detached[..., None] * inferred_normal, -2)

	target_normal_map = inferred_normal_map
	if target_normal_map_for_radiance_calculation == "ground_truth":
		target_normal_map = normalize(2 * gt_values["normal"] - 1, dim=-1)
	elif target_normal_map_for_radiance_calculation == "inferred_normal_map":
		target_normal_map = inferred_normal_map
	elif target_normal_map_for_radiance_calculation == "normal_map_from_sigma_gradient":
		target_normal_map = normal_map_from_sigma_gradient
	elif target_normal_map_for_radiance_calculation == "normal_map_from_sigma_gradient_surface":
		target_normal_map = normal_map_from_sigma_gradient_surface
	elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient":
		target_normal_map = normal_map_from_depth_gradient
	elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient_epsilon":
		target_normal_map = normal_map_from_depth_gradient_epsilon
	elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient_direction":
		target_normal_map = normal_map_from_depth_gradient_direction
	elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient_direction_epsilon":
		target_normal_map = normal_map_from_depth_gradient_direction_epsilon

	target_albedo_map = albedo_map
	# if target_albedo_map_for_radiance_calculation == "ground_truth":
	# 	target_albedo_map = gt_values["albedo"]

	target_roughness_map = roughness_map
	# if target_roughness_map_for_radiance_calculation == "ground_truth":
	# 	target_roughness_map = gt_values["roughness"][...,0]

	# (7) calculate color from split-sum approximation
	n_dot_v = torch.sum(-rays_d * target_normal_map, -1)
	n_dot_v = torch.clip(n_dot_v, 0, 1)

	# grid_sample input is  [-1, 1] x [-1, 1]
	BRDF_2D_LUT_uv = torch.stack([2 * n_dot_v - 1, 2 * target_roughness_map - 1], -1)
	envBRDF = F.grid_sample(brdf_lut[None, ...], BRDF_2D_LUT_uv[None, :, None, ...], align_corners=True)
	envBRDF = envBRDF.permute((0, 2, 3, 1))
	envBRDF = envBRDF.squeeze()

	# dielectric
	F0 = torch.tensor([0.04, 0.04, 0.04])
	F0 = F0.repeat(*depth_map.shape, 1)
	target_metallic_map = (1-target_roughness_map)[...,None]
	F0 = F0 * (1-target_metallic_map) + target_albedo_map * target_metallic_map

	envBRDF_coefficient1 = envBRDF[..., 0]
	envBRDF_coefficient0 = envBRDF[..., 1]
	envBRDF_coefficient1 = torch.stack(3 * [envBRDF_coefficient1], -1)
	fresnel_map = fresnel_schlick_roughness(n_dot_v, F0, target_roughness_map)
	specular_map = fresnel_map * envBRDF_coefficient1 + envBRDF_coefficient0[..., None]

	with torch.no_grad():
		reflected_dirs = rays_d - 2 * torch.sum(target_normal_map * rays_d, -1, keepdim=True) * target_normal_map
		x_surface = rays_o + rays_d * depth_map[..., None]

		reflected_pts = x_surface[..., None, :] + reflected_dirs[..., None, :] * z_vals_constant[..., :, None]
		reflected_ray_raw = network_query_fn(reflected_pts, reflected_dirs, network_fn)
		reflected_radiance_map, reflected_coarse_radiance_map = raw2outputs_simple(reflected_ray_raw, z_vals_constant, reflected_dirs)

		prefiltered_env_maps = torch.stack([reflected_radiance_map] + reflected_coarse_radiance_map, dim=1)

	N_pref = len(reflected_coarse_radiance_map) + 1
	mipmap_index1 = (roughness_map * (N_pref - 1)).long()
	mipmap_index1 = torch.clip(mipmap_index1, 0, N_pref - 1)
	mipmap_index2 = torch.clip(mipmap_index1 + 1, 0, N_pref - 1)
	mipmap_remainder = ((roughness_map * (N_pref - 1)) - mipmap_index1)[..., None]
	prefiltered_reflected_map = \
		(1-mipmap_remainder) * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index1] +\
		mipmap_remainder * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index2]

	diffuse_map = (1 - fresnel_map) * (1-target_metallic_map) * target_albedo_map * irradiance_map[..., None]
	specular_map = specular_map * prefiltered_reflected_map
	approximated_radiance_map = diffuse_map + specular_map

	# [N_rays, N_samples, 3]

	# approximated_radiance_map = None
	# specular_map = None

	# Organize results
	results = {}
	results["color_map"] = approximated_radiance_map
	results["radiance_map"] = radiance_map
	for k in range(len(coarse_radiance_maps)):
		results["radiance_map_%d" % (k + 1)] = coarse_radiance_maps[k]

	results["irradiance_map"] = irradiance_map
	results["albedo_map"] = albedo_map
	results["roughness_map"] = roughness_map
	results["specular_map"] = specular_map
	results["diffuse_map"] = diffuse_map
	results["n_dot_v_map"] = n_dot_v

	results["normal_map_from_sigma_gradient"] = normal_map_from_sigma_gradient
	results["normal_map_from_sigma_gradient_surface"] = normal_map_from_sigma_gradient_surface
	results["normal_map_from_depth_gradient"] = normal_map_from_depth_gradient
	results["normal_map_from_depth_gradient_direction"] = normal_map_from_depth_gradient_direction
	results["normal_map_from_depth_gradient_epsilon"] = normal_map_from_depth_gradient_epsilon
	results["normal_map_from_depth_gradient_direction_epsilon"] = normal_map_from_depth_gradient_direction_epsilon

	results["inferred_normal_map"] = inferred_normal_map

	results["disp_map"] = disp_map
	results["acc_map"] = acc_map
	results["depth_map"] = depth_map
	results["weights"] = weights

	return results


# def render_rays(ray_batch, network_fn, network_query_fn, N_samples, init_basecolor, retraw=False, lindisp=False,
# 				perturb=0., N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False,
# 				pytest=False, is_instance_label_logit=False, decompose=False, label_encoder=None, use_viewdirs=None
# 				, brdf_lut=None, calculate_normal_from_sigma=True, depth_mlp=None,
# 				calculate_normal_from_depth=True, infer_depth=False, calculate_normal_from_surface=False):
def render_rays(ray_batch, network_fn, network_query_fn, N_samples, init_basecolor, retraw=False, lindisp=False,
				perturb=0., N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False,
				pytest=False, neighbor=False, **kwargs):
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

	# (1) Sample positions
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

	z_vals_constant = z_vals
	result = raw2outputs(
		rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, network_fn, raw_noise_std, pytest, neighbor, **kwargs
	)

	# (2) need importance sampling
	if N_importance > 0:
		weights = result["weights"]
		z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
		z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
		z_samples = z_samples.detach()
		run_fn = network_fn if network_fine is None else network_fine

		z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
		result_fine = raw2outputs(
			rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, run_fn, raw_noise_std, pytest, neighbor, **kwargs
		)

		for k, v in result.items():
			result_fine[k + "0"] = result[k]

		result = result_fine

	if N_importance > 0:
		result['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

	result = {k: v for k, v in result.items() if v is not None}

	if kwargs.get("infer_depth", False):
		inferred_depth_map = network_query_fn(rays_o[..., None, :], viewdirs, kwargs["depth_mlp"])
		inferred_depth_map = F.relu(inferred_depth_map[..., 0])
		inferred_depth_map.squeeze_()
		result["inferred_depth_map"] = inferred_depth_map

	for k in result:
		if (torch.isnan(result[k]).any() or torch.isinf(result[k]).any()) and DEBUG:
			print(f"! [Numerical Error] {k} contains nan or inf.")

	return result


def batchify_rays(rays_flat, chunk=1024 * 32, label_encoder=None, neighbor=False, **kwargs):
	"""Render rays in smaller minibatches to avoid OOM.
	"""
	all_ret = {}
	gt_values = kwargs.get("gt_values", None)
	N = rays_flat.shape[0]
	for i in range(0, N, chunk):
		gt_values_ith = {}
		if gt_values is not None:
			for k in gt_values.keys():
				gt_values_ith[k] = gt_values[k][i:min(i+chunk, N)]
		kwargs["gt_values"] = gt_values_ith
		ret = render_rays(rays_flat[i:i + chunk], label_encoder=label_encoder, neighbor=neighbor, **kwargs)
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
		c2w_staticcam=None, label_encoder=None, rays_neigh=None, **kwargs
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
	all_ret_neigh = None
	if c2w is not None:
		# special case to render full image
		rays_o, rays_d = get_rays(H, W, K, c2w)
	else:
		# use provided ray batch
		rays_o, rays_d = rays
		if rays_neigh is not None:
			rays_o_neigh, rays_d_neigh = rays_neigh
			viewdirs_neigh = rays_d_neigh
			viewdirs_neigh = viewdirs_neigh / torch.norm(viewdirs_neigh, dim=-1, keepdim=True)
			viewdirs_neigh = torch.reshape(viewdirs_neigh, [-1, 3]).float()
			sh_neigh = rays_d_neigh.shape
			if ndc:
				rays_o_neigh, rays_d_neigh = ndc_rays(H, W, K[0][0], 1., rays_o_neigh, rays_d_neigh)
			rays_o_neigh = torch.reshape(rays_o_neigh, [-1, 3]).float()
			rays_d_neigh = torch.reshape(rays_d_neigh, [-1, 3]).float()
			near_neigh, far_neigh = near * torch.ones_like(rays_d_neigh[..., :1]), far * torch.ones_like(rays_d_neigh[..., :1])
			rays_neighbor = torch.cat([rays_o_neigh, rays_d_neigh, near_neigh, far_neigh], -1)
			rays_neighbor = torch.cat([rays_neighbor, viewdirs_neigh], -1)
			all_ret_neigh = batchify_rays(rays_neighbor, chunk, label_encoder=label_encoder, neighbor=True, **kwargs)
			for k in all_ret_neigh:
				k_sh_neigh = list(sh_neigh[:-1]) + list(all_ret_neigh[k].shape[1:])
				all_ret_neigh[k] = torch.reshape(all_ret_neigh[k], k_sh_neigh)

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
	return all_ret, all_ret_neigh


from dataset.dataset_interface import NerfDataset

def render_decomp_path(dataset_test: NerfDataset, hwf, K, chunk, render_kwargs,
					   savedir=None, render_factor=0, init_basecolor=None,
					   gt_values = None,
					   calculate_normal_from_depth_map=False):
	H, W, focal = hwf
	render_poses = dataset_test.poses

	if render_factor != 0:
		# Render downsampled for speed
		H = H // render_factor
		W = W // render_factor
		focal = focal / render_factor

	K = np.array([
		[focal, 0, 0.5 * W],
		[0, focal, 0.5 * H],
		[0, 0, 1]
	]).astype(np.float32)

	results = {}

	def append_result(render_decomp_results, key_name, index, out_name):
		if key_name not in render_decomp_results:
			return
		result_image = render_decomp_results[key_name]
		if result_image is None:
			return
		if out_name not in results:
			results[out_name] = []
		if "normal" in out_name:
			result_image = (result_image + 1) * 0.5
		elif "depth" in key_name:
			# depth to disp
			result_image = 1. / torch.max(1e-10 * torch.ones_like(result_image), result_image)

		results[out_name].append(result_image.cpu().numpy())
		if savedir is not None:
			result_image_8bit = to8b(results[out_name][-1])

			filename = os.path.join(savedir, (out_name + '_{:03d}.png').format(index))
			imageio.imwrite(filename, result_image_8bit)

	for i, c2w in enumerate(tqdm(render_poses)):

		gt_values = dataset_test.get_resized_normal_albedo(render_factor, i)
		for k in gt_values.keys():
			gt_values[k] = torch.reshape(gt_values[k], [-1, gt_values[k].shape[-1]])
		results_i, _ = render_decomp(H, W, K, chunk=chunk, c2w=c2w[:3, :4], init_basecolor=init_basecolor, gt_values=gt_values, **render_kwargs)
		append_result(results_i, "color_map", i, "rgb")
		append_result(results_i, "radiance_map", i, "radiance")
		for k in range(render_kwargs["coarse_radiance_number"]):
			append_result(results_i, "radiance_map_%d" % (k+1), i, "radiance_%d" % (k+1))

		append_result(results_i, "irradiance_map", i, "irradiance")
		append_result(results_i, "albedo_map", i, "albedo")

		append_result(results_i, "roughness_map", i, "roughness")
		append_result(results_i, "specular_map", i, "specular")
		append_result(results_i, "diffuse_map", i, "diffuse")
		append_result(results_i, "n_dot_v_map", i, "n_dot_v")

		append_result(results_i, "normal_map_from_sigma_gradient", i, "normal_map_from_sigma_gradient")
		append_result(results_i, "normal_map_from_sigma_gradient_surface", i, "normal_map_from_sigma_gradient_surface")
		append_result(results_i, "normal_map_from_depth_gradient", i, "normal_map_from_depth_gradient")
		append_result(results_i, "normal_map_from_depth_gradient_direction", i, "normal_map_from_depth_gradient_direction")
		append_result(results_i, "normal_map_from_depth_gradient_epsilon", i, "normal_map_from_depth_gradient_epsilon")
		append_result(results_i, "normal_map_from_depth_gradient_direction_epsilon", i, "normal_map_from_depth_gradient_direction_epsilon")

		append_result(results_i, "inferred_normal_map", i, "inferred_normal_map")

		append_result(results_i, "inferred_depth_map", i, "inferred_disp")

		append_result(results_i, "disp_map", i, "disp")
		append_result(results_i, "depth_map", i, "depth")

		if calculate_normal_from_depth_map:
			depth_image = results_i["depth_map"]
			results_i["normal_map_from_depth_map"] = depth_to_normal_image_space(depth_image, c2w[:3, :4], K)
			append_result(results_i, "normal_map_from_depth_map", i, "normal_from_depth")

	for k, v in results.items():
		results[k] = np.stack(v, 0)
	return results
