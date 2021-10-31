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


def raw2outputs_simple(raw, z_vals, rays_d):
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	sigma = raw2sigma(raw[..., 0], dists)
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]

	radiance = torch.sigmoid(raw[..., 6:6 + 3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	return radiance_map


def raw2outputs(raw, z_vals, normal, rays_o, rays_d, instance_label_dimension=0, raw_noise_std=0.0, white_bkgd=False, pytest=False,
				is_instance_label_logit=True, label_encoder=None, init_basecolor=None, albedo_mod=None,
				calculate_normal=False, infer_normal=False, brdf_lut=None, network_fn=None, network_query_fn=None, z_vals_constant=None,
				approximate_from_irradiance=False):
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

	irradiance = torch.sigmoid(raw[..., 5])
	irradiance_map = torch.sum(weights * irradiance, -1)

	radiance = torch.sigmoid(raw[..., 6:6+3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	if normal is not None:
		normal_map = torch.sum(weights[..., None] * normal, -2)  # [N_rays, 3]
	else:
		normal_map = None

	if infer_normal:
		inferred_normal = torch.sigmoid(raw[..., 9:9+3])
		inferred_normal_map = torch.sum(weights[..., None] * inferred_normal, -2)
		target_normal = 2 * inferred_normal - 1
		target_normal_map = inferred_normal_map
	else:
		inferred_normal = None
		inferred_normal_map = None
		target_normal = normal
		target_normal_map = normal_map

	# others
	depth_map = torch.sum(weights * z_vals, -1)
	disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
	acc_map = torch.sum(weights, -1)


	if approximate_from_irradiance:
		n_dot_v = torch.sum(rays_d.unsqueeze(1) * target_normal, -1)
		n_dot_v = F.relu(n_dot_v)

		BRDF_2D_LUT_uv = torch.stack([n_dot_v, roughness], -1)
		envBRDF = F.grid_sample(brdf_lut[None, ...], BRDF_2D_LUT_uv[None, ...], align_corners=True)
		envBRDF = envBRDF.permute((0, 2, 3, 1))
		envBRDF = envBRDF.squeeze(0)

		F0 = torch.tensor([0.04, 0.04, 0.04])
		F0 = F0.repeat(*sigma.shape, 1)
		envBRDF_coefficient1 = envBRDF[..., 0]
		envBRDF_coefficient0 = envBRDF[..., 1]
		envBRDF_coefficient1 = torch.stack(3 * [envBRDF_coefficient1], -1)
		fresnel = fresnel_schlick_roughness(n_dot_v, F0, roughness)
		specular = fresnel * envBRDF_coefficient1 + envBRDF_coefficient0[..., None]
		specular_map = torch.sum(weights[..., None] * specular, -2)

		# print(fresnel.shape, "fresnel")
		# print(specular.shape, "specular")
		K_d = 1 - fresnel
		approximated_radiance = (K_d * albedo + specular) * irradiance[..., None]  # [N_rays, N_samples, 3]
		approximated_radiance_map = torch.sum(weights[..., None] * approximated_radiance, -2)
	else:
		n_dot_v = torch.sum(rays_d * target_normal_map, -1)
		n_dot_v = F.relu(n_dot_v)

		BRDF_2D_LUT_uv = torch.stack([n_dot_v, roughness_map], -1)
		envBRDF = F.grid_sample(brdf_lut[None, ...], BRDF_2D_LUT_uv[None, :, None, ...], align_corners=True)
		envBRDF = envBRDF.permute((0, 2, 3, 1))
		envBRDF = envBRDF.squeeze()

		#
		F0 = torch.tensor([0.04, 0.04, 0.04])
		F0 = F0.repeat(*depth_map.shape, 1)
		envBRDF_coefficient1 = envBRDF[..., 0]
		envBRDF_coefficient0 = envBRDF[..., 1]
		envBRDF_coefficient1 = torch.stack(3 * [envBRDF_coefficient1], -1)
		fresnel_map = fresnel_schlick_roughness(n_dot_v, F0, roughness_map)
		specular_map = fresnel_map * envBRDF_coefficient1 + envBRDF_coefficient0[..., None]
		#print(specular_map.shape, "specular_map")
		# specular_map = torch.sum(weights[..., None] * specular, -2)

		#with torch.no_grad():
		reflected_dirs = rays_d - 2 * torch.sum(target_normal_map * rays_d, -1, keepdim=True) * target_normal_map
		x_surface = rays_o + rays_d * depth_map[..., None]

		#print(x_surface.shape, "x_surface")
		reflected_pts = x_surface[..., None, :] + reflected_dirs[..., None, :] * z_vals_constant[..., :, None]
		#print(reflected_pts.shape, "reflected_pts")
		reflected_ray_raw = network_query_fn(reflected_pts, reflected_dirs, network_fn)
		reflected_radiance_map = raw2outputs_simple(reflected_ray_raw, z_vals_constant, reflected_dirs)
		# reflected_radiance.detach_()

		approximated_radiance_map = (1-fresnel_map) * albedo_map * irradiance_map[..., None] + specular_map * reflected_radiance_map
	# [N_rays, N_samples, 3]
	# approximated_radiance_map = torch.sum(weights[..., None] * approximated_radiance, -2)  # [N_rays, 3]



	results = {}
	results["radiance_map"] = radiance_map
	results["color_map"] = approximated_radiance_map
	results["albedo_map"] = albedo_map
	results["roughness_map"] = roughness_map
	results["normal_map"] = normal_map
	results["inferred_normal_map"] = inferred_normal_map
	results["irradiance_map"] = irradiance_map
	results["specular_map"] = specular_map

	results["disp_map"] = disp_map
	results["acc_map"] = acc_map
	results["depth_map"] = depth_map

	results["weights"] = weights

	return results


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, init_basecolor, retraw=False, lindisp=False,
				perturb=0., N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False,
				pytest=False, is_instance_label_logit=False, decompose=False, label_encoder=None, use_viewdirs=None
				,brdf_lut=None, calculate_normal_from_sigma=True):
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
	raw = network_query_fn(pts, viewdirs, network_fn)  # raw = [sigma, albedo_res, indirect_illumination, direct_illumination, basecolor_score]

	# Calculate normal
	if calculate_normal_from_sigma:
		pts.requires_grad = True
		sigma_x = network_query_fn(pts, None, network_fn)
		sigma_x = F.relu(sigma_x)
		sigma_x.backward(torch.ones_like(sigma_x))
		normal = -normalize(pts.grad, dim=-1)
		normal = normal.detach()
		pts.requires_grad = False
	else:
		normal = None
	result = raw2outputs(
		raw, z_vals, normal, rays_o, rays_d, network_fn.instance_label_dimension, raw_noise_std, white_bkgd, pytest=pytest,
		is_instance_label_logit=is_instance_label_logit, label_encoder=label_encoder, init_basecolor=init_basecolor,
		infer_normal=network_fn.infer_normal, brdf_lut=brdf_lut, network_fn=network_fn, network_query_fn=network_query_fn,
		z_vals_constant=z_vals
	)
	z_vals_constant = z_vals

	if N_importance > 0:
		weights = result["weights"]
		z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
		z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
		z_samples = z_samples.detach()

		z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
		pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples + N_importance, 3]

		run_fn = network_fn if network_fine is None else network_fine

		raw = network_query_fn(pts, viewdirs, run_fn)

		# Calculate normal
		if calculate_normal_from_sigma:
			pts.requires_grad = True
			sigma_x = network_query_fn(pts, None, run_fn)
			sigma_x = F.relu(sigma_x)
			sigma_x.backward(torch.ones_like(sigma_x))
			normal = -normalize(pts.grad, dim=-1)
			normal = normal.detach()
			pts.requires_grad = False
		else:
			normal = None

		result_fine = raw2outputs(
			raw, z_vals, normal, rays_o, rays_d, run_fn.instance_label_dimension, raw_noise_std, white_bkgd, pytest=pytest,
			is_instance_label_logit=is_instance_label_logit, label_encoder=label_encoder, init_basecolor=init_basecolor,
			infer_normal=network_fn.infer_normal, brdf_lut=brdf_lut, network_fn=run_fn, network_query_fn=network_query_fn,
			z_vals_constant=z_vals_constant
		)

		for k, v in result.items():
			result_fine[k + "_0"] = result[k]

		result = result_fine

	if retraw:
		result['raw'] = raw

	if N_importance > 0:
		result['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

	for k in result:
		if (torch.isnan(result[k]).any() or torch.isinf(result[k]).any()) and DEBUG:
			print(f"! [Numerical Error] {k} contains nan or inf.")

	return result


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

	K = np.array([
		[focal, 0, 0.5 * W],
		[0, focal, 0.5 * H],
		[0, 0, 1]
	]).astype(np.float32)

	results = {}

	def append_result(render_decomp_results, key_name, index, out_name):
		result_image = render_decomp_results[key_name]
		if result_image is None:
			return
		if out_name not in results:
			results[out_name] = []
		results[out_name].append(result_image.cpu().numpy())
		if savedir is not None:
			result_image_8bit = to8b(results[out_name][-1])
			filename = os.path.join(savedir, (out_name + '_{:03d}.png').format(index))
			imageio.imwrite(filename, result_image_8bit)

	for i, c2w in enumerate(tqdm(render_poses)):
		results_i = render_decomp(H, W, K, chunk=chunk, c2w=c2w[:3, :4], init_basecolor=init_basecolor, **render_kwargs)
		append_result(results_i, "color_map", i, "rgb")
		append_result(results_i, "radiance_map", i, "radiance")
		append_result(results_i, "irradiance_map", i, "irradiance")
		append_result(results_i, "albedo_map", i, "albedo")
		append_result(results_i, "roughness_map", i, "roughness")
		append_result(results_i, "specular_map", i, "specular")
		append_result(results_i, "normal_map", i, "normal")
		append_result(results_i, "inferred_normal_map", i, "inferred_normal")
		append_result(results_i, "disp_map", i, "disp")

	for k, v in results.items():
		results[k] = np.stack(v, 0)
	return results