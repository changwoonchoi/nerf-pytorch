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

import matplotlib.pyplot as plt
from utils.math_utils import get_TBN
from nerf_models.microfacet import Microfacet
from utils.math_utils import *

gamma = 2.2
epsilon_srgb = 1e-12


def rgb_to_srgb(x):
	return torch.pow(x + epsilon_srgb, 1.0/gamma)


def tonemap_reinherd(x):
	return x / (x + 1)


def high_dynamic_range_radiance_f(x):
	return F.relu(x)


def raw2outputs_simple(raw, z_vals, rays_d, coarse_radiance_number=3, detach=False, is_radiance_sigmoid=True):
	if is_radiance_sigmoid:
		radiance_f = torch.sigmoid
	else:
		radiance_f = high_dynamic_range_radiance_f

	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	sigma = raw2sigma(raw[..., 0], dists)

	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
	if detach:
		weights = weights.detach()

	radiance = radiance_f(raw[..., 6:6 + 3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	# (5)-A additional coarse radiance maps
	N = 9
	coarse_radiance_maps = []
	for i in range(coarse_radiance_number):
		coarse_radiance = radiance_f(raw[..., N:N + 3])
		coarse_radiance_map = torch.sum(weights[..., None] * coarse_radiance, -2)
		coarse_radiance_maps.append(coarse_radiance_map)
		N += 3

	return radiance_map, coarse_radiance_maps


def raw2outputs_neigh(rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, network_fn, raw_noise_std, is_radiance_sigmoid):
	if is_radiance_sigmoid:
		radiance_f = torch.sigmoid
	else:
		radiance_f = high_dynamic_range_radiance_f

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

	roughness = torch.sigmoid(raw[..., 4])  # (N_rand * 8, N_sample)
	albedo = torch.sigmoid(raw[..., 1:4])  # (N_rand * 8, N_sample, 3)
	irradiance = radiance_f(raw[..., 5])  # (N_rand * 8, N_sample, )

	roughness_map = torch.sum(weights * roughness, -1)  # (N_rand * 8, )
	albedo_map = torch.sum(weights[..., None] * albedo, 1)  # (N_rand * 8, 3)
	irradiance_map = torch.sum(weights * irradiance, -1)  # (N_rand * 8, )

	if is_radiance_sigmoid:
		radiance_to_ldr = lambda x:x
	else:
		radiance_to_ldr = tonemap
	#radiance_to_ldr = lambda x: None if x is None else torch.pow(radiance_to_ldr_temp(x) + epsilon_srgb, 1.0 / gamma)


	results = {}
	# don't calculate gradient for neighborhood pixels
	results["roughness_map"] = roughness_map
	results["albedo_map"] = albedo_map
	results["irradiance_map"] = radiance_to_ldr(irradiance_map)
	results["weights"] = weights
	return results


def raw2outputs_depth(rays_o, rays_d, z_vals, network_query_fn, network_fn, raw_noise_std):
	# sample points
	pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
	raw = network_query_fn(pts, None, network_fn)

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
	visibility_cum = torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)
	visibility = visibility_cum[:, -1]
	weights = sigma * visibility_cum[:, :-1]

	depth_map = torch.sum(weights * z_vals, -1)

	results = {}
	results["depth_map"] = depth_map
	results["weights"] = weights
	results['visibility'] = visibility
	return results


def raw2outputs(rays_o, rays_d, z_vals, z_vals_constant,
				network_query_fn, network_fn,
				raw_noise_std=0., pytest=False,
				is_neighbor=False,
				is_depth_only=False,
				# calculate_normal_from_sigma_gradient=False,
				# calculate_normal_from_sigma_gradient_surface=False,
				# calculate_normal_from_depth_gradient=False,
				# calculate_normal_from_depth_gradient_direction=False,
				# calculate_normal_from_depth_gradient_epsilon=False,
				# calculate_normal_from_depth_gradient_direction_epsilon=False,
				infer_normal=False,
				infer_normal_at_surface=False,
				normal_mlp=None,
				albedo_mlp=None,
				roughness_mlp=None,
				irradiance_mlp=None,
				brdf_lut=None,
				epsilon=0.01,
				epsilon_direction=0.01,
				gt_values=None,
				target_normal_map_for_radiance_calculation="ground_truth",
				calculate_irradiance_from_gt=False,
				calculate_albedo_from_gt = False,
				calculate_roughness_from_gt=False,
				roughness_exp_coefficient=1.0,
				target_albedo_map_for_radiance_calculation="ground_truth",
				target_roughness_map_for_radiance_calculation="ground_truth",
				target_radiance_map_for_radiance_calculation="ground_truth",
				use_instance=False, is_instance_label_logit=True,
				N_hemisphere_sample_sqrt=0, hemisphere_samples = None,
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
	if kwargs.get("add_object_mode", False):
		return raw2outputs_additional(rays_o, rays_d, z_vals, z_vals_constant,
							   network_query_fn, network_fn,
							   raw_noise_std=raw_noise_std, pytest=pytest,
							   gt_values=gt_values,
							   **kwargs)

	# radiance & gamma correct setting
	is_radiance_sigmoid = not kwargs.get('use_radiance_linear', False)
	gamma_correct = kwargs.get('gamma_correct', False)

	if is_radiance_sigmoid:
		radiance_f = torch.sigmoid
	else:
		radiance_f = high_dynamic_range_radiance_f

	if is_neighbor:
		return raw2outputs_neigh(
			rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, network_fn, raw_noise_std, is_radiance_sigmoid
		)
	elif is_depth_only:
		return raw2outputs_depth(rays_o, rays_d, z_vals, network_query_fn, network_fn, raw_noise_std)
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

	# mask_rgb = gt_values["mask"][:, 0]
	# mask = mask_rgb > 0
	if kwargs.get("edit_roughness", False) or kwargs.get("edit_normal", False):
		mask_rgb = gt_values["mask"][:, 0]
		mask_upper = 0.7 < mask_rgb
		mask_lower = torch.logical_and(0 < mask_rgb, mask_rgb < 0.7)
		# norm_dir = torch.Tensor([0.10610942, 0.99428456, -0.01178994]).to(mask_upper.device)  # side
		# norm_dir = torch.Tensor([0.61423, -0.747126, -0.2542576]).to(mask_upper.device)  # front
		# norm_dir = -torch.Tensor([0.866898, -0.443436, -0.2277105]).to(mask_upper.device)  # front
		theta = 1.57
		phi = 0.6283
		norm_dir = torch.Tensor([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]).to(mask_upper.device)
		"""
		mask = mask_rgb > 0
		# theta = kwargs.get('theta')
		# phi = kwargs.get('phi')
		# norm_dir = torch.Tensor([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]).to(mask.device)
		# norm_dir = torch.Tensor([0.1667948, 0.02185217, 0.985794945]).to(mask.device)  # 3
		norm_dir = torch.Tensor([0., 0., 1.]).to(mask.device)  # 4
		"""
		"""
		mask_1 = mask_rgb > 0.9
		mask_2 = torch.logical_and(0.75 < mask_rgb, mask_rgb <= 0.9)
		mask_3 = torch.logical_and(0.5 < mask_rgb, mask_rgb <= 0.75)
		mask_4 = torch.logical_and(0.25 < mask_rgb, mask_rgb <= 0.5)
		"""

	# (0) get sigma
	raw2sigma = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)
	sigma = raw2sigma(raw[..., 0] + noise, dists)

	# (1) get weight from sigma
	weights = sigma * torch.cumprod(torch.cat([torch.ones((sigma.shape[0], 1)), 1. - sigma + 1e-10], -1), -1)[:, :-1]
	weights_detached = weights.detach()

	# (2) get depth / disp / acc map
	depth_map = torch.sum(weights * z_vals, -1)
	target_depth_map = depth_map
	if kwargs.get("depth_map_from_ground_truth", False):
		target_depth_map = gt_values["depth"][..., 0]

	disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
	acc_map = torch.sum(weights, -1)

	# (3) get surface point surface_x
	x_surface = rays_o + rays_d * target_depth_map[..., None]
	x_surface.detach_()

	if kwargs.get("edit_roughness", False) or kwargs.get("edit_normal", False):
		# x_surface[mask_lower] = rays_o[mask_lower] + 0.95 * rays_d[mask_lower] * target_depth_map[..., None][mask_lower]
		"""
		# print(x_surface[mask].shape)
		t = -(rays_o[:, 2:] + 0.9) / rays_d[:, 2:]
		x_surface[mask] = rays_o[mask] + rays_d[mask] * t[mask]
		# x_surface[mask] += 0.15 * norm_dir
		# x_surface_z = x_surface[mask].clone()
		# x_surface_z[:, 2] = -0.95
		# x_surface[mask] = 0.
		# x_surface[mask] += x_surface_z
		# print(x_surface[mask])
		"""
		pass

	# (4A) calculate normal from sigma gradient or read ground_truth value
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

	target_normal_map = None

	"""
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
	"""


	# (5) other values
	albedo = torch.sigmoid(raw[..., 1:4])
	albedo_map = torch.sum(weights_detached[..., None] * albedo, -2)

	roughness = torch.sigmoid(raw[..., 4])
	roughness_map = torch.sum(weights_detached * roughness, -1)

	irradiance = radiance_f(raw[..., 5])
	irradiance_map = torch.sum(weights_detached * irradiance, -1)

	# (5) other values
	if albedo_mlp is not None:
		raw_albedo = network_query_fn(pts, None, albedo_mlp)
		albedo = torch.sigmoid(raw_albedo[..., 0:3])
		albedo_map = torch.sum(weights_detached[..., None] * albedo, -2)

	if roughness_mlp is not None:
		raw_roughness = network_query_fn(pts, None, roughness_mlp)
		roughness = torch.sigmoid(raw_roughness[..., 0])
		roughness_map = torch.sum(weights_detached * roughness, -1)

	if irradiance_mlp is not None:
		raw_irradiance = network_query_fn(pts, None, irradiance_mlp)
		irradiance = torch.sigmoid(raw_irradiance[..., 0])
		irradiance_map = torch.sum(weights_detached * irradiance, -1)

	radiance = radiance_f(raw[..., 6:6 + 3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	if use_instance:
		if is_instance_label_logit:
			instance_score = raw[..., 9 + 3 * network_fn.coarse_radiance_number:]  # no sigmoid -> just use logits
		else:
			instance_score = torch.sigmoid(raw[..., 9 + 3 * network_fn.coarser_radiance_number:])  # (N_rays, N_samples, instance_label_dimension)
		instance_map = torch.sum(weights[..., None] * instance_score, -2)  # (N_rays, instance_label_dimension)
	else:
		instance_map = None

	# (5)-A additional coarse radiance maps
	N = 9
	coarse_radiance_maps = []
	for i in range(network_fn.coarse_radiance_number):
		coarse_radiance = radiance_f(raw[..., N:N + 3])
		coarse_radiance_map = torch.sum(weights_detached[..., None] * coarse_radiance, -2)
		coarse_radiance_maps.append(coarse_radiance_map)

		N += 3
	"""
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
	"""

	target_albedo_map = albedo_map
	if calculate_albedo_from_gt:
		target_albedo_map = gt_values["albedo"]

	target_roughness_map = roughness_map
	if calculate_roughness_from_gt:
		target_roughness_map = gt_values["roughness"][...,0]

	target_irradiance_map = irradiance_map[...,None]
	if calculate_irradiance_from_gt:
		target_irradiance_map = gt_values["irradiance"]
		#target_irradiance_map = torch.pow(target_irradiance_map, gamma)
		#target_irradiance_map = target_irradiance_map / torch.clip(1 - target_irradiance_map, 0.000001, 1)


	# target_albedo_map = None
	# target_roughness_map = None
	# target_irradiance_map = None
	target_binormal_map = None
	target_tangent_map = None
	approximated_radiance_map = None
	specular_map = None
	diffuse_map = None
	min_irradiance_map = None
	max_irradiance_map = None
	visibility_average_map = None
	n_dot_v = None
	reflected_radiance_map = None
	prefiltered_reflected_map = None
	reflected_coarse_radiance_map = []
	if kwargs.get('approximate_radiance', False):

		# calculate normal only approximate radiance
		if target_normal_map_for_radiance_calculation == "normal_map_from_sigma_gradient":
			normal_map_from_sigma_gradient = get_normal_from_sigma_gradient(pts, weights_detached, network_query_fn, network_fn)
			target_normal_map = normal_map_from_sigma_gradient
		elif target_normal_map_for_radiance_calculation == "normal_map_from_sigma_gradient_surface":
			normal_map_from_sigma_gradient_surface = get_normal_from_sigma_gradient_surface(x_surface, network_query_fn, network_fn)
			target_normal_map = normal_map_from_sigma_gradient_surface
		elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient":
			normal_map_from_depth_gradient = get_normal_from_depth_gradient(rays_o, rays_d, network_query_fn, network_fn, z_vals)
			normal_map_from_depth_gradient.detach_()
			target_normal_map = normal_map_from_depth_gradient
		elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient_epsilon":
			with torch.no_grad():
				normal_map_from_depth_gradient_epsilon = get_normal_from_depth_gradient_epsilon(rays_o, rays_d, network_query_fn, network_fn, z_vals, epsilon=epsilon)
			target_normal_map = normal_map_from_depth_gradient_epsilon
		elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient_direction":
			normal_map_from_depth_gradient_direction = get_normal_from_depth_gradient_direction(rays_o, rays_d, network_query_fn, network_fn, z_vals)
			normal_map_from_depth_gradient_direction.detach_()
			target_normal_map = normal_map_from_depth_gradient_direction
		elif target_normal_map_for_radiance_calculation == "normal_map_from_depth_gradient_direction_epsilon":
			with torch.no_grad():
				normal_map_from_depth_gradient_direction_epsilon = get_normal_from_depth_gradient_direction_epsilon(rays_o, rays_d, network_query_fn, network_fn, z_vals, epsilon=epsilon_direction)
			target_normal_map = normal_map_from_depth_gradient_direction_epsilon
		elif target_normal_map_for_radiance_calculation == "ground_truth":
			target_normal_map = normalize(2 * gt_values["normal"] - 1, dim=-1)
		elif target_normal_map_for_radiance_calculation == "inferred_normal_map":
			target_normal_map = inferred_normal_map
		else:
			raise ValueError

		# Edit!!
		if kwargs.get("edit_roughness", False) or kwargs.get("edit_normal", False):
			"""
			if kwargs.get("edit_roughness", False):
				target_roughness_map[mask] = 0.
			if kwargs.get("edit_normal", False):
				target_normal_map[mask] = norm_dir
				# roi_albedo = target_albedo_map[mask]
				# roi_albedo_norm = torch.linalg.norm(roi_albedo, dim=-1)
				# th = 0.3
				# albedo_mask = roi_albedo_norm < th
				# target_albedo_map[mask][albedo_mask] += 0.2
				# target_albedo_map[mask] = torch.tensor([70/255., 43/255., 26/255.])
				# target_albedo_map[mask] = torch.clamp(target_albedo_map[mask], min=0.3, max=0.5)
			if kwargs.get("edit_albedo", False):
				target_albedo_map[mask] = gt_values["edit_albedo"][mask]
				target_irradiance_map[mask] *= 0.7
			"""
			if kwargs.get("edit_roughness", False):
				target_roughness_map[mask_upper] = 1.
				target_roughness_map[mask_lower] = 1.
				target_normal_map[mask_upper] = norm_dir
				target_normal_map[mask_lower] = norm_dir
				# norm_dir = torch.Tensor([0.10610942, 0.99428456, -0.01178994]).to(mask.device)  # side
				norm_dir = torch.Tensor([0.57674571, -0.64736752, -0.49827682]).to(mask_upper.device)  # front
				target_albedo_map[mask_lower] = torch.Tensor([210 / 255., 217 / 255., 192 / 255.]).to(mask_lower.device)  # green
				target_albedo_map[mask_upper] = torch.Tensor([240 / 255., 188 / 255., 175 / 255.]).to(mask_lower.device)  # pink
			if kwargs.get("edit_normal", False):
				target_normal_map[mask_lower] = norm_dir
			if kwargs.get("edit_albedo", False):
				target_albedo_map[mask_lower] = gt_values["edit_albedo"][mask_lower]
				target_irradiance_map[mask_lower] *= 0.7
			"""
			if kwargs.get('edit_roughness', False):
				target_roughness_map[mask_1] = 1.
				target_roughness_map[mask_2] = 1.
				target_roughness_map[mask_3] = 1.
				target_roughness_map[mask_4] = 1.
			if kwargs.get('edit_normal', False):
				target_normal_map[mask_1] = gt_values["normal"][mask_1]
				target_normal_map[mask_2] = gt_values["normal"][mask_2]
				target_normal_map[mask_3] = gt_values["normal"][mask_3]
				target_normal_map[mask_4] = gt_values["normal"][mask_4]
				# target_albedo_map[mask_1] *= torch.tensor([300 / 255., 50 / 255., 50 / 255.])
				# target_albedo_map[mask_2] *= torch.tensor([300 / 255., 50 / 255., 50 / 255.])
				# target_albedo_map[mask_3] *= torch.tensor([50 / 255., 140 / 255., 50 / 255.])
				# target_albedo_map[mask_4] *= torch.tensor([50 / 255., 140 / 255., 50 / 255.])
				target_albedo_map[mask_1] *= torch.tensor([222 / 255., 83 / 255., 113 / 255.])
				target_albedo_map[mask_2] *= torch.tensor([222 / 255., 83 / 255., 113 / 255.])
				target_albedo_map[mask_3] *= torch.tensor([31 / 255., 115 / 255., 34 / 255.])
				target_albedo_map[mask_4] *= torch.tensor([31 / 255., 115 / 255., 34 / 255.])
			"""
			if kwargs.get("edit_irradiance", False):
				pass
				# target_irradiance_map[mask_1] = gt_values["irradiance"][mask_1][:, 0:1]


		n_dot_v = torch.sum(-rays_d * target_normal_map, -1)
		n_dot_v = torch.clip(n_dot_v, 0, 1)

		if kwargs.get('use_monte_carlo_integration', True):
			# N_hemisphere_sample = N_hemisphere_sample_sqrt * N_hemisphere_sample_sqrt
			#hemisphere_samples = get_hemisphere_samples(N_hemisphere_sample_sqrt)
			#hemisphere_samples = torch.Tensor(hemisphere_samples)

			# print(hemisphere_samples.shape, "HEMI SHAPEEEEEEEEEEEEEEEEEE")
			microfacet = Microfacet()
			# with torch.no_grad():
			target_binormal_map, target_tangent_map = get_TBN(target_normal_map)
			target_TBNs = torch.stack([target_tangent_map, target_binormal_map, target_normal_map], dim=-1)
			target_sampled_hemisphere_dirs = torch.tensordot(target_TBNs, hemisphere_samples, dims=[[2], [1]])
			target_sampled_hemisphere_dirs = torch.permute(target_sampled_hemisphere_dirs, (0, 2, 1))

			sampled_dirs = torch.reshape(target_sampled_hemisphere_dirs, [-1, 3])
			sampled_pos = x_surface.repeat_interleave(hemisphere_samples.shape[0], dim=0)
			monte_carlo_integration_method = kwargs.get('monte_carlo_integration_method', 'surface')
			if monte_carlo_integration_method=='volume':
				# use only have of z_vals
				z_vals_constant = z_vals_constant[...,::2]
				z_vals_constant_repeated = z_vals_constant.repeat_interleave(hemisphere_samples.shape[0], dim=0)

				pts = sampled_pos[..., None, :] + sampled_dirs[..., None, :] * z_vals_constant_repeated[..., :, None]
				reflected_ray_raw = network_query_fn(pts, sampled_dirs, network_fn)
				sampled_dir_radiance, _ = raw2outputs_simple(reflected_ray_raw, z_vals_constant_repeated, sampled_dirs, 0, detach=True, is_radiance_sigmoid=is_radiance_sigmoid)
			elif monte_carlo_integration_method == 'surface':
				# calculate surface point
				z_vals_constant_repeated = z_vals_constant.repeat_interleave(hemisphere_samples.shape[0], dim=0)

				with torch.no_grad():
					result = raw2outputs_depth(sampled_pos, sampled_dirs, z_vals_constant_repeated, network_query_fn, network_fn, raw_noise_std)
					x2_surface = sampled_pos + sampled_dirs * result['depth_map'][..., None]

				reflected_ray_raw = network_query_fn(x2_surface[..., None, :], sampled_dirs, network_fn)
				sampled_dir_radiance = radiance_f(reflected_ray_raw[..., 6:6 + 3])

			elif monte_carlo_integration_method == 'depth_mlp':
				sampled_dir_depth_map = network_query_fn(sampled_pos[..., None, :], sampled_dirs, kwargs["depth_mlp"])
				sampled_dir_depth_map = F.relu(sampled_dir_depth_map[..., 0])
				sampled_dir_x_surface = sampled_pos + sampled_dirs * sampled_dir_depth_map

				reflected_ray_raw = network_query_fn(sampled_dir_x_surface[..., None, :], sampled_dirs, network_fn)
				sampled_dir_radiance = radiance_f(reflected_ray_raw[..., 6:6 + 3])

			elif monte_carlo_integration_method == 'env_map':
				z_vals_constant_repeated = z_vals_constant.repeat_interleave(hemisphere_samples.shape[0], dim=0)
				with torch.no_grad():
					result = raw2outputs_depth(sampled_pos, sampled_dirs, z_vals_constant_repeated, network_query_fn, network_fn, raw_noise_std)
					visibility = result['visibility']
				env_map = kwargs.get('env_map')
				radiance = env_map.get_radiance(sampled_pos, sampled_dirs)
				sampled_dir_radiance = visibility[..., None] * radiance
				visibility_average_map = torch.reshape(visibility, [-1, hemisphere_samples.shape[0]])
				visibility_average_map = torch.mean(visibility_average_map, dim=-1)
			else:
				sampled_dir_radiance = None

			sampled_dir_radiance = torch.reshape(sampled_dir_radiance, (-1, *hemisphere_samples.shape))

			if not kwargs.get('use_gradient_for_incident_radiance', False):
				sampled_dir_radiance = sampled_dir_radiance.detach()

			brdf_specular, brdf_diffuse, l_dot_n = microfacet(target_sampled_hemisphere_dirs, -rays_d, target_normal_map, target_albedo_map, target_roughness_map[..., None])

			# Multiply solid angle corresponding to the lighting sample
			# equal area sampling on hemisphere --> solid angle = 2 * pi / N_sample
			specular_map = torch.mean(sampled_dir_radiance * brdf_specular * l_dot_n, dim=1) * 2 * np.pi
			diffuse_map = torch.mean(sampled_dir_radiance * brdf_diffuse * l_dot_n, dim=1) * 2 * np.pi
			irradiance_map = torch.mean(sampled_dir_radiance * l_dot_n * (1/np.pi), dim=1) * 2 * np.pi
			max_irradiance_map = torch.amax(sampled_dir_radiance * l_dot_n * (1/np.pi), dim=1) * 2 * np.pi
			min_irradiance_map = torch.amin(sampled_dir_radiance * l_dot_n * (1/np.pi), dim=1) * 2 * np.pi

			# i = 22
			# brdf = ((brdf_specular + brdf_diffuse) * l_dot_n)[i]
			# result = target_sampled_hemisphere_dirs[i]
			# dir_vec = -rays_d[i]
			# normal_vec = target_normal_map[i]
			# brdf = brdf.cpu().detach().numpy()
			# result = result.cpu().detach().numpy()
			# dir_vec = dir_vec.cpu().detach().numpy()
			# normal_vec = normal_vec.cpu().detach().numpy()
			#
			# print(brdf.shape, "brdf")
			# print(result.shape, "result")
			#
			# fig = plt.figure()
			# ax = fig.gca(projection='3d')
			# ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=brdf[:, 0])
			# ax.plot([0, dir_vec[0]], [0, dir_vec[1]], [0, dir_vec[2]], 'r')
			# ax.plot([0, normal_vec[0]], [0, normal_vec[1]], [0, normal_vec[2]], 'g')
			#
			# ax.set_xlim3d(-1, 1)
			# ax.set_ylim3d(-1, 1)
			# ax.set_zlim3d(-1, 1)
			# plt.show()
			#
			# print(brdf_specular[0, ...], "brdf_specular")
			# print(brdf_diffuse[0, ...], "brdf_diffuse")

			#print(torch.min(irradiance_map), "irradiance MIN")
			#print(torch.max(irradiance_map), "irradiance MAX")

			# irradiance_map = torch.mean(sampled_dir_radiance * l_dot_n, dim=1) * 2 * np.pi

			approximated_radiance_map = specular_map + diffuse_map

			brdf_specular_nan = torch.any(brdf_specular.isnan())
			brdf_diffuse_nan = torch.any(brdf_diffuse.isnan())

			if brdf_specular_nan.item() > 0:
				print("brdf_specular_nan")
			if brdf_diffuse_nan.item() > 0:
				print("brdf_diffuse_nan")
		else:
			# (7) calculate color from split-sum approximation

			# grid_sample input is  [-1, 1] x [-1, 1]
			BRDF_2D_LUT_uv = torch.stack([2 * n_dot_v - 1, 2 * target_roughness_map - 1], -1)
			envBRDF = F.grid_sample(brdf_lut[None, ...], BRDF_2D_LUT_uv[None, :, None, ...], align_corners=True)
			envBRDF = envBRDF.permute((0, 2, 3, 1))
			envBRDF = envBRDF.squeeze()

			# dielectric
			F0 = torch.tensor([0.04, 0.04, 0.04])
			F0 = F0.repeat(*depth_map.shape, 1)
			target_metallic_map = (1-target_roughness_map)[..., None]
			F0 = F0 * (1-target_metallic_map) + target_albedo_map * target_metallic_map

			envBRDF_coefficient1 = envBRDF[..., 0]
			envBRDF_coefficient0 = envBRDF[..., 1]
			envBRDF_coefficient1 = torch.stack(3 * [envBRDF_coefficient1], -1)
			fresnel_map = fresnel_schlick_roughness(n_dot_v, F0, target_roughness_map)
			if kwargs.get('lut_coefficient') == 'F':
				specular_map = fresnel_map * envBRDF_coefficient1 + envBRDF_coefficient0[..., None]
			elif kwargs.get('lut_coefficient') == 'F0':
				specular_map = F0 * envBRDF_coefficient1 + envBRDF_coefficient0[..., None]
			else:
				raise ValueError
			reflected_dirs = rays_d - 2 * torch.sum(target_normal_map * rays_d, -1, keepdim=True) * target_normal_map
			reflected_pts = x_surface[..., None, :] + reflected_dirs[..., None, :] * z_vals_constant[..., :, None]

			if not kwargs.get('use_gradient_for_incident_radiance', False):

				with torch.no_grad():
					reflected_ray_raw = network_query_fn(reflected_pts, reflected_dirs, network_fn)
					reflected_radiance_map, reflected_coarse_radiance_map = raw2outputs_simple(reflected_ray_raw, z_vals_constant, reflected_dirs, is_radiance_sigmoid=is_radiance_sigmoid)

					prefiltered_env_maps = torch.stack([reflected_radiance_map] + reflected_coarse_radiance_map, dim=1)
			else:
				reflected_ray_raw = network_query_fn(reflected_pts, reflected_dirs, network_fn)
				reflected_radiance_map, reflected_coarse_radiance_map = raw2outputs_simple(reflected_ray_raw, z_vals_constant, reflected_dirs, is_radiance_sigmoid=is_radiance_sigmoid)

				prefiltered_env_maps = torch.stack([reflected_radiance_map] + reflected_coarse_radiance_map, dim=1)

			N_pref = len(reflected_coarse_radiance_map) + 1
			if kwargs.get("correct_depth_for_prefiltered_radiance_infer", False):
				depth_0 = (kwargs["far"] + kwargs["near"]) * 0.5
				depth_map_detached = depth_map.detach()
				mipmap_level = roughness_map * depth_map_detached / depth_0[..., 0]
				mipmap_level = torch.clip(mipmap_level, 0, 1)
			else:
				mipmap_level = roughness_map

			mipmap_index1 = (mipmap_level * (N_pref - 1)).long()
			mipmap_index1 = torch.clip(mipmap_index1, 0, N_pref - 1)
			mipmap_index2 = torch.clip(mipmap_index1 + 1, 0, N_pref - 1)
			mipmap_remainder = ((mipmap_level * (N_pref - 1)) - mipmap_index1)[..., None]
			prefiltered_reflected_map = \
				(1-mipmap_remainder) * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index1] +\
				mipmap_remainder * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index2]

			diffuse_map = (1 - fresnel_map) * (1-target_metallic_map) * target_albedo_map * target_irradiance_map
			specular_map = specular_map * prefiltered_reflected_map
			approximated_radiance_map = diffuse_map + specular_map

	# [N_rays, N_samples, 3]

	# approximated_radiance_map = None
	# specular_map = None

	# Organize results
	results = {}

	if is_radiance_sigmoid:
		ldr_f = lambda x: x
	else:
		ldr_f = lambda x: tonemap_reinherd(x)

	if gamma_correct:
		gamma_correct_f = lambda x: rgb_to_srgb(x)
	else:
		gamma_correct_f = lambda x: x
	#print(is_radiance_sigmoid, "RADIANCE SIGMOID", gamma_correct, "GAMMA CORRECT")

	output_f = lambda x: x if x is None else gamma_correct_f(ldr_f(x))
	albedo_f = lambda x: x if x is None else gamma_correct_f(x)

	results["color_map"] = output_f(approximated_radiance_map)
	results["radiance_map"] = output_f(radiance_map)
	for k in range(len(coarse_radiance_maps)):
		results["radiance_map_%d" % (k + 1)] = output_f(coarse_radiance_maps[k])
	for k in range(len(reflected_coarse_radiance_map)):
		results["reflected_coarse_radiance_map_%d" % (k + 1)] = output_f(reflected_coarse_radiance_map[k])

	results["irradiance_map"] = output_f(target_irradiance_map)
	results["min_irradiance_map"] = output_f(min_irradiance_map)
	results["max_irradiance_map"] = output_f(max_irradiance_map)
	results["reflected_radiance_map"] = output_f(reflected_radiance_map)
	results["prefiltered_reflected_map"] = output_f(prefiltered_reflected_map)

	results["albedo_map"] = albedo_f(target_albedo_map)
	results["roughness_map"] = target_roughness_map
	results["specular_map"] = output_f(specular_map)
	results["diffuse_map"] = output_f(diffuse_map)
	results["n_dot_v_map"] = n_dot_v
	results["instance_map"] = instance_map
	results["visibility_average_map"] = visibility_average_map

	# results["normal_map_from_sigma_gradient"] = normal_map_from_sigma_gradient
	# results["normal_map_from_sigma_gradient_surface"] = normal_map_from_sigma_gradient_surface
	# results["normal_map_from_depth_gradient"] = normal_map_from_depth_gradient
	# results["normal_map_from_depth_gradient_direction"] = normal_map_from_depth_gradient_direction
	# results["normal_map_from_depth_gradient_epsilon"] = normal_map_from_depth_gradient_epsilon
	# results["normal_map_from_depth_gradient_direction_epsilon"] = normal_map_from_depth_gradient_direction_epsilon

	results["inferred_normal_map"] = inferred_normal_map
	results["target_normal_map"] = target_normal_map
	results["target_binormal_map"] = target_binormal_map
	results["target_tangent_map"] = target_tangent_map

	results["disp_map"] = disp_map
	results["acc_map"] = acc_map
	results["depth_map"] = depth_map
	results["target_depth_map"] = target_depth_map

	results["weights"] = weights

	return results


def raw2outputs_additional(rays_o, rays_d, z_vals, z_vals_constant,
				network_query_fn, network_fn,
				raw_noise_std=0., pytest=False,
				gt_values=None,
				**kwargs):
	is_radiance_sigmoid = not kwargs.get('use_radiance_linear', False)

	if is_radiance_sigmoid:
		radiance_f = torch.sigmoid
	else:
		radiance_f = high_dynamic_range_radiance_f

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
	#weights_detached = weights.detach()

	# (2) get depth / disp / acc map
	depth_map = torch.sum(weights * z_vals, -1)

	# Radiance & Additional radiance
	radiance = radiance_f(raw[..., 6:6 + 3])
	radiance_map = torch.sum(weights[..., None] * radiance, -2)

	N = 9
	coarse_radiance_maps = []
	for i in range(network_fn.coarse_radiance_number):
		coarse_radiance = radiance_f(raw[..., N:N + 3])
		coarse_radiance_map = torch.sum(weights[..., None] * coarse_radiance, -2)
		coarse_radiance_maps.append(coarse_radiance_map)

		N += 3

	# depth
	gt_depth_map = gt_values["depth"][..., 0]
	mask = (depth_map < gt_depth_map) | (gt_depth_map == 0)
	target_depth_map = torch.where(mask, depth_map, gt_depth_map)

	# (3) get surface point surface_x
	x_surface = rays_o + rays_d * target_depth_map[..., None]
	x_surface.detach_()

	# (4A) calculate normal from sigma gradient or read ground_truth value
	target_normal_map = normalize(2 * gt_values["normal"] - 1, dim=-1)

	reflected_dirs = rays_d - 2 * torch.sum(target_normal_map * rays_d, -1, keepdim=True) * target_normal_map
	reflected_pts = x_surface[..., None, :] + reflected_dirs[..., None, :] * z_vals_constant[..., :, None]

	reflected_ray_raw = network_query_fn(reflected_pts, reflected_dirs, network_fn)
	reflected_radiance_map, reflected_coarse_radiance_map = raw2outputs_simple(reflected_ray_raw, z_vals_constant, reflected_dirs, radiance_f=radiance_f)
	prefiltered_env_maps = torch.stack([reflected_radiance_map] + reflected_coarse_radiance_map, dim=1)

	#roughness_map = kwargs["roughness"]
	roughness_map = torch.ones_like(depth_map) * kwargs["roughness"]
	#print(roughness_map, "ROUGHNESS!!")

	N_pref = len(reflected_coarse_radiance_map) + 1
	mipmap_index1 = (roughness_map * (N_pref - 1)).long()
	mipmap_index1 = torch.clip(mipmap_index1, 0, N_pref - 1)
	mipmap_index2 = torch.clip(mipmap_index1 + 1, 0, N_pref - 1)
	mipmap_remainder = ((roughness_map * (N_pref - 1)) - mipmap_index1)[..., None]
	prefiltered_reflected_map = \
		(1 - mipmap_remainder) * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index1] + \
		mipmap_remainder * prefiltered_env_maps[torch.arange(prefiltered_env_maps.size(0)), mipmap_index2]

	#reflected_radiance_map = torch.ones_like(reflected_radiance_map) * roughness + (1-roughness) * reflected_radiance_map
	new_radiance_map = torch.where(mask[...,None], radiance_map, prefiltered_reflected_map)

	# Organize results
	results = {}
	results["color_map"] = new_radiance_map
	results["radiance_map"] = radiance_map
	results["weights"] = weights

	return results


def render_rays(ray_batch, network_fn, network_query_fn, N_samples, init_basecolor, retraw=False, lindisp=False,
				perturb=0., N_importance=0, network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False,
				pytest=False, is_neighbor=False, **kwargs):
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
		rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, network_fn, raw_noise_std, pytest, is_neighbor, near=near, far=far, **kwargs
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
			rays_o, rays_d, z_vals, z_vals_constant, network_query_fn, run_fn, raw_noise_std, pytest, is_neighbor, near=near, far=far, **kwargs
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


def batchify_rays(rays_flat, chunk=1024 * 32, label_encoder=None, is_neighbor=False, **kwargs):
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
		ret = render_rays(rays_flat[i:i + chunk], label_encoder=label_encoder, is_neighbor=is_neighbor, **kwargs)
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
		c2w_staticcam=None, label_encoder=None, is_neighbor=False, is_depth_only=False, **kwargs
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
		# if rays_neigh is not None:
		# 	rays_o_neigh, rays_d_neigh = rays_neigh
		# 	viewdirs_neigh = rays_d_neigh
		# 	viewdirs_neigh = viewdirs_neigh / torch.norm(viewdirs_neigh, dim=-1, keepdim=True)
		# 	viewdirs_neigh = torch.reshape(viewdirs_neigh, [-1, 3]).float()
		# 	sh_neigh = rays_d_neigh.shape
		# 	if ndc:
		# 		rays_o_neigh, rays_d_neigh = ndc_rays(H, W, K[0][0], 1., rays_o_neigh, rays_d_neigh)
		# 	rays_o_neigh = torch.reshape(rays_o_neigh, [-1, 3]).float()
		# 	rays_d_neigh = torch.reshape(rays_d_neigh, [-1, 3]).float()
		# 	near_neigh, far_neigh = near * torch.ones_like(rays_d_neigh[..., :1]), far * torch.ones_like(rays_d_neigh[..., :1])
		# 	rays_neighbor = torch.cat([rays_o_neigh, rays_d_neigh, near_neigh, far_neigh], -1)
		# 	rays_neighbor = torch.cat([rays_neighbor, viewdirs_neigh], -1)
		# 	all_ret_neigh = batchify_rays(rays_neighbor, chunk, label_encoder=label_encoder, neighbor=True, **kwargs)
		# 	for k in all_ret_neigh:
		# 		k_sh_neigh = list(sh_neigh[:-1]) + list(all_ret_neigh[k].shape[1:])
		# 		all_ret_neigh[k] = torch.reshape(all_ret_neigh[k], k_sh_neigh)

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
	all_ret = batchify_rays(rays, chunk, label_encoder=label_encoder, is_neighbor=is_neighbor, is_depth_only=is_depth_only, **kwargs)
	for k in all_ret:
		k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
		all_ret[k] = torch.reshape(all_ret[k], k_sh)
	return all_ret


from dataset.dataset_interface import NerfDataset


def render_decomp_path(
		dataset_test: NerfDataset, hwf, K, chunk, render_kwargs, savedir=None, render_factor=0, init_basecolor=None,
		calculate_normal_from_depth_map=False, use_instance=False, label_encoder=None, **kwargs
):
	H, W, focal = hwf
	render_poses = dataset_test.poses
	if K is None:
		if render_factor != 0:
			# Render downsampled for speed
			H = H // render_factor
			W = W // render_factor
			focal = focal / render_factor

		intrinsic = np.array([
			[focal, 0, 0.5 * W],
			[0, focal, 0.5 * H],
			[0, 0, 1]
		]).astype(np.float32)
	else:
		H = H // render_factor
		W = W // render_factor
		intrinsic = K / render_factor

	results = {}

	def append_result(render_decomp_results, key_name, index, out_name, label_encoder=None, theta=None, phi=None):
		if key_name not in render_decomp_results:
			return
		result_image = render_decomp_results[key_name]
		if result_image is None:
			return
		if out_name not in results:
			results[out_name] = []
		if "normal" in out_name or 'tangent' in out_name:
			result_image = (result_image + 1) * 0.5
		elif "depth" in key_name:
			# depth to disp
			result_image = result_image / (dataset_test.far * 0.1)
			result_image = 1. / torch.max(1e-10 * torch.ones_like(result_image), result_image)
		elif "instance" in key_name:
			result_image = label_encoder.encoded_label_to_colored_label(result_image)

		results[out_name].append(result_image.cpu().numpy())
		if savedir is not None:
			result_image_8bit = to8b(results[out_name][-1])
			if theta is not None and phi is not None:
				filename = os.path.join(savedir, (out_name + '_{:03d}_{}_{}.png').format(index, theta, phi))
			else:
				filename = os.path.join(savedir, (out_name + '_{:03d}.png').format(index))
			imageio.imwrite(filename, result_image_8bit)

	for i, c2w in enumerate(tqdm(render_poses)):
		roughness_t = np.sin(i / 100 * np.pi * 2) * 0.5 + 0.5

		gt_values = dataset_test.get_resized_normal_albedo(render_factor, i)
		for k in gt_values.keys():
			gt_values[k] = torch.reshape(gt_values[k], [-1, gt_values[k].shape[-1]])
		results_i = render_decomp(
			H, W, K=intrinsic, chunk=chunk, c2w=c2w[:3, :4], init_basecolor=init_basecolor, gt_values=gt_values,
			use_instance=use_instance, label_encoder=label_encoder, roughness=roughness_t, **render_kwargs, **kwargs
		)
		append_result(results_i, "color_map", i, "rgb", theta=kwargs.get('theta', None), phi=kwargs.get('phi', None))
		append_result(results_i, "radiance_map", i, "radiance")
		for k in range(render_kwargs["coarse_radiance_number"]):
			append_result(results_i, "radiance_map_%d" % (k+1), i, "radiance_%d" % (k+1))
			append_result(results_i, "reflected_coarse_radiance_map_%d" % (k + 1), i, "reflected_coarse_radiance_%d" % (k + 1))


		append_result(results_i, "irradiance_map", i, "irradiance")
		append_result(results_i, "max_irradiance_map", i, "max_irradiance")
		append_result(results_i, "min_irradiance_map", i, "min_irradiance")

		append_result(results_i, "albedo_map", i, "albedo")
		append_result(results_i, "reflected_radiance_map", i, "reflected_radiance")
		append_result(results_i, "prefiltered_reflected_map", i, "prefiltered_reflected")

		append_result(results_i, "roughness_map", i, "roughness", theta=kwargs.get('theta', None), phi=kwargs.get('phi', None))
		append_result(results_i, "specular_map", i, "specular")
		append_result(results_i, "diffuse_map", i, "diffuse")
		append_result(results_i, "n_dot_v_map", i, "n_dot_v", theta=kwargs.get('theta', None), phi=kwargs.get('phi', None))

		# append_result(results_i, "normal_map_from_sigma_gradient", i, "normal_map_from_sigma_gradient")
		# append_result(results_i, "normal_map_from_sigma_gradient_surface", i, "normal_map_from_sigma_gradient_surface")
		# append_result(results_i, "normal_map_from_depth_gradient", i, "normal_map_from_depth_gradient")
		# append_result(results_i, "normal_map_from_depth_gradient_direction", i, "normal_map_from_depth_gradient_direction")
		# append_result(results_i, "normal_map_from_depth_gradient_epsilon", i, "normal_map_from_depth_gradient_epsilon")
		# append_result(results_i, "normal_map_from_depth_gradient_direction_epsilon", i, "normal_map_from_depth_gradient_direction_epsilon")

		append_result(results_i, "inferred_normal_map", i, "inferred_normal_map")
		append_result(results_i, "target_normal_map", i, "target_normal_map", theta=kwargs.get('theta', None), phi=kwargs.get('phi', None))
		append_result(results_i, "target_binormal_map", i, "target_binormal_map")
		append_result(results_i, "target_tangent_map", i, "target_tangent_map")
		append_result(results_i, "visibility_average_map", i, "visibility_average_map")

		append_result(results_i, "inferred_depth_map", i, "inferred_disp")

		append_result(results_i, "disp_map", i, "disp")
		append_result(results_i, "depth_map", i, "depth")
		append_result(results_i, "target_depth_map", i, "target_depth")

		append_result(results_i, "instance_map", i, "instance_map", label_encoder=label_encoder)

		if "depth_map" in results_i:
			depth_image = results_i["depth_map"]
			results_i["normal_map_from_depth_map"] = depth_to_normal_image_space(depth_image, c2w[:3, :4], K)
			append_result(results_i, "normal_map_from_depth_map", i, "normal_from_depth")

	for k, v in results.items():
		results[k] = np.stack(v, 0)
	return results
