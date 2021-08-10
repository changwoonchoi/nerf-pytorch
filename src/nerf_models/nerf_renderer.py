from nerf_models.nerf_renderer_helper import *


def raw2outputs(raw, z_vals, rays_d, instance_num=0, raw_noise_std=0, white_bkgd=False, pytest=False, decompose=False):
	"""
	Transforms model's predictions to semantically meaningful values.
	Args:
		raw:
		- instance_num==0: [num_rays, num_samples along ray, 4]. Prediction from model. (R,G,B,a)
		- instance_num >0: [num_rays, num_samples along ray, 10]. (R,G,B,a,instance_num)
		z_vals: [num_rays, num_samples along ray]. Integration time.
		rays_d: [num_rays, 3]. Direction of each ray.
		instance_num: Number of instances in the scene.
		decompose: if True returns per-instance RGBs.
	Returns:
		rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
		disp_map: [num_rays]. Disparity map. Inverse of depth map.
		acc_map: [num_rays]. Sum of weights along each ray.
		weights: [num_rays, num_samples]. Weights assigned to each sampled color.
		depth_map: [num_rays]. Estimated distance to object.
	"""
	raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

	dists = z_vals[..., 1:] - z_vals[..., :-1]
	dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]
	dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

	noise = 0.
	if raw_noise_std > 0.:
		noise = torch.randn(raw[..., 3].shape) * raw_noise_std

		# Overwrite randomly sampled data if pytest
		if pytest:
			np.random.seed(0)
			noise = np.random.rand(*list(raw[..., 3].shape)) * raw_noise_std
			noise = torch.Tensor(noise)

	alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
	# weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
	weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]

	rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]

	if instance_num > 0:
		instance_score = torch.sigmoid(raw[..., 4:])  # (N_rays, N_samples, instance_num)
		if decompose:
			decomposed_rgb_map = []
			instance_mask = torch.argmax(instance_score, dim=-1)
			for i in range(instance_num):
				mask_i = instance_mask == i

				alpha_i = torch.zeros_like(alpha)
				alpha_i[mask_i] = alpha[mask_i]
				weights_i = alpha_i * torch.cumprod(
					torch.cat([torch.ones((alpha_i.shape[0], 1)), 1. - alpha_i + 1e-10], -1), -1
				)[:, :-1]

				decomposed_rgb = torch.zeros([*instance_score.shape[:-1], rgb.shape[-1]])
				decomposed_rgb[mask_i] = rgb[mask_i]
				decomposed_rgb_map_i = torch.sum(weights_i[..., None] * decomposed_rgb, -2)
				decomposed_rgb_map.append(decomposed_rgb_map_i)
			decomposed_rgb_map = torch.stack(decomposed_rgb_map, dim=0)
			instance_map = torch.sum(weights[..., None] * instance_score, -2)  # (N_rays, instance_num)
			rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
		else:
			instance_map = torch.sum(weights[..., None] * instance_score, -2)  # (N_rays, instance_num)
			rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
			decomposed_rgb_map = None
	else:
		rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]
		instance_map = None
		decomposed_rgb_map = None

	depth_map = torch.sum(weights * z_vals, -1)
	disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
	acc_map = torch.sum(weights, -1)

	if white_bkgd:
		rgb_map = rgb_map + (1. - acc_map[..., None])
	return rgb_map, disp_map, acc_map, weights, depth_map, instance_map, decomposed_rgb_map



def render_rays(
		ray_batch, network_fn, network_query_fn, N_samples, retraw=False, lindisp=False, perturb=0., N_importance=0,
		network_fine=None, white_bkgd=False, raw_noise_std=0., verbose=False, pytest=False, decompose=False
):
	"""
	Volumetric rendering.
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
		z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)

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

	# raw = run_network(pts)
	raw = network_query_fn(pts, viewdirs, network_fn)
	rgb_map, disp_map, acc_map, weights, depth_map, instance_map, decomposed_rgb_map = raw2outputs(
		raw, z_vals, rays_d, network_fn.instance_label_dimension, raw_noise_std, white_bkgd, pytest=pytest, decompose=decompose
	)

	if N_importance > 0:

		rgb_map_0, disp_map_0, acc_map_0, instance_map0 = rgb_map, disp_map, acc_map, instance_map

		z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
		z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], N_importance, det=(perturb == 0.), pytest=pytest)
		z_samples = z_samples.detach()

		z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
		pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
		# [N_rays, N_samples + N_importance, 3]

		run_fn = network_fn if network_fine is None else network_fine
		# raw = run_network(pts, fn=run_fn)
		raw = network_query_fn(pts, viewdirs, run_fn)
		rgb_map, disp_map, acc_map, weights, depth_map, instance_map, decomposed_rgb_map0 = raw2outputs(
			raw, z_vals, rays_d, run_fn.instance_label_dimension, raw_noise_std, white_bkgd, pytest=pytest, decompose=decompose
		)
	if decompose:
		ret = {
			'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'instance_map': instance_map,
			'decomposed_rgb_map': decomposed_rgb_map
		}
	else:
		ret = {'rgb_map': rgb_map, 'disp_map': disp_map, 'acc_map': acc_map, 'instance_map': instance_map}
	if retraw:
		ret['raw'] = raw
	if N_importance > 0:
		ret['rgb0'] = rgb_map_0
		ret['disp0'] = disp_map_0
		ret['acc0'] = acc_map_0
		ret['instance0'] = instance_map0
		if decompose:
			ret['decomposed_rgb_map0'] = decomposed_rgb_map0
		ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)  # [N_rays]

	for k in ret:
		if (torch.isnan(ret[k]).any() or torch.isinf(ret[k]).any()) and DEBUG:
			print(f"! [Numerical Error] {k} contains nan or inf.")

	return ret



def batchify_rays(rays_flat, chunk=1024 * 32, decompose=False, **kwargs):
	"""
	Render rays in smaller minibatches to avoid OOM.
	"""
	all_ret = {}
	for i in range(0, rays_flat.shape[0], chunk):
		ret = render_rays(rays_flat[i:i + chunk], decompose=decompose, **kwargs)
		for k in ret:
			if k not in all_ret:
				all_ret[k] = []
			all_ret[k].append(ret[k])
	# all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}
	for k in all_ret:
		if k == 'decomposed_rgb_map' or k == 'decomposed_rgb_map0':
			all_ret[k] = torch.cat(all_ret[k], dim=1)
		else:
			all_ret[k] = torch.cat(all_ret[k], dim=0)
	return all_ret



def render(
		H, W, K, chunk=1024 * 32, rays=None, c2w=None, ndc=True, near=0., far=1.,
		use_viewdirs=False, c2w_staticcam=None, decompose=False, **kwargs
):
	"""
	Render rays
	Args:
		H: int. Height of image in pixels.
		W: int. Width of image in pixels.
		focal: float. Focal length of pinhole camera.
		chunk: int. Maximum number of rays to process simultaneously.
			Used to control maximum memory usage. Does not affect final results.
		rays: array of shape [2, batch_size, 3]. Ray origin and direction for each example in batch.
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

	if use_viewdirs:
		# provide ray directions as input
		viewdirs = rays_d
		if c2w_staticcam is not None:
			# special case to visualize effect of viewdirs
			rays_o, rays_d = get_rays(H, W, K, c2w_staticcam)
		viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)  # (N_rand, 3)
		viewdirs = torch.reshape(viewdirs, [-1, 3]).float()  # (N_rand, 3)

	sh = rays_d.shape  # [..., 3]
	if ndc:
		# for forward facing scenes
		rays_o, rays_d = ndc_rays(H, W, K[0][0], 1., rays_o, rays_d)

	# Create ray batch
	rays_o = torch.reshape(rays_o, [-1, 3]).float()
	rays_d = torch.reshape(rays_d, [-1, 3]).float()

	near, far = near * torch.ones_like(rays_d[..., :1]), far * torch.ones_like(rays_d[..., :1])
	rays = torch.cat([rays_o, rays_d, near, far], -1)
	if use_viewdirs:
		rays = torch.cat([rays, viewdirs], -1)

	# Render and reshape
	all_ret = batchify_rays(rays, chunk, decompose=decompose, **kwargs)
	for k in all_ret:
		if k == 'decomposed_rgb_map' or k == 'decomposed_rgb_map0':
			k_sh = list(all_ret[k].shape[:1]) + list(sh[:-1]) + list(all_ret[k].shape[2:])
			all_ret[k] = torch.reshape(all_ret[k], k_sh)
		else:
			k_sh = list(sh[:-1]) + list(all_ret[k].shape[1:])
			all_ret[k] = torch.reshape(all_ret[k], k_sh)
	if decompose:
		k_extract = ['rgb_map', 'disp_map', 'acc_map', 'instance_map', 'decomposed_rgb_map']
	else:
		k_extract = ['rgb_map', 'disp_map', 'acc_map', 'instance_map']
	ret_list = [all_ret[k] for k in k_extract]
	ret_dict = {k: all_ret[k] for k in all_ret if k not in k_extract}
	return ret_list + [ret_dict]



def render_path(
		render_poses, hwf, K, chunk, render_kwargs, gt_imgs=None, gt_masks=None,
		color_list=None, savedir=None, render_factor=0, decompose=False
):

	H, W, focal = hwf

	if render_factor != 0:
		# Render downsampled for speed
		H = H // render_factor
		W = W // render_factor
		focal = focal / render_factor

	rgbs = []
	disps = []
	instances = []
	instance_colors = []
	decomposed_rgbs = []

	t = time.time()
	for i, c2w in enumerate(tqdm(render_poses)):
		print(i, time.time() - t)
		t = time.time()
		if decompose:
			rgb, disp, acc, instance, decomposed_rgb, _ = render(
				H, W, K, chunk=chunk, c2w=c2w[:3, :4], decompose=decompose, **render_kwargs
			)
		else:
			rgb, disp, acc, instance, _ = render(H, W, K, chunk=chunk, c2w=c2w[:3, :4], **render_kwargs)
			decomposed_rgb = None
		rgbs.append(rgb.cpu().numpy())
		disps.append(disp.cpu().numpy())
		instances.append(instance.cpu().numpy())
		if decompose:
			decomposed_rgbs.append(decomposed_rgb.cpu().numpy())
		if i == 0:
			print(rgb.shape, disp.shape)

		"""
		if gt_imgs is not None and render_factor==0:
			p = -10. * np.log10(np.mean(np.square(rgb.cpu().numpy() - gt_imgs[i])))
			print(p)
		"""

		if savedir is not None:
			rgb8 = to8b(rgbs[-1])
			
			instance_infer = torch.zeros_like(instance)
			argmax = torch.argmax(instance, dim=-1)
			for j in range(instance.shape[-1]):
				onehot_j = torch.zeros(instance.shape[-1])
				onehot_j[j] = 1.
				mask_j = argmax == j
				instance_infer[mask_j] = onehot_j
			instance_color = label2color(instance_infer, color_list).cpu().numpy().astype(np.uint8)
			instance_colors.append(instance_color)
			if decompose:
				for k, object in enumerate(decomposed_rgb):
					filename_object = os.path.join(savedir, 'decomposed_rgb_{:03d}_{}.png'.format(i, k))
					imageio.imwrite(filename_object, to8b(object.cpu().numpy()))
			
			filename_rgb = os.path.join(savedir, '{:03d}.png'.format(i))
			filename_instance = os.path.join(savedir, 'mask_{:03d}.png'.format(i))
			imageio.imwrite(filename_rgb, rgb8)
			imageio.imwrite(filename_instance, instance_color)

	rgbs = np.stack(rgbs, 0)
	disps = np.stack(disps, 0)
	instances = np.stack(instances, 0)
	instance_colors = np.stack(instance_colors, 0)
	if decompose:
		decomposed_rgbs = np.stack(decomposed_rgbs, 0)

	return rgbs, disps, instances, instance_colors, decomposed_rgbs
