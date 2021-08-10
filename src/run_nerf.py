import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter

import matplotlib.pyplot as plt

from label_encoder import OneHotLabelEncoder
from data_loader.load_clevr import load_clevr_data, load_clevr_instance_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


from config_parser import config_parser
from nerf_models.nerf_renderer import render, render_path
from nerf_models.nerf_renderer_helper import get_rays, get_rays_np
from nerf_models.nerf import create_nerf


def train():
	parser = config_parser()
	args = parser.parse_args()
	args.device = device

	# Load data (only clevr)
	K = None
	if args.dataset_type == 'clevr':
		if args.instance_mask:
			images, instance_label_mask, instance_color_list, poses, render_poses, hwf, i_split = load_clevr_instance_data(
				args.datadir, args.half_res, args.testskip
			)
			label_encoder = OneHotLabelEncoder(instance_color_list)
			encoded_instance_label_mask = label_encoder.encode_np(instance_label_mask)

			args.label_encoder = label_encoder
		else:
			images, poses, render_poses, hwf, i_split = load_clevr_data(
				args.datadir, args.half_res, args.testskip
			)
		print('Loaded CLEVR', images.shape, render_poses.shape, hwf, args.datadir)
		i_train, i_val, i_test = i_split

		hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
		near = hemi_R - 4
		far = hemi_R + 4

	else:
		print('Unknown dataset type', args.dataset_type, 'exiting')
		return

	# # Cast intrinsics to right types
	# H, W, focal = hwf
	# H, W = int(H), int(W)
	# hwf = [H, W, focal]

	# if K is None:
	# 	K = np.array([
	# 		[focal, 0, 0.5 * W],
	# 		[0, focal, 0.5 * H],
	# 		[0, 0, 1]
	# 	])

	# if args.render_test:
	# 	render_poses = np.array(poses[i_test])

	# # Create log dir and copy the config file
	# basedir = args.basedir
	# expname = args.expname
	# os.makedirs(os.path.join(basedir, expname), exist_ok=True)
	# f = os.path.join(basedir, expname, 'args.txt')
	# with open(f, 'w') as file:
	# 	for arg in sorted(vars(args)):
	# 		attr = getattr(args, arg)
	# 		file.write('{} = {}\n'.format(arg, attr))
	# if args.config is not None:
	# 	f = os.path.join(basedir, expname, 'config.txt')
	# 	with open(f, 'w') as file:
	# 		file.write(open(args.config, 'r').read())
	
	# writer = SummaryWriter(log_dir=os.path.join(basedir, expname))

	# # Create nerf model
	# render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args)
	# global_step = start

	# bds_dict = {
	# 	'near': near,
	# 	'far': far,
	# }
	# render_kwargs_train.update(bds_dict)
	# render_kwargs_test.update(bds_dict)

	# # Move testing data to GPU
	# render_poses = torch.Tensor(render_poses).to(device)

	# # Short circuit if only rendering out from trained model
	# if args.render_only:
	# 	print('RENDER ONLY')
	# 	with torch.no_grad():
	# 		if args.render_test:
	# 			# render_test switches to test poses
	# 			images = images[i_test]
	# 		else:
	# 			# Default is smoother render_poses path
	# 			images = None

	# 		testsavedir = os.path.join(
	# 			basedir, expname, 'renderonly_{}_{:06d}'.format('test' if args.render_test else 'path', start)
	# 		)
	# 		os.makedirs(testsavedir, exist_ok=True)
	# 		print('test poses shape', render_poses.shape)

	# 		render_path(
	# 			render_poses, hwf, K, args.chunk, render_kwargs_test,
	# 			color_list=instance_color_list, savedir=testsavedir, render_factor=args.render_factor,
	# 			decompose=args.render_decompose
	# 		)
	# 		print('Done rendering', testsavedir)
	# 		# imageio.mimwrite(os.path.join(testsavedir, 'video.mp4'), to8b(rgbs), fps=30, quality=8)

	# 		return

	# # Prepare raybatch tensor if batching random rays
	# N_rand = args.N_rand
	# use_batching = not args.no_batching
	# if use_batching:
	# 	# For random ray batching
	# 	print('get rays')
	# 	rays = np.stack([get_rays_np(H, W, K, p) for p in poses[:, :3, :4]], 0)  # [N, ro+rd, H, W, 3]
	# 	print('done, concats')
	# 	rays_rgb = np.concatenate([rays, images[:, None]], 1)  # [N, ro+rd+rgb, H, W, 3]
	# 	rays_rgb = np.transpose(rays_rgb, [0, 2, 3, 1, 4])  # [N, H, W, ro+rd+rgb, 3]
	# 	rays_rgb = np.stack([rays_rgb[i] for i in i_train], 0)  # train images only
	# 	rays_rgb = np.reshape(rays_rgb, [-1, 3, 3])  # [(N-1)*H*W, ro+rd+rgb, 3]
	# 	rays_rgb = rays_rgb.astype(np.float32)
	# 	print('shuffle rays')
	# 	np.random.shuffle(rays_rgb)

	# 	print('done')
	# 	i_batch = 0

	# # Move training data to GPU
	# if use_batching:
	# 	images = torch.Tensor(images).to(device)
	# poses = torch.Tensor(poses).to(device)
	# if use_batching:
	# 	rays_rgb = torch.Tensor(rays_rgb).to(device)

	# N_iters = args.N_iter + 1
	# print('Begin')
	# print('TRAIN views are', i_train)
	# print('TEST views are', i_test)
	# print('VAL views are', i_val)

	# # Summary writers
	# # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

	# start = start + 1
	# for i in trange(start, N_iters):
	# 	time0 = time.time()

	# 	# Sample random ray batch
	# 	if use_batching:
	# 		# TODO: implement for batch
	# 		# Random over all images
	# 		batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
	# 		batch = torch.transpose(batch, 0, 1)
	# 		batch_rays, target_s = batch[:2], batch[2]

	# 		i_batch += N_rand
	# 		if i_batch >= rays_rgb.shape[0]:
	# 			print("Shuffle data after an epoch!")
	# 			rand_idx = torch.randperm(rays_rgb.shape[0])
	# 			rays_rgb = rays_rgb[rand_idx]
	# 			i_batch = 0
	# 	else:
	# 		# Random from one image
	# 		img_i = np.random.choice(i_train)
	# 		target = images[img_i]
	# 		target = torch.Tensor(target).to(device)
	# 		if args.instance_num > 0:
	# 			target_mask = masks[img_i]
	# 			target_mask = torch.Tensor(target_mask).to(device)
	# 			target_mask_onehot = masks_onehot[img_i]
	# 			target_mask_onehot = torch.Tensor(target_mask_onehot).to(device)
	# 		pose = poses[img_i, :3, :4]

	# 		if N_rand is not None:
	# 			rays_o, rays_d = get_rays(H, W, K, torch.Tensor(pose))  # (H, W, 3), (H, W, 3)

	# 			if i < args.precrop_iters:
	# 				dH = int(H // 2 * args.precrop_frac)
	# 				dW = int(W // 2 * args.precrop_frac)
	# 				coords = torch.stack(
	# 					torch.meshgrid(
	# 						torch.linspace(H // 2 - dH, H // 2 + dH - 1, 2 * dH),
	# 						torch.linspace(W // 2 - dW, W // 2 + dW - 1, 2 * dW)
	# 					), -1)
	# 				if i == start:
	# 					print(
	# 						f"[Config] Center cropping of size {2*dH} x {2*dW} is enabled until iter \
	# 						{args.precrop_iters}"
	# 					)
	# 			else:
	# 				coords = torch.stack(torch.meshgrid(torch.linspace(0, H - 1, H), torch.linspace(0, W - 1, W)), -1)
	# 				# (H, W, 2)

	# 			coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)
	# 			select_inds = np.random.choice(coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
	# 			select_coords = coords[select_inds].long()  # (N_rand, 2)
	# 			rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
	# 			rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
	# 			batch_rays = torch.stack([rays_o, rays_d], 0)
	# 			target_s = target[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
	# 			target_mask_s = target_mask[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, N_instance)
	# 			target_mask_onehot_s = target_mask_onehot[select_coords[:, 0], select_coords[:, 1]]  #(N_rand, N_instance)

	# 	#####  Core optimization loop  #####
	# 	rgb, disp, acc, instance, extras = render(
	# 		H, W, K, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True, **render_kwargs_train
	# 	)

	# 	optimizer.zero_grad()
	# 	img_loss = img2mse(rgb, target_s)

	# 	instance_loss = label_encoder.error(instance, target_mask, target_mask)

	# 	trans = extras['raw'][..., -1]
	# 	psnr = mse2psnr(img_loss)

	# 	if 'rgb0' in extras:
	# 		img_loss0 = img2mse(extras['rgb0'], target_s)
	# 		img_loss = img_loss + img_loss0
	# 		psnr0 = mse2psnr(img_loss0)

	# 	if 'instance0' in extras:
	# 		# instance_loss0 = CEloss(extras['instance0'], target_mask_s.long()) if args.instance_num > 0 else 0
	# 		instance_loss0 = label_encoder.error(instance, target_mask, target_mask) if args.instance_num > 0 else 0

	# 		instance_loss = instance_loss + instance_loss0

	# 	alpha = 0.01
	# 	loss = img_loss + alpha * instance_loss
	# 	if i % 100 == 0:
	# 		writer.add_scalar('Loss/rgb_MSE', img_loss, i)
	# 		writer.add_scalar('Loss/instance_CrossEntropy', instance_loss, i)
	# 		writer.add_scalar('Loss/total_loss', loss, i)

	# 	loss.backward()
	# 	optimizer.step()

	# 	# NOTE: IMPORTANT!
	# 	###   update learning rate   ###
	# 	decay_rate = 0.1
	# 	decay_steps = args.lrate_decay * 1000
	# 	new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
	# 	for param_group in optimizer.param_groups:
	# 		param_group['lr'] = new_lrate
	# 	################################

	# 	dt = time.time()-time0
	# 	# print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
	# 	#####           end            #####

	# 	# Rest is logging
	# 	if i % args.i_weights == 0:
	# 		path = os.path.join(basedir, expname, '{:06d}.tar'.format(i))
	# 		torch.save({
	# 			'global_step': global_step,
	# 			'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
	# 			'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
	# 			'optimizer_state_dict': optimizer.state_dict(),
	# 		}, path)
	# 		print('Saved checkpoints at', path)

	# 	if i % args.i_video == 0 and i > 0:
	# 		# Turn on testing mode
	# 		with torch.no_grad():
	# 			rgbs, _, _, _, _ = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
	# 		print('Done, saving', rgbs.shape)
	# 		moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
	# 		imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)

	# 		# if args.use_viewdirs:
	# 		#     render_kwargs_test['c2w_staticcam'] = render_poses[0][:3,:4]
	# 		#     with torch.no_grad():
	# 		#         rgbs_still, _ = render_path(render_poses, hwf, args.chunk, render_kwargs_test)
	# 		#     render_kwargs_test['c2w_staticcam'] = None
	# 		#     imageio.mimwrite(moviebase + 'rgb_still.mp4', to8b(rgbs_still), fps=30, quality=8)

	# 	if i % args.i_testset == 0 and i > 0:
	# 		testsavedir = os.path.join(basedir, expname, 'testset_{:06d}'.format(i))
	# 		os.makedirs(testsavedir, exist_ok=True)
	# 		print('test poses shape', poses[i_test].shape)
	# 		with torch.no_grad():
	# 			rgbs, _, instances, instance_colors, decomposed_rgbs = render_path(
	# 				torch.Tensor(poses[i_test]).to(device), hwf, K, args.chunk, render_kwargs_test,
	# 				gt_imgs=images[i_test], gt_masks=masks[i_test], color_list=instance_color_list,
	# 				savedir=testsavedir, decompose=args.render_decompose
	# 			)

	# 			gt_img_batch = np.zeros((len(i_test), 3, images[i_test[0]].shape[0], images[i_test[0]].shape[1]))
	# 			for test_idx in range(len(i_test)):
	# 				gt_img_batch[test_idx] = images[i_test[test_idx]].transpose((2, 0, 1))

	# 			writer.add_images('test/gt_rgb', gt_img_batch, i)
	# 			writer.add_images('test/inferred_rgb', rgbs.transpose((0, 3, 1, 2)), i)
	# 			writer.add_images('test/inferred_mask', instance_colors.transpose((0, 3, 1, 2)), i)
	# 			if args.render_decompose:
	# 				for mask_idx in range(decomposed_rgbs.shape[1]):
	# 					writer.add_images(
	# 						'test/decomposed_rgb_{}'.format(mask_idx),
	# 						decomposed_rgbs[:, mask_idx, ...].transpose((0, 3, 1, 2)), i
	# 					)
	# 		print('Saved test set')

	# 	if i % args.i_print == 0:
	# 		tqdm.write(
	# 			f"[TRAIN] Iter: {i} Loss: {loss.item()} MSE: {img_loss.item()}\
	# 			instance_CE: {instance_loss.item()} PSNR: {psnr.item()}"
	# 		)

	# 	global_step += 1
	# writer.flush()


if __name__ == '__main__':
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	train()
