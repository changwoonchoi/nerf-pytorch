import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from utils import color2label, label2color


trans_t = lambda t: torch.Tensor([
	[1, 0, 0, 0],
	[0, 1, 0, 0],
	[0, 0, 1, t],
	[0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
	[1, 0, 0, 0],
	[0, np.cos(phi), -np.sin(phi), 0],
	[0, np.sin(phi), np.cos(phi), 0],
	[0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
	[np.cos(th), 0, -np.sin(th), 0],
	[0, 1, 0, 0],
	[np.sin(th), 0, np.cos(th), 0],
	[0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
	c2w = trans_t(radius)
	c2w = rot_phi(phi / 180. * np.pi) @ c2w
	c2w = rot_theta(theta / 180. * np.pi) @ c2w
	c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
	return c2w


def load_clevr_data(basedir, half_res=False, testskip=1):
	splits = ['train', 'val', 'test']
	metas = {}
	for s in splits:
		with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
			metas[s] = json.load(fp)
	
	all_imgs = []
	all_poses = []
	counts = [0]
	for s in splits:
		meta = metas[s]
		imgs = []
		poses = []
		if s == 'train' or testskip == 0:
			skip = 1
		else:
			skip = testskip
			
		for frame in meta['frames'][::skip]:
			fname = os.path.join(basedir, frame['file_path'])
			imgs.append(imageio.imread(fname)[..., :3])
			poses.append(np.array(frame['transform_matrix']))
		imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
		poses = np.array(poses).astype(np.float32)
		counts.append(counts[-1] + imgs.shape[0])
		all_imgs.append(imgs)
		all_poses.append(poses)
	
	i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
	
	imgs = np.concatenate(all_imgs, 0)
	poses = np.concatenate(all_poses, 0)
	
	H, W = imgs[0].shape[:2]
	camera_angle_x = float(meta['camera_angle_x'])
	focal = .5 * W / np.tan(.5 * camera_angle_x)
	
	render_poses = torch.stack([pose_spherical(angle, -30.0, 11.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
	
	if half_res:
		H = H // 2
		W = W // 2
		focal = focal / 2.

		imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
		for i, img in enumerate(imgs):
			imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
		imgs = imgs_half_res

	return imgs, poses, render_poses, [H, W, focal], i_split


def load_clevr_instance_data(basedir, half_res=False, testskip=1):
	splits = ['train', 'val', 'test']
	metas = {}
	for s in splits:
		with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
			metas[s] = json.load(fp)
	
	instance_color_list = torch.from_numpy(np.loadtxt(os.path.join(basedir, 'train/instance_label_render.txt'))).to(torch.uint8)
	instance_num = instance_color_list.shape[0]  # include background

	all_imgs = []
	all_masks_onehot = []
	all_masks = []
	all_poses = []
	counts = [0]
	for s in splits:
		meta = metas[s]
		imgs = []
		masks_onehot = []
		masks = []
		poses = []
		if s == 'train' or testskip == 0:
			skip = 1
		else:
			skip = testskip
			
		for frame in meta['frames'][::skip]:
			fname = os.path.join(basedir, frame['file_path'])
			imgs.append(imageio.imread(fname)[..., :3])
			colored_mask = imageio.imread(os.path.join(os.path.split(fname)[0], 'mask_' + os.path.split(fname)[1]))
			colored_mask = torch.from_numpy(colored_mask[..., :3])
			mask_onehot, mask = color2label(colored_mask, instance_color_list).cpu().numpy()
			masks_onehot.append(mask_onehot)
			masks.append(mask)
			poses.append(np.array(frame['transform_matrix']))
		imgs = (np.array(imgs) / 255.).astype(np.float32)  # keep all 4 channels (RGBA)
		poses = np.array(poses).astype(np.float32)
		counts.append(counts[-1] + imgs.shape[0])
		all_imgs.append(imgs)
		all_masks_onehot.append(masks_onehot)
		all_masks.append(masks)
		all_poses.append(poses)
	
	i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
	
	imgs = np.concatenate(all_imgs, 0)
	masks_onehot = np.concatenate(all_masks_onehot, 0)
	masks = np.concatenate(all_masks, 0)
	poses = np.concatenate(all_poses, 0)
	
	H, W = imgs[0].shape[:2]
	camera_angle_x = float(meta['camera_angle_x'])
	focal = .5 * W / np.tan(.5 * camera_angle_x)
	
	render_poses = torch.stack([pose_spherical(angle, -30.0, 11.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
	
	if half_res:
		H = H // 2
		W = W // 2
		focal = focal / 2.

		imgs_half_res = np.zeros((imgs.shape[0], H, W, 3))
		for i, img in enumerate(imgs):
			imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
		imgs = imgs_half_res

		masks_onehot_half_res = np.zeros((masks_onehot.shape[0], H, W, -1))
		masks_half_res = np.zeros((masks.shape[0], H, W, -1))
		for i, mask in enumerate(masks):
			masks_half_res[i] = cv2.resize(masks, (W, H), interpolation=cv2.INTER_AREA)
		for i, mask_oneho in enumerate(masks_onehot):
			masks_onehot_half_res[i] = cv2.resize(masks_onehot, (W, H), interpolation=cv2.INTER_AREA)

		masks_onehot = masks_onehot_half_res
		masks = masks_half_res

	return imgs, masks_onehot, masks, instance_num, poses, render_poses, [H, W, focal], i_split
