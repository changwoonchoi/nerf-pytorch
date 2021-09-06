from abc import ABC

from torch.utils.data import Dataset
import os
import numpy as np
import json
import imageio
import torch
from utils.label_utils import colored_mask_to_label_map_np
from utils.math_utils import pose_spherical

import matplotlib.pyplot as plt
from dataset.dataset_interface import NerfDataset
from torchvision import transforms
import cv2


class ClevrDataset(NerfDataset):
	def __init__(self, basedir, **kwargs):
		super().__init__("clevr", **kwargs)
		with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
			self.meta = json.load(fp)

		self.instance_color_list = np.loadtxt(os.path.join(basedir, 'train/instance_label_render.txt'))
		self.instance_num = len(self.instance_color_list)
		self.basedir = basedir

		self.skip = kwargs.get("skip", 1)
		if self.split == "train":
			self.skip = 1

		self.camera_angle_x = float(self.meta['camera_angle_x'])

		image0_path = os.path.join(self.basedir, self.split, os.path.split(self.meta['frames'][0]['file_path'])[1])
		image0 = imageio.imread(image0_path, pilmode='RGB')
		self.original_height, self.original_width, _ = image0.shape

		self.height = int(self.original_height * self.scale)
		self.width = int(self.original_width * self.scale)
		self.focal = .5 * self.width / np.tan(0.5 * self.camera_angle_x)
		self.load_near_far_plane(**kwargs)

	def load_near_far_plane(self, **kwargs):
		"""
		Load near and far plane
		:return:
		"""
		# need average from all data
		poses = []
		for split in ["train", "val", "test"]:
			with open(os.path.join(self.basedir, 'transforms_{}.json'.format(split)), 'r') as fp:
				meta = json.load(fp)
			for frame in meta['frames']:
				pose = np.array(frame['transform_matrix'])
				poses.append(pose)
		poses = np.asarray(poses)
		hemi_R = np.mean(np.linalg.norm(poses[:, :3, -1], axis=-1))
		sample_length = kwargs.get("sample_length", 8)
		near = hemi_R - sample_length / 2
		far = hemi_R + sample_length / 2
		self.near = near
		self.far = far

	def __len__(self):
		return len(self.meta['frames'][::self.skip])

	def __getitem__(self, index):
		"""
		Load single data corresponding to specific index
		:param index: data index
		"""
		frame = self.meta['frames'][::self.skip][index]
		image_file_path = os.path.join(self.basedir, self.split, os.path.split(frame['file_path'])[1])
		mask_file_path = os.path.join(os.path.split(image_file_path)[0], 'mask_' + os.path.split(image_file_path)[1])

		# (1) load RGB Image
		image = cv2.imread(image_file_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.scale != 1:
			image = cv2.resize(image, None, fx=self.scale, fy=self.scale)

		# (2) load colored mask and convert into labeled mask
		instance_label_mask = None
		if self.load_instance_label_mask:
			colored_mask = cv2.imread(mask_file_path)
			colored_mask = cv2.cvtColor(colored_mask, cv2.COLOR_BGR2RGB)
			if self.scale != 1:
				colored_mask = cv2.resize(colored_mask, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)
			instance_label_mask = colored_mask_to_label_map_np(colored_mask, self.instance_color_list)

		# (3) load pose information
		pose = np.array(frame['transform_matrix']).astype(np.float32)

		image = image.astype(np.float32)
		image /= 255.0

		sample = {}
		sample["image"] = image
		if self.load_instance_label_mask:
			sample["mask"] = instance_label_mask
		sample["pose"] = pose
		return sample

	def get_test_render_poses(self):
		return torch.stack([pose_spherical(angle, -30.0, 11.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
