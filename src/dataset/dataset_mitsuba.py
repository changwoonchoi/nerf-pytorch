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
import math


class MitsubaDataset(NerfDataset):
	def __init__(self, basedir, **kwargs):
		super().__init__("mitsuba", **kwargs)
		self.scene_name = basedir.split("/")[-1]

		with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
			self.meta = json.load(fp)

		self.instance_color_list = np.loadtxt(os.path.join(basedir, 'instance_label.txt'))
		self.instance_num = len(self.instance_color_list)
		self.basedir = basedir

		self.skip = kwargs.get("skip", 1)
		if self.split == "train":
			self.skip = 1

		self.camera_angle_x = float(self.meta['frames'][0]['fov_degree']) / 180.0 * math.pi

		image0_path = os.path.join(self.basedir, "train/1.png")
		image0 = imageio.imread(image0_path, pilmode='RGB')
		self.original_height, self.original_width, _ = image0.shape

		self.height = int(self.original_height * self.scale)
		self.width = int(self.original_width * self.scale)
		self.focal = .5 * self.width / np.tan(0.5 * self.camera_angle_x)
		self.load_near_far_plane()

	def load_near_far_plane(self):
		"""
		Load near and far plane
		:return:
		"""
		self.near = 1
		self.far = 20

	def __len__(self):
		return len(self.meta['frames'][::self.skip])

	def __getitem__(self, index):
		"""
		Load single data corresponding to specific index
		:param index: data index
		"""
		frame = self.meta['frames'][::self.skip][index]
		image_file_path = os.path.join(self.basedir, self.split, "%d.png" % (self.skip * index + 1))
		mask_file_path = os.path.join(self.basedir, self.split, "%d_mask.png" % (self.skip * index + 1))

		# (1) load RGB Image
		image = cv2.imread(image_file_path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		if self.scale != 1:
			image = cv2.resize(image, None, fx=self.scale, fy=self.scale)

		# (2) load instance_label_mask
		instance_label_mask = None
		if self.load_instance_label_mask:
			mask = cv2.imread(mask_file_path)
			mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

			if self.scale != 1:
				mask = cv2.resize(mask, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

			mask = mask[:, :, 0] + 256 * mask[:, :, 1] + 256 * 256 * mask[:, :, 2]
			instance_label_mask = mask.astype(np.int32)

		# (3) load pose information
		pose = np.array(frame['transform']).astype(np.float32)
		# Mitsuba --> camera forward is +Z !!
		pose[:3, 0] *= -1
		pose[:3, 2] *= -1
		image = image.astype(np.float32)
		image /= 255.0

		sample = {}
		sample["image"] = image
		if self.load_instance_label_mask:
			sample["mask"] = instance_label_mask
		sample["pose"] = pose
		return sample

	def get_test_render_poses(self):
		# TODO : implement
		return None
