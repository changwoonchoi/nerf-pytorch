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
from utils.image_utils import *


class NeRFSyntheticDataset(NerfDataset):
	def __init__(self, basedir, **kwargs):
		super().__init__("nerf_synthetic", **kwargs)
		self.scene_name = basedir.split("/")[-1]
		self.near = 1
		self.far = 20

		with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
			self.meta = json.load(fp)

		if self.load_instance_label_mask:
			self.instance_color_list = np.loadtxt(os.path.join(basedir, 'instance_label.txt'))
			self.instance_num = len(self.instance_color_list)
		else:
			self.instance_color_list = []
			self.instance_num = 0

		self.basedir = basedir

		self.skip = kwargs.get("skip", 1)
		if self.split == "train":
			self.skip = 1

		self.camera_angle_x = float(self.meta['camera_angle_x'])

		image0_path = os.path.join(self.basedir, "train/r_0.png")
		image0 = imageio.imread(image0_path, pilmode='RGB')
		self.original_height, self.original_width, _ = image0.shape

		self.height = int(self.original_height * self.scale)
		self.width = int(self.original_width * self.scale)
		self.focal = .5 * self.width / np.tan(0.5 * self.camera_angle_x)

	def __len__(self):
		return len(self.meta['frames'][::self.skip])

	def __getitem__(self, index):
		sample = {}

		"""
		Load single data corresponding to specific index
		:param index: data index
		"""
		frame = self.meta['frames'][::self.skip][index]
		image_file_path = os.path.join(self.basedir, self.split, "r_%d.png" % (self.skip * index))

		# (1) load RGB Image
		if self.load_image:
			sample["image"] = load_image_from_path(image_file_path, scale=self.scale)

		# (3) load pose information
		pose = np.array(frame['transform_matrix']).astype(np.float32)
		# Mitsuba --> camera forward is +Z !!
		# pose[:3, 0] *= -1
		# pose[:3, 2] *= -1
		sample["pose"] = pose
		print(pose, "POSE!!!!!!!")

		return sample

	def get_test_render_poses(self):
		# TODO : implement
		return None
