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


class ReplicaDataset(NerfDataset):
	def __init__(self, basedir, **kwargs):
		super().__init__("replica", **kwargs)
		self.scene_name = basedir.split("/")[-1]
		self.near = kwargs['near_plane']
		self.far = kwargs['far_plane']

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

		#self.camera_angle_x = float(self.meta['camera_angle_x']) / 180.0 * math.pi
		self.camera_angle_x = 90.0 / 180.0 * math.pi

		image0_path = os.path.join(self.basedir, "train/frame000000.jpg")
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
		image_file_path = os.path.join(self.basedir, self.split, 'frame{:06d}.jpg'.format(self.skip * index))
		depth_file_path = os.path.join(self.basedir, self.split, 'depth{:06d}.png'.format(self.skip * index))
		normal_file_path = os.path.join(self.basedir, self.split, 'normal{:06d}.png'.format(self.skip * index))
		prior_albedo_file_path = os.path.join(self.basedir, self.split, 'frame{:06d}_bell_r.png'.format(self.skip * index))
		prior_irradiance_file_path = os.path.join(self.basedir, self.split, 'frame{:06d}_bell_s.png'.format(self.skip * index))

		# (1) load RGB Image
		if self.load_image:
			sample["image"] = load_image_from_path(image_file_path, scale=self.scale)
		if self.load_normal:
			sample["normal"] = load_image_from_path(normal_file_path, scale=self.scale)
		if self.load_depth:
			depth = cv2.imread(depth_file_path, -1)
			depth = cv2.resize(depth, None, fx=self.scale, fy=self.scale)
			depth_scale = 65535.0 * 0.1
			depth = depth.astype(np.float32)
			depth = depth / depth_scale
			sample["depth"] = depth[..., None]
		if self.load_priors:
			sample["prior_albedo"] = load_image_from_path(prior_albedo_file_path, scale=self.scale)
			sample["prior_irradiance"] = load_image_from_path(prior_irradiance_file_path, scale=self.scale)

		# (3) load pose information
		transform = []
		for t1 in frame['transform'].split('\n'):
			transform_i = []
			for t2 in t1.split():
				transform_i.append(float(t2))
			transform.append(transform_i)
		pose = np.array(transform).astype(np.float32)
		pose = np.linalg.inv(pose)
		# Replica --> RDF --> down to up, forward to backward (RUB)
		pose[:3, 1] *= -1
		pose[:3, 2] *= -1
		sample["pose"] = pose
		#print(pose, "POSE!!!")
		return sample

	def get_test_render_poses(self):
		# TODO : implement
		return None
