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


class ColmapDataset(NerfDataset):
	def __init__(self, basedir, **kwargs):
		super().__init__("colmap", **kwargs)
		self.scene_name = basedir.split("/")[-1]
		if kwargs.get("load_depth_range_from_file", False):
			with open(os.path.join(basedir, 'min_max_depth.json'), 'r') as fp:
				f = json.load(fp)
				self.near = f["min_depth"] * 0.9
				self.far = f["max_depth"] * 1.1
			print("LOAD FROM FILE!!!!!!!!!!!!!!!!!!!!!!!")
			print(self.near)
			print(self.far)

		if self.load_priors:
			with open(os.path.join(basedir, 'avg_irradiance.json'), 'r') as fp:
				f = json.load(fp)
				self.prior_irradiance_mean = f["mean_" + self.prior_type]

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

		self.camera_angle_x = float(self.meta['frames'][0]['fov_degree']) / 180.0 * math.pi

		image0_path = os.path.join(self.basedir, "train/1.png")
		image0 = imageio.imread(image0_path, pilmode='RGB')
		self.original_height, self.original_width, _ = image0.shape

		self.height = int(self.original_height * self.scale)
		self.width = int(self.original_width * self.scale)
		self.focal = .5 * self.width / np.tan(0.5 * self.camera_angle_x)


	# 	self.load_near_far_plane()
	#
	# def load_near_far_plane(self):
	# 	"""
	# 	Load near and far plane
	# 	:return:
	# 	"""
	# 	self.near = 1
	# 	self.far = 20

	def __len__(self):
		return len(self.meta['frames'][::self.skip])

	def __getitem__(self, index):
		sample = {}

		"""
		Load single data corresponding to specific index
		:param index: data index
		"""
		frame = self.meta['frames'][::self.skip][index]
		image_file_path = os.path.join(self.basedir, self.split, "%d.png" % (self.skip * index + 1))
		mask_file_path = os.path.join(self.basedir, self.split, "%d_mask.png" % (self.skip * index + 1))
		normal_file_path = os.path.join(self.basedir, self.split, "%d_normal.png" % (self.skip * index + 1))
		albedo_file_path = os.path.join(self.basedir, self.split, "%d_albedo.png" % (self.skip * index + 1))
		roughness_file_path = os.path.join(self.basedir, self.split, "%d_roughness.png" % (self.skip * index + 1))
		depth_file_path = os.path.join(self.basedir, self.split, "%d_depth.npy" % (self.skip * index + 1))
		diffuse_file_path = os.path.join(self.basedir, self.split, "%d_diffuse.png" % (self.skip * index + 1))
		specular_file_path = os.path.join(self.basedir, self.split, "%d_specular.png" % (self.skip * index + 1))
		irradiance_file_path = os.path.join(self.basedir, self.split, "%d_irradiance.png" % (self.skip * index + 1))
		prior_albedo_file_path = os.path.join(self.basedir, self.split, "{}_{}_r.png".format(self.skip * index + 1, self.prior_type))
		prior_irradiance_file_path = os.path.join(self.basedir, self.split, "{}_{}_s.png".format(self.skip * index + 1, self.prior_type))

		# (1) load RGB Image
		if self.load_image:
			sample["image"] = load_image_from_path(image_file_path, scale=self.scale)
		if self.load_normal:
			sample["normal"] = load_image_from_path(normal_file_path, scale=self.scale)
		if self.load_albedo:
			albedo_linear = load_image_from_path(albedo_file_path, scale=self.scale)
			# albedo_srgb = np.power(albedo_linear, 1/2.2)
			sample["albedo"] = albedo_linear
		if self.load_roughness:
			sample["roughness"] = load_image_from_path(roughness_file_path, scale=self.scale)[..., 0:1]
		if self.load_depth:
			sample["depth"] = load_numpy_from_path(depth_file_path, scale=self.scale)[..., None]
		if self.load_irradiance:
			irradiance = load_image_from_path(irradiance_file_path, scale=self.scale)
			#irradiance = np.power(irradiance, 2.2)
			#irradiance = irradiance / np.maximum(1 - irradiance, 0.000001)
			sample["irradiance"] = irradiance

		if self.load_diffuse_specular:
			sample["diffuse"] = load_image_from_path(diffuse_file_path, scale=self.scale)
			sample["specular"] = load_image_from_path(specular_file_path, scale=self.scale)

		if self.load_priors:
			sample["prior_albedo"] = load_image_from_path(prior_albedo_file_path, scale=self.scale)
			sample["prior_irradiance"] = load_image_from_path(prior_irradiance_file_path, scale=self.scale)

		# (2) load instance_label_mask
		if self.load_instance_label_mask:
			mask = cv2.imread(mask_file_path)
			mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

			if self.scale != 1:
				mask = cv2.resize(mask, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_NEAREST)

			mask = mask[:, :, 0] + 256 * mask[:, :, 1] + 256 * 256 * mask[:, :, 2]
			instance_label_mask = mask.astype(np.int32)
			sample["mask"] = instance_label_mask

		# (3) load pose information
		pose = np.array(frame['transform']).astype(np.float32)
		# Mitsuba --> camera forward is +Z !!
		pose[:3, 0] *= -1
		pose[:3, 2] *= -1
		sample["pose"] = pose
		return sample

	def get_test_render_poses(self):
		# TODO : implement
		return None
