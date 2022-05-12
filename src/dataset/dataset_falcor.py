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

from pathlib import Path
import re
import numpy as np
import natsort
import pprint


def find_all_images(basedir):
	image_name_list = []
	for path in Path(basedir).rglob("*.png"):
		if "_r" in str(path):
			pass
		elif "_s" in str(path):
			pass
		else:
			image_name_list.append(str(path))
	image_name_list = natsort.natsorted(image_name_list)
	#print(len(image_name_list))
	#pprint.pprint(image_name_list)
	return image_name_list


def load_transforms(path):
	parser = lambda s: np.array(s.replace("float3(", "").replace(")", "").split(",")).astype(np.float32)
	transforms = []
	with open(path) as f:
		while True:
			line = f.readline()
			if len(line) == 0:
				break
			p = re.compile('float3\(.*?\)')
			result = p.findall(line)
			position = parser(result[0])
			target = parser(result[1])
			up = parser(result[2])
			transform = get_lookat_matrix(position, target, up)
			transforms.append(transform)
			if not line:
				break
	return transforms


def normalize(mat):
	return mat / np.linalg.norm(mat)


def get_lookat_matrix(origin, target, up):
	forward = normalize(target - origin)
	right = normalize(np.cross(forward, up))
	up = normalize(np.cross(right, forward))

	matrix = np.array([
		[right[0], right[1], right[2], 0],
		[up[0], up[1], up[2], 0],
		[-forward[0], -forward[1], -forward[2], 0],
		[origin[0], origin[1], origin[2], 1]
	]).transpose()

	return matrix


def load_transforms_simple(path):
	config = json.load(open(path))
	frames = config["frames"]
	transforms = []
	for frame in frames:
		transforms.append(np.asarray(frame["transform"]))
		#print("TRANSFORM", frame["transform"])
	return transforms


class FalcorDataset(NerfDataset):
	def __init__(self, basedir, **kwargs):
		super().__init__("falcor", **kwargs)
		self.near = kwargs['near_plane']
		self.far = kwargs['far_plane']

		self.scene_name = basedir.split("/")[-1]
		self.transforms = load_transforms_simple(os.path.join(basedir, 'transforms_%s.json'% kwargs.get("split", "train")))
		#self.transforms = load_transforms(os.path.join(basedir, 'viewpoints.txt'))
		self.image_lists = find_all_images(basedir)
		#print(len(self.transforms))
		#print(len(self.image_lists))
		#assert len(self.transforms) == len(self.image_lists)

		if self.load_instance_label_mask:
			self.instance_color_list = np.loadtxt(os.path.join(basedir, 'instance_label.txt'))
			self.instance_num = len(self.instance_color_list)
		else:
			self.instance_color_list = []
			self.instance_num = 0

		self.basedir = basedir

		if self.load_priors:
			with open(os.path.join(basedir, 'avg_irradiance.json'), 'r') as fp:
				f = json.load(fp)
				self.prior_irradiance_mean = f["mean_" + self.prior_type]

		self.skip = kwargs.get("skip", 1)

		self.camera_angle_x = float(60) / 180.0 * math.pi

		image0 = imageio.imread(self.image_lists[0], pilmode='RGB')
		self.original_height, self.original_width, _ = image0.shape

		self.height = int(self.original_height * self.scale)
		self.width = int(self.original_width * self.scale)
		self.focal = .5 * self.width / np.tan(0.5 * self.camera_angle_x)

	def __len__(self):
		return len(self.transforms[::self.skip])

	def __getitem__(self, index):
		sample = {}
		"""
		Load single data corresponding to specific index
		:param index: data index
		"""

		transform = self.transforms[::self.skip][index]
		basedir = "/data1/juhyeonkim/projects/SegNerfDataGenerator/result_20220510/kitchen_copy/sphere"
		normal_file_path = os.path.join(basedir, "%d_normal.png" % (self.skip * index + 1))
		depth_file_path = os.path.join(basedir, "%d_depth.npy" % (self.skip * index + 1))

		# (1) load RGB Image
		if self.load_image:
			image_file_path = self.image_lists[::self.skip][index]
			sample["image"] = load_image_from_path(image_file_path, scale=self.scale)
		if self.load_depth:
			sample["depth"] = load_numpy_from_path(depth_file_path, scale=self.scale)[..., None]
		if self.load_normal:
			sample["normal"] = load_image_from_path(normal_file_path, scale=self.scale)
		if self.load_priors:
			prior_albedo_file_path = self.image_lists[::self.skip][index][:-4] + "_{}_r.png".format(self.prior_type)
			prior_irradiance_file_path = self.image_lists[::self.skip][index][:-4] + "_{}_s.png".format(self.prior_type)
			sample["prior_albedo"] = load_image_from_path(prior_albedo_file_path, scale=self.scale)
			sample["prior_irradiance"] = load_image_from_path(prior_irradiance_file_path, scale=self.scale)

		# (3) load pose information
		sample["pose"] = transform

		#print(transform, "TRANSFROM!")
		return sample

	def get_test_render_poses(self):
		# TODO : implement
		return None
