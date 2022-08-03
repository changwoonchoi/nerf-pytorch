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
from utils.colmap_utils import *

class ColmapDataset(NerfDataset):
	def __init__(self, basedir, **kwargs):
		super().__init__("colmap", **kwargs)
		self.scene_name = basedir.split("/")[-1]

		self.instance_color_list = []
		self.instance_num = 0

		self.basedir = basedir

		self.skip = kwargs.get("skip", 1)
		if self.split == "train":
			self.skip = 1

		self.height, self.width, self.K = self.read_intrinsics()
		self.imdata = read_images_binary(os.path.join(self.basedir, 'sparse/0/images.bin'))
		self.image_names = [self.imdata[k].name for k in self.imdata]
		if self.split == "train":
			index_list_tmp = [i * 8 + j + 1 for i in range(len(self.image_names) // 8 + 1) for j in range(7)]
			self.index_list = [i for i in index_list_tmp if i < len(self.image_names)]
		elif self.split in ["val", "test"]:
			index_list_tmp = [i * 8 for i in range(len(self.image_names) // 8 + 1)]
			self.index_list = [i for i in index_list_tmp if i < len(self.image_names)]
		self.center_pose = kwargs.get("center_pose", False)
		if self.center_pose:
			w2c_mats = []
			for k in self.imdata:
				im_tmp = self.imdata[k]
				R = im_tmp.qvec2rotmat(); t = im_tmp.tvec.reshape(3, 1)
				w2c_mats += [np.concatenate([np.concatenate([R, t], 1), np.array([[0, 0, 0, 1.]])], 0)]
			w2c_mats += np.stack(w2c_mats, 0)
			self.poses_for_center = np.linalg.inv(w2c_mats)[:, :3]
			pts3d = read_points3d_binary(os.path.join(self.basedir, 'sparse/0/points3D.bin'))
			self.pts3d = np.array([pts3d[k].xyz for k in pts3d])

	def __len__(self):
		return len(self.index_list)

	def __getitem__(self, index):
		sample = {}

		"""
		Load single data corresponding to specific index
		:param index: data index
		"""
		image_file_path = os.path.join(self.basedir, 'images', self.image_names[self.index_list[index]])

		# (1) load RGB Image
		if self.load_image:
			sample["image"] = load_image_from_path(image_file_path, scale=self.scale, is_colmap=True)
		if self.load_normal:
			raise ValueError
		if self.load_albedo:
			raise ValueError
		if self.load_roughness:
			raise ValueError
		if self.load_depth:
			raise ValueError
		if self.load_irradiance:
			raise ValueError
		if self.load_diffuse_specular:
			raise ValueError
		if self.load_priors:
			raise ValueError

		# (2) load instance_label_mask
		if self.load_instance_label_mask:
			raise ValueError

		# (3) load pose information
		im = self.imdata[self.index_list[index] + 1]
		bottom = np.array([[0, 0, 0, 1.]])
		R = im.qvec2rotmat()
		t = im.tvec.reshape(3, 1)
		w2c = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
		pose = np.linalg.inv(w2c)  # (4, 4)

		if self.center_pose:
			poses, _ = self.center_poses(self.poses_for_center, self.pts3d)
			pose = poses[self.index_list[index]]

		# Mitsuba --> camera forward is +Z !!
		# pose[:3, 0] *= -1
		# pose[:3, 2] *= -1
		sample["pose"] = pose
		return sample

	def read_intrinsics(self):
		camdata = read_cameras_binary(os.path.join(self.basedir, 'sparse/0/cameras.bin'))
		h = int(camdata[1].height * self.scale)
		w = int(camdata[1].width * self.scale)

		if camdata[1].model == 'SIMPLE_RADIAL':
			fx = fy = camdata[1].params[0] * self.scale
			cx = camdata[1].params[1] * self.scale
			cy = camdata[1].params[2] * self.scale
		elif camdata[1].model in ['PINHOLE', 'OPENCV']:
			fx = camdata[1].params[0] * self.scale
			fy = camdata[1].params[1] * self.scale
			cx = camdata[1].params[2] * self.scale
			cy = camdata[1].params[3] * self.scale
		else:
			raise ValueError(f"Please parse the intrinsics for camera model {camdata[1].model}")
		K = np.array([
			[fx, 0, cx],
			[0, fy, cy],
			[0, 0, 1]]
		)
		return h, w, K

	def center_poses(self, poses, pts3d):
		pose_avg = self.average_poses(poses, pts3d)  # (3,4)
		pose_avg_homo = np.eye(4)
		pose_avg_homo[:3] = pose_avg
		pose_avg_inv = np.linalg.inv(pose_avg_homo)
		last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
		poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate
		poses_centered = pose_avg_inv @ poses_homo
		pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T
		return poses_centered, pts3d_centered

	def average_poses(self, poses, pts3d):
		center = pts3d.mean(0)
		z = self.normalize(poses[..., 2].mean(0))
		y_ = poses[..., 1].mean(0)
		x = self.normalize(np.cross(y_, z))
		y = np.cross(z, x)
		pose_avg = np.stack([x, y, z, center], 1)
		return pose_avg

	def normalize(self, v):
		return v / np.linalg.norm(v)
