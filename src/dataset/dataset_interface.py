import logging

import numpy as np
from torch.utils.data import Dataset, DataLoader
from abc import ABC
from utils.logging_utils import load_logger
import torch
from utils.color_utils import get_basecolor
from torchvision import transforms
from utils.image_utils import *

class NerfDataset(Dataset, ABC):
	def __init__(self, name, **kwargs):
		self.original_width = 0
		self.original_height = 0
		self.width = 0
		self.height = 0
		self.scale = kwargs.get("image_scale", 1)

		self.split = kwargs.get("split", "train")
		self.name = name

		self.focal = 0
		self.K = None

		self.near = 0
		self.far = 0

		self.images = []
		self.prefiltered_images = []
		self.poses = []
		self.masks = []
		self.normals = []
		self.albedos = []
		self.roughness = []
		self.instances = []
		self.depths = []
		self.diffuses = []
		self.speculars = []
		self.irradiances = []

		self.prior_albedos = []
		self.prior_irradiances = []
		self.prior_type = kwargs.get("prior_type", "bell")

		self.prior_irradiance_mean = 0.7

		self.load_image = kwargs.get("load_image", True)
		self.load_normal = kwargs.get("load_normal", False)
		self.load_albedo = kwargs.get("load_albedo", False)
		self.load_roughness = kwargs.get("load_roughness", False)
		self.load_depth = kwargs.get("load_depth", False)
		self.load_diffuse_specular = kwargs.get("load_diffuse_specular", False)
		self.load_irradiance = kwargs.get("load_irradiance", False)

		self.load_priors = kwargs.get("load_priors", False)

		self.instance_color_list = []
		self.instance_num = 0

		self.full_data_loaded = False
		self.use_caching = True
		self.logger = load_logger("NeRF DataSet")
		self.logger.setLevel(logging.DEBUG)

		self.load_instance_label_mask = kwargs.get("load_instance_label_mask", False)

		# base color clustering related
		self.cluster_image_resize = 0.5
		self.cluster_image_number = 10
		self.num_init_cluster = 10
		self.cluster_th = 0
		self.init_basecolor = None
		self.num_cluster = 0
		self.coarse_radiance_number = kwargs.get("coarse_radiance_number")

		self.coarse_resize_scale = 4

		self.near = kwargs.get("near_plane", 1)
		self.far = kwargs.get("far_plane", 10)

	def get_focal_matrix(self):
		if self.K is None:
			K = np.array([
				[self.focal, 0, 0.5 * self.width],
				[0, self.focal, 0.5 * self.height],
				[0, 0, 1]
			]).astype(np.float32)
		else:
			K = self.K
		return K

	def get_resized_normal_albedo(self, resize_factor, i):
		t = transforms.Resize(size=(self.height // resize_factor, self.width // resize_factor), antialias=True)
		result = {}
		if self.load_albedo:
			albedo_temp = self.albedos[i].permute((2, 0, 1))
			result["albedo"] = t(albedo_temp).permute((1, 2, 0))

		if self.load_normal:
			normal_temp = self.normals[i].permute((2, 0, 1))
			result["normal"] = t(normal_temp).permute((1, 2, 0))

		if self.load_irradiance:
			irradiance_temp = self.irradiances[i].permute((2, 0, 1))
			result["irradiance"] = t(irradiance_temp).permute((1, 2, 0))

		if self.load_roughness:
			roughness_temp = self.roughness[i].permute((2,0,1))
			result["roughness"] = t(roughness_temp).permute((1,2,0))

		if self.load_depth:
			depth_temp = self.depths[i].permute((2,0,1))
			result["depth"] = t(depth_temp).permute((1, 2, 0))

		if self.load_priors:
			prior_albedo_temp = self.prior_albedos[i].permute((2, 0, 1))
			prior_irradiance_temp = self.prior_irradiances[i].permute((2, 0, 1))
			result["prior_albedo"] = t(prior_albedo_temp).permute((2, 0, 1))
			result["prior_irradiance"] = t(prior_irradiance_temp).permute((2, 0, 1))

		return result

	def get_coarse_images(self, level):
		new_images = []
		# t_orig = transforms.Resize(size=(self.height, self.width), antialias=True)
		t_orig = transforms.Resize(size=(self.height, self.width), antialias=True)
		for i in range(len(self)):
			image_temp = self.images[i].permute((2, 0, 1))
			# image_temp = image_temp.permute((1,2,0))
			sh = int(self.height / self.scale)
			sw = int(self.width / self.scale)
			for _ in range(level):
				sh = sh//self.coarse_resize_scale
				sw = sw//self.coarse_resize_scale
			# t = transforms.Resize(size=(sh, sw), antialias=True)
			t = transforms.Resize(size=(sh, sw), antialias=True)
			image_temp = t_orig(t(image_temp)).permute((1, 2, 0))
			new_images.append(image_temp)
		return torch.stack(new_images, 0)

	def get_info(self, image_index, u, v):
		pixel_info = {}
		# t_orig = transforms.Resize(size=(self.height, self.width), antialias=True)
		#
		# image_temp = self.images[image_index].permute((2, 0, 1))
		#
		# sh = self.height * 0.5 / self.scale
		# sw = self.width * 0.5 / self.scale
		# for i in range(self.coarse_radiance_number):
		#
		# 	sh = sh//self.coarse_resize_scale
		# 	sw = sw//self.coarse_resize_scale
		# 	t = transforms.Resize(size=(sh, sw), antialias=True)
		# 	image_temp = t(image_temp)
		# 	new_image = t_orig(image_temp).permute((1, 2, 0))
		# 	pixel_info["rgb_%d" % (i+1)] = new_image[v, u, :]

		pixel_info["rgb"] = self.images[image_index][v, u, :]
		for i in range(self.coarse_radiance_number):
			pixel_info["rgb_%d" % (i + 1)] = self.prefiltered_images[i][image_index][v, u, :]

		if self.load_albedo:
			pixel_info["albedo"] = self.albedos[image_index][v, u, :]
		if self.load_normal:
			pixel_info["normal"] = self.normals[image_index][v, u, :]
		if self.load_instance_label_mask:
			pixel_info["label"] = self.masks[image_index][v, u]
		if self.load_roughness:
			pixel_info["roughness"] = self.roughness[image_index][v, u]
		if self.load_depth:
			pixel_info["depth"] = self.depths[image_index][v, u]
		if self.load_irradiance:
			pixel_info["irradiance"] = self.irradiances[image_index][v, u, :]
		if self.load_priors:
			pixel_info["prior_albedo"] = self.prior_albedos[image_index][v, u, :]
			pixel_info["prior_irradiance"] = self.prior_irradiances[image_index][v, u, 0]  # our irradiance map has one channel
		return pixel_info

	def get_near_far_plane(self):
		return {'near': self.near, 'far': self.far}

	def get_test_render_poses(self):
		pass

	def load_all_data(self, num_of_workers=4):
		"""
		Load all data using multiprocessing
		:param num_of_workers: number of multiprocess
		:return: None
		"""
		if self.full_data_loaded:
			return
		data_loader = DataLoader(self, num_workers=num_of_workers, batch_size=1)
		for i, data in enumerate(data_loader):
			if "image" in data:
				image = data["image"][0]
				self.images.append(image)
			if "pose" in data:
				self.poses.append(data["pose"][0])
			if self.load_instance_label_mask:
				self.masks.append(data["mask"][0])
			if self.load_normal:
				self.normals.append(data["normal"][0])
			if self.load_albedo:
				self.albedos.append(data["albedo"][0])
			if self.load_roughness:
				self.roughness.append(data["roughness"][0])
			if self.load_depth:
				self.depths.append(data["depth"][0])
			if self.load_diffuse_specular:
				self.diffuses.append(data["diffuse"][0])
				self.speculars.append(data["specular"][0])
			if self.load_irradiance:
				self.irradiances.append(data["irradiance"][0])
			if self.load_priors:
				self.prior_albedos.append(data["prior_albedo"][0])
				self.prior_irradiances.append(data["prior_irradiance"][0])
		self.full_data_loaded = True

	def to_tensor(self, device):
		if len(self.images) > 0:
			self.images = torch.stack(self.images, 0).to(device)

			for i in range(self.coarse_radiance_number):
				prefiltered_image = self.get_coarse_images(i + 1)
				self.prefiltered_images.append(prefiltered_image)

			for i in range(self.coarse_radiance_number):
				self.prefiltered_images[i] = self.prefiltered_images[i].to(device)

		if len(self.poses) > 0:
			self.poses = torch.stack(self.poses, 0).to(device)
		if self.init_basecolor is not None:
			self.init_basecolor = torch.from_numpy(self.init_basecolor).to(device)
		if self.load_instance_label_mask:
			self.masks = torch.stack(self.masks, 0).to(device)
		if self.load_normal:
			self.normals = torch.stack(self.normals, 0).to(device)
		if self.load_albedo:
			self.albedos = torch.stack(self.albedos, 0).to(device)
		if self.load_roughness:
			self.roughness = torch.stack(self.roughness, 0).to(device)
		if self.load_depth:
			self.depths = torch.stack(self.depths, 0).to(device)
		if self.load_diffuse_specular:
			self.diffuses = torch.stack(self.diffuses, 0).to(device)
			self.speculars = torch.stack(self.speculars, 0).to(device)
		if self.load_irradiance:
			self.irradiances = torch.stack(self.irradiances, 0).to(device)
		if self.load_priors:
			self.prior_albedos = torch.stack(self.prior_albedos, 0).to(device)
			self.prior_irradiances = torch.stack(self.prior_irradiances, 0).to(device)
	def __getitem__(self, item):
		pass

	def __str__(self):
		logs = ["[Dataset]"]
		logs += ["\t- type : %s" % self.name]
		logs += ["\t- split : %s" % self.split]
		logs += ["\t- scale : %s" % str(self.scale)]
		logs += ["\t- size (raw) : %d x %d" % (self.original_width, self.original_height)]
		logs += ["\t- size : %d x %d" % (self.width, self.height)]
		logs += ["\t- image number : %d" % len(self)]
		logs += ["\t- instance number : %d" % self.instance_num]
		logs += ["\t= cluster number : %d" % self.num_cluster]
		return "\n".join(logs)

	def __len__(self):
		pass

	def get_base_color(
			self,
			learn_from_gt_albedo_map=False,
			cluster_image_number=50,
			cluster_image_resize=0.5,
			cluster_init_number=20,
			cluster_merge_threshold=0.1,
			cluster_number_lower_bound=8,
			visualize=False
	):
		if learn_from_gt_albedo_map:
			target_images = self.albedos
			normalize = False
		else:
			target_images = self.images
			normalize = True

		if cluster_image_number == -1:
			random_indices = np.arange(len(target_images))
		else:
			random_indices = np.random.permutation(len(target_images))[0:cluster_image_number]
		new_width = int(self.width * cluster_image_resize)
		new_height = int(self.height * cluster_image_resize)
		p = transforms.Compose([transforms.Resize((new_height, new_width))])
		sampled_imgs = []
		for i in random_indices:
			sampled_imgs.append(target_images[i])
		sampled_imgs = torch.stack(sampled_imgs, 0)
		sampled_imgs = sampled_imgs.permute((0, 3, 1, 2))
		sampled_imgs = p(sampled_imgs)
		sampled_imgs = sampled_imgs.permute((0, 2, 3, 1))

		self.init_basecolor = get_basecolor(
			img=sampled_imgs,
			n_clusters=cluster_init_number,
			cluster_th=cluster_merge_threshold,
			n_clusters_minimum=cluster_number_lower_bound,
			visualize=visualize,
			normalize=normalize
		)
		self.num_cluster = self.init_basecolor.shape[0]


def load_dataset(dataset_type, basedir, **kwargs) -> NerfDataset:
	from dataset.dataset_clevr import ClevrDataset
	from dataset.dataset_mitsuba import MitsubaDataset
	from dataset.dataset_mitsuba_eval import MitsubaEvalDataset
	from dataset.dataset_nerf_synthetic import NeRFSyntheticDataset
	from dataset.dataset_replica import ReplicaDataset
	from dataset.dataset_falcor import FalcorDataset
	from dataset.dataset_colmap import ColmapDataset
	from dataset.dataset_nerf_colmap import NeRFColmapDataset
	from dataset.dataset_real import RealDataset

	if dataset_type == "clevr":
		return ClevrDataset(basedir, **kwargs)
	elif dataset_type == "mitsuba":
		return MitsubaDataset(basedir, **kwargs)
	elif dataset_type == "mitsuba_eval":
		return MitsubaEvalDataset(basedir, **kwargs)
	elif dataset_type == "nerf_synthetic":
		return NeRFSyntheticDataset(basedir, **kwargs)
	elif dataset_type == "replica":
		return ReplicaDataset(basedir, **kwargs)
	elif dataset_type == "falcor":
		return FalcorDataset(basedir, **kwargs)
	elif dataset_type == "colmap":
		return ColmapDataset(basedir, **kwargs)
	elif dataset_type == "nerfcolmap":
		return NeRFColmapDataset(basedir, **kwargs)
	elif dataset_type == "real":
		return RealDataset(basedir, **kwargs)
