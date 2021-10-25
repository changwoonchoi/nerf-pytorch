import logging

import numpy as np
from torch.utils.data import Dataset, DataLoader
from abc import ABC
from utils.logging_utils import load_logger
import torch
from utils.color_utils import get_basecolor
from torchvision import transforms


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

		self.near = 0
		self.far = 0

		self.images = []
		self.poses = []
		self.masks = []

		self.instance_color_list = []
		self.instance_num = 0

		self.full_data_loaded = False
		self.use_caching = True
		self.logger = load_logger("NeRF DataSet")
		self.logger.setLevel(logging.DEBUG)

		self.load_instance_label_mask = False

		# base color clustering related
		self.cluster_image_resize = 0.5
		self.cluster_image_number = 10
		self.num_init_cluster = 10
		self.cluster_th = 0
		self.init_basecolor = None
		self.num_cluster = 0

	def get_focal_matrix(self):
		K = np.array([
			[self.focal, 0, 0.5 * self.width],
			[0, self.focal, 0.5 * self.height],
			[0, 0, 1]
		]).astype(np.float32)
		return K

	def get_near_far_plane(self):
		return {'near': self.near, 'far': self.far}

	def get_test_render_poses(self):
		pass

	def load_all_data(self, num_of_workers=10):
		"""
		Load all data using multiprocessing
		:param num_of_workers: number of multiprocess
		:return: None
		"""
		if self.full_data_loaded:
			return
		data_loader = DataLoader(self, num_workers=num_of_workers, batch_size=1)
		for i, data in enumerate(data_loader):
			self.images.append(data["image"][0])
			self.poses.append(data["pose"][0])
			if self.load_instance_label_mask:
				self.masks.append(data["mask"][0])
		self.full_data_loaded = True

	def to_tensor(self, device):
		self.images = torch.stack(self.images, 0).to(device)
		self.poses = torch.stack(self.poses, 0).to(device)
		if self.init_basecolor is not None:
			self.init_basecolor = torch.from_numpy(self.init_basecolor).to(device)
		if self.load_instance_label_mask:
			self.masks = torch.stack(self.masks, 0).to(device)

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
			cluster_image_number=50,
			cluster_image_resize=0.5,
			cluster_init_number=20,
			cluster_merge_threshold=0.1,
			cluster_number_lower_bound=8,
			visualize=False
	):
		if cluster_image_number == -1:
			random_indices = np.arange(len(self.images))
		else:
			random_indices = np.random.permutation(len(self.images))[0:cluster_image_number]
		new_width = int(self.width * cluster_image_resize)
		new_height = int(self.height * cluster_image_resize)
		p = transforms.Compose([transforms.Resize((new_height, new_width))])
		sampled_imgs = []
		for i in random_indices:
			sampled_imgs.append(self.images[i])
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
		)
		self.num_cluster = self.init_basecolor.shape[0]


def load_dataset(dataset_type, basedir, **kwargs) -> NerfDataset:
	from dataset.dataset_clevr import ClevrDataset
	from dataset.dataset_mitsuba import MitsubaDataset
	if dataset_type == "clevr":
		return ClevrDataset(basedir, **kwargs)
	elif dataset_type == "mitsuba":
		return MitsubaDataset(basedir, **kwargs)
	#elif dataset_type == "clevr_decomp":
	#	return ClevrDecompDataset(basedir, **kwargs)
