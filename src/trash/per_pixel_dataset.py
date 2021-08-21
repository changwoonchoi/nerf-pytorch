from torch.utils.data import Dataset
from dataset.dataset_interface import NerfDataset
import numpy as np


class PerPixelDataset(Dataset):
	def __init__(self, images):
		self.height, self.width, _ = images[0].shape
		self.images = images
		self.pixel_rgb = np.reshape(self.images, [-1, 3])

	def __len__(self):
		return len(self.images) * self.width * self.height

	def __getitem__(self, index):
		rgb = self.pixel_rgb[index]
		sample = {}
		sample['rgb'] = rgb
		return sample
