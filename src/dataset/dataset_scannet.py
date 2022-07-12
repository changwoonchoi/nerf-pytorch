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
import glob
from utils.image_utils import *


class ScannetDataset(NerfDataset):
    def __init__(self, basedir, **kwargs):
        super().__init__("scannet", **kwargs)
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

        # with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
        #     self.meta = json.load(fp)

        self.instance_color_list = []
        self.instance_num = 0

        self.basedir = basedir

        self.skip = kwargs.get("skip", 1)
        if self.split == "train":
            self.skip = 10

        image0_path = os.path.join(self.basedir, "color", "0.jpg")
        image0 = imageio.imread(image0_path, pilmode='RGB')
        self.original_height, self.original_width, _ = image0.shape

        self.height = int(self.original_height * self.scale)
        self.width = int(self.original_width * self.scale)
        self.K = np.loadtxt(os.path.join(self.basedir, "intrinsic", "intrinsic_color.txt"))[:3, :3]
        self.K *= self.scale
        self.sequence_length = glob.glob(os.path.join(self.basedir, "color", "*.jpg")).__len__()
        # self.focal = .5 * self.width / np.tan(0.5 * self.camera_angle_x)

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
        if self.split == "train":
            return self.sequence_length // self.skip + 1
        elif self.split == "val" or "test":
            return (self.sequence_length - 5) // self.skip + 1
        # return len(glob.glob(os.path.join(self.basedir, "color", "*.jpg"))[::self.skip])

    def __getitem__(self, index):
        sample = {}

        """
        Load single data corresponding to specific index
        :param index: data index
        """
        if self.split == "train":
            image_file_path = os.path.join(self.basedir, "color", "%d.jpg" % (self.skip * index))  # train dataset : 0, 10, 20, ...
            prior_albedo_file_path = os.path.join(self.basedir, "color", "{}_{}_r.png".format(self.skip * index, self.prior_type))
            prior_irradiance_file_path = os.path.join(self.basedir, "color", "{}_{}_s.png".format(self.skip * index, self.prior_type))
            pose_file_path = os.path.join(self.basedir, "pose", "%d.txt" % (self.skip * index))
        elif self.split == "val" or "test":
            image_file_path = os.path.join(self.basedir, "color", "%d.jpg" % (self.skip * index + 5))  # validation dataset : 5, 15, 25, ...
            prior_albedo_file_path = os.path.join(self.basedir, "color", "{}_{}_r.png".format(self.skip * index + 5, self.prior_type))
            prior_irradiance_file_path = os.path.join(self.basedir, "color", "{}_{}_s.png".format(self.skip * index + 5, self.prior_type))
            pose_file_path = os.path.join(self.basedir, "pose", "%d.txt" % (self.skip * index + 5))
        else:
            raise ValueError

        # (1) load RGB Image
        if self.load_image:
            sample["image"] = load_image_from_path(image_file_path, scale=self.scale)
        if self.load_normal and self.load_albedo and self.load_roughness and self.load_depth and self.load_irradiance and self.load_diffuse_specular:
            raise ValueError
        if self.load_priors:
            sample["prior_albedo"] = load_image_from_path(prior_albedo_file_path, scale=self.scale)
            sample["prior_irradiance"] = load_image_from_path(prior_irradiance_file_path, scale=self.scale)


        # (2) load instance_label_mask
        if self.load_instance_label_mask:
            raise ValueError

        # (3) load pose information
        pose = np.loadtxt(pose_file_path)
        # Mitsuba --> camera forward is +Z !!
        # pose[:3, 0] *= -1
        # pose[:3, 2] *= -1
        sample["pose"] = pose
        return sample

    def get_test_render_poses(self):
        # TODO : implement
        return None
