import os
import json
import imageio

from dataset.dataset_interface import NerfDataset
import math
from utils.image_utils import *

# Lots of codes are borrowed from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_llff.py

class RealDataset(NerfDataset):
    def __init__(self, basedir, **kwargs):
        super().__init__("real", **kwargs)

        self.basedir = basedir

        with open(os.path.join(basedir, 'transforms_{}.json'.format(self.split)), 'r') as fp:
            self.meta = json.load(fp)
        self.skip = kwargs.get("skip", 1)
        if self.split == "train":
            self.skip = 1
        # if kwargs.get("force_skip", False):
        #     self.skip = kwargs.get("skip", 1)

        image0_path = os.path.join(self.basedir, 'train/1.png')
        image0 = imageio.imread(image0_path, pilmode='RGB')
        self.original_height, self.original_width, _ = image0.shape

        self.height = int(self.original_height * self.scale)
        self.width = int(self.original_width * self.scale)

        self.near = self.meta["near"]
        self.far = self.meta["far"]

        self.focal = self.meta["focal"] * self.scale

    def __len__(self):
        return len(self.meta['frames'][::self.skip])

    def __getitem__(self, index):
        sample = {}

        frame = self.meta['frames'][::self.skip][index]
        image_file_path = os.path.join(self.basedir, self.split, "%d.png" % (self.skip * index + 1))

        sample["image"] = load_image_from_path(image_file_path, scale=self.scale)
        pose = np.array(frame['transform']).astype(np.float32)
        sample["pose"] = pose

        return sample
