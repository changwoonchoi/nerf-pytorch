import numpy as np
from dataset.dataset_interface import NerfDataset
from torch.utils.data import DataLoader
from nerf_models.nerf_renderer_helper import *
import torch
import matplotlib.pyplot as plt
from utils.logging_utils import load_logger
from utils.timing_utils import time_measure


def single_image_data_generator(image, ray_origins, ray_directions, batch_size):
    pixel_rgb = torch.reshape(image, [-1, 3])
    pixel_ray_o = torch.reshape(ray_origins, [-1, 3])
    pixel_ray_d = torch.reshape(ray_directions, [-1, 3])
    length = pixel_rgb.shape[0]
    for i in range(0, length, batch_size):
        s = i
        e = min(i + batch_size, length)
        yield pixel_rgb[s:e], pixel_ray_o[s:e], pixel_ray_d[s:e]


def sample_generator_all_image_merged(dataset: NerfDataset, batch_size=1024):
    """
    Get sample from all data merged. (all images should be loaded once)
    :param dataset: Nerf dataset
    :param batch_size: default is 1024
    :return:
    """
    # Generate ray information
    rays = [get_rays(dataset.height, dataset.width, dataset.get_focal_matrix(), p[:3,:4]) for p in dataset.poses]
    ray_o, ray_d = zip(*rays)
    ray_o = torch.stack(ray_o, 0)
    ray_d = torch.stack(ray_d, 0)
    images = torch.stack(dataset.images, 0)

    while True:
        yield from single_image_data_generator(images, ray_o, ray_d, batch_size)


def sample_generator_exhaustive_single_image(dataset: NerfDataset, batch_size=1024):
    """
    Get sample from random image.
    Use all samples from the sampled image, then move to next.
    :param dataset: Nerf dataset
    :param batch_size: default is 1024
    :return:
    """
    data_loader = DataLoader(dataset, shuffle=True, num_workers=10, batch_size=1)
    while True:
        for data in data_loader:
            ray_o, ray_d = get_rays(dataset.height, dataset.width, dataset.get_focal_matrix(), data["pose"][0, :3, :4])
            yield from single_image_data_generator(data["image"], ray_o, ray_d, batch_size)


def sample_generator_single_image(
        dataset: NerfDataset,
        batch_size=1024,
        visualize=False,
        precrop_iters=500,
        precrop_frac=0.5,
        initial_iters=0
):
    """
    Get sample from a single random image.
    Use only single random batch from the sampled image, then move to next.
    All dataset is loaded in advance
    :param dataset: Nerf dataset
    :param batch_size: default is 1024
    :param visualize : visualize sampled ray position
    :param precrop_iters: precrop iteration
    :param precrop_frac: precrop fraction
    :param initial_iters: initial iteration number
    :return:
    """

    logger = load_logger("Sample Generator")

    n_iters = initial_iters
    while True:
        random_image_index = np.random.randint(0, len(dataset), 1)[0]
        H = dataset.height
        W = dataset.width
        if n_iters < precrop_iters:
            dH = int(H//2 * precrop_frac)
            sH = max(H // 2 - dH, 0)
            eH = min(H // 2 + dH, H)
            dW = int(W//2 * precrop_frac)
            sW = max(W // 2 - dW, 0)
            eW = min(W // 2 + dW, W)
            # logger.info(f"\nCenter cropping of size {2*dH} x {2*dW} is enabled until iter {precrop_iters}")
        else:
            sH = 0
            eH = H
            sW = 0
            eW = W
        random_u = np.random.randint(sW, eW, batch_size)
        random_v = np.random.randint(sH, eH, batch_size)
        random_uv = np.stack([random_u, random_v], 1)

        pixel_info = dataset.get_info(random_image_index, random_u, random_v)

        pose = dataset.poses[random_image_index]
        # image = dataset.images[random_image_index]
        # pixel_rgb = image[random_v, random_u, :]    # height is first!!!
        # pixel_label = None
        # if dataset.load_instance_label_mask:
        #     mask = dataset.masks[random_image_index]
        #     pixel_label = mask[random_v, random_u]  # height is first!!!

        uv_t = torch.Tensor(random_uv)
        ray_o, ray_d = get_rays_few(uv_t, dataset.get_focal_matrix(), pose[:3, :4])

        if visualize:
            pixel_info["rgb"][random_v, random_u, 0] = 255
            pixel_info["rgb"][random_v, random_u, 1] = 0
            pixel_info["rgb"][random_v, random_u, 2] = 0
            plt.imshow(pixel_info["rgb"])
            plt.show()

        n_iters += 1
        yield pixel_info, ray_o, ray_d
