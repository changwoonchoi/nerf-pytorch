import cv2
import numpy as np
import math

RESULT_DIR = '/home/ccw/project/nerf-pytorch/logs/'
GT_DIR = '/home/ccw/project/nerf-pytorch/data/mitsuba/'

def PSNR(inferred_img, target_img):
    mse = np.mean((inferred_img - target_img)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def SSIM(inferred_img, target_img):



def MSE(inferred_img, target_img):
    return np.mean((inferred_img - target_img)**2)


def test_metric():
    raise NotImplementedError


if __name__ == "__main__":
    test_metric()
