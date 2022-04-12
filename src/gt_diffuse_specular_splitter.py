from utils.image_utils import *
import numpy as np
import matplotlib.pyplot as plt
import os


def bilinear_interpolate_single(im, x, y):
	x = x * im.shape[1]
	y = y * im.shape[0]
	x = np.asarray(x)
	y = np.asarray(y)

	x0 = np.floor(x).astype(int)
	x1 = x0 + 1
	y0 = np.floor(y).astype(int)
	y1 = y0 + 1

	x0r = np.clip(x0, 0, im.shape[1]-1)
	x1r = np.clip(x1, 0, im.shape[1]-1)
	y0r = np.clip(y0, 0, im.shape[0]-1)
	y1r = np.clip(y1, 0, im.shape[0]-1)

	Ia = im[ y0r, x0r ]
	Ib = im[ y1r, x0r ]
	Ic = im[ y0r, x1r ]
	Id = im[ y1r, x1r ]

	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)

	v = Ia*wa + Ib*wb + Ic*wc + Id*wd

	return v

def bilinear_interpolate(im, x, y):
	original_shape = x.shape
	x = x * im.shape[1]
	y = y * im.shape[0]
	x = np.asarray(x)
	y = np.asarray(y)
	x = x.flatten()
	y = y.flatten()

	x0 = np.floor(x).astype(int)
	x1 = x0 + 1
	y0 = np.floor(y).astype(int)
	y1 = y0 + 1

	x0r = np.clip(x0, 0, im.shape[1]-1)
	x1r = np.clip(x1, 0, im.shape[1]-1)
	y0r = np.clip(y0, 0, im.shape[0]-1)
	y1r = np.clip(y1, 0, im.shape[0]-1)

	Ia = im[ y0r, x0r ]
	Ib = im[ y1r, x0r ]
	Ic = im[ y0r, x1r ]
	Id = im[ y1r, x1r ]

	wa = (x1-x) * (y1-y)
	wb = (x1-x) * (y-y0)
	wc = (x-x0) * (y1-y)
	wd = (x-x0) * (y-y0)

	wa = wa[..., None]
	wb = wb[..., None]
	wc = wc[..., None]
	wd = wd[..., None]

	v = Ia*wa + Ib*wb + Ic*wc + Id*wd

	return v.reshape((*original_shape, 3))

def lerp(x1, x2, y1, y2, x):
	return (x - x1) / (x2 - x1) * (y2- y1) + y1

ibl = load_image_from_path("../data/ibl_brdf_lut.png")

def get_prefiltered_value(roughness_list, roughness):
	for i, value in enumerate(roughness_list):
		r, image = value
		if roughness < r:
			r_prev = roughness_list[i-1][0]
			image_prev = roughness_list[i - 1][1]
			return lerp(image_prev, image, r_prev, r, roughness)
	return roughness_list[-1][1]

def tonemap_and_gamma(img, gamma=2.2):
	img[img < 0] = 0.0
	img = img / (1 + img)
	img = np.power(img, 1 / gamma)
	return img

def split_sum_approximation(folder, i, out_path=None):
	def load_image_specific(target, to_linear=False):
		if target is None:
			path = "%s/%d.png" % (folder, i)
		else:
			path = "%s/%d_%s.png" % (folder, i, target)

		image = load_image_from_path(path)
		if to_linear:
			image = np.power(image, 2.2)
			image = image / np.maximum(1 - image, 0.000001)
		return image

	radiance = load_image_specific(None, to_linear=True)
	albedo = load_image_specific("albedo", to_linear=False)
	irradiance = load_image_specific("irradiance", to_linear=True)
	#mirror = load_image_specific("mirror", to_linear=True)

	n_dot_v = load_image_specific("n_dot_v", to_linear=False)[...,0]
	roughness = load_image_specific("roughness", to_linear=False)[...,0]

	envBRDF = bilinear_interpolate(ibl, n_dot_v, roughness)
	envBRDF1 = envBRDF[..., 0:1]
	envBRDF0 = envBRDF[..., 1:2]

	roughness = np.stack([roughness] * 3, axis=-1)
	n_dot_v = np.stack([n_dot_v] * 3, axis=-1)

	F0 = np.array([0.04, 0.04, 0.04]) * roughness + albedo * (1-roughness)
	F1 = np.maximum(1 - roughness, F0) - F0

	F = F0 + F1 * np.power(np.clip(1-n_dot_v, 0, 1), 5)
	specular_coeff = F * envBRDF1 + envBRDF0
	diffuse = albedo * irradiance * roughness * (1-F)
	#prefiltered_value = mirror * (1-roughness) + irradiance * roughness

	#specular = specular_coeff * prefiltered_value
	# diffuse = np.where(radiance < diffuse, radiance, diffuse)
	diffuse = np.where(roughness == 1, radiance, diffuse)
	diffuse = np.where(roughness == 0, np.zeros_like(diffuse), diffuse)
	diffuse = np.where(radiance < diffuse, radiance, diffuse)

	specular = radiance - diffuse
	specular = np.where(specular < 0, 0, specular)

	diffuse = tonemap_and_gamma(diffuse)
	specular = tonemap_and_gamma(specular)

	if out_path:
		save_pred_images(diffuse, out_path + "/%d_diffuse" % i)
		save_pred_images(specular, out_path + "/%d_specular" % i)

	return diffuse, specular


if __name__ == "__main__":
	basedir = "../data/mitsuba/"
	out_basedir = "../data/mitsuba/"

	targets = ["bathroom2", "bedroom", "kitchen", "living-room-2", "living-room-3", "staircase", "veach-ajar", "veach_door_simple"]
	targets = [ "kitchen", "living-room-2", "bedroom", "veach-ajar"]
	targets = ["living-room", "classroom", "bathroom", "dining-room"]
	splits = ["train", "test", "val"]

	target_configs = []
	for target in targets:
		for split in splits:
			path = os.path.join(basedir, target, split)
			out_path = os.path.join(out_basedir, target, split)

			for i in range(100):
				target_configs.append((path, i+1, out_path))
	from multiprocessing import Pool

	with Pool(40) as p:
		p.starmap(split_sum_approximation, target_configs)