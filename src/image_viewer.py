import cv2
import numpy as np

def load_image_from_path(image_file_path, scale=1):
	image = cv2.imread(image_file_path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	if scale != 1:
		image = cv2.resize(image, None, fx=scale, fy=scale)
	image = image.astype(np.float32)
	image /= 255.0

	return image

import matplotlib.pyplot as plt
import torch

def visualize_images_vertical(images, use_colorbar=True,\
	clim_val=None, horizontal=True, title=None, titles=None):
	s = 2
	if horizontal:
		plt.figure(figsize=(len(images) * s, s))
	else:
		plt.figure(figsize=(1, len(images)))

	if title is not None:
		plt.suptitle(title)

	n = len(images)
	for i in range(n):
		image = images[i]
		if isinstance(image, torch.Tensor):
			image = image.cpu().numpy()[0]

		if horizontal:
			plt.subplot(1, n, i+1)
		else:
			plt.subplot(n, 1, i+1)

		plt.imshow(image)
		plt.axis('off')
		if titles:
			plt.title(titles[i])

		if isinstance(clim_val, list):
			plt.clim(0, clim_val[i])
		elif isinstance(clim_val, float):
			plt.clim(0, clim_val)
		#if use_colorbar:
		#	plt.colorbar(label='color')


def split_sum_approximation(paths, i, target=None, titles=None):
	def load_image_specific(folder):
		if target is None:
			path = "%s/rgb_00%d.png" % (folder, i)
		else:
			path = "%s/%s_00%d.png" % (folder, target, i)
		print(path)
		image = load_image_from_path(path)
		return image

	images = []
	for p in paths:
		images.append(load_image_specific(p))
	visualize_images_vertical(images, title=target, titles=titles)



if __name__ == "__main__":
	# path = "../result_20211130/seg_nerf_dataset_scale_4/cornell-box_specular/0.05/train"
	# path_diff = ["0.0", "0.01", "0.05", "0.1", "0.2"]
	# path_diff = ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]
	path_diff = ["1", "0.3", "0.1", "0.03", "0.01"]

	basedir = "../logs/roughness_smooth_compare/kitchen/roughness_smooth_decay_"
	path = [(basedir + p + "/testset_070000") for p in path_diff]
	targets = ["radiance", "rgb", "albedo", "roughness", "specular", "diffuse", "irradiance", "depth"]

	for target in targets:
		split_sum_approximation(path, 1, target=target, titles=path_diff)
	plt.show()
	#print(bilinear_interpolate_single(ibl, 0, 1))

