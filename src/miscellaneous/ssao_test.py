import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


# Ray helpers
def depth_to_position(H, W, K, c2w, d):
	i, j = torch.meshgrid(torch.linspace(0, W-1, W), torch.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
	i = i.t()
	j = j.t()
	dirs = torch.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -torch.ones_like(i)], -1)
	dirs = F.normalize(dirs, dim=-1)
	# Rotate ray directions from camera frame to the world frame
	rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
	# Translate camera frame's origin to the world frame. It is the origin of all rays.
	rays_p = c2w[:3,-1] + rays_d.numpy() * d
	return rays_p


def normalize(x):
	return x / np.linalg.norm(x)


def depth_to_normal(depth_path, pose, camera_angle_x):
	depth = cv2.imread(depth_path)
	depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
	depth = np.asarray(depth, dtype=np.float32) / 255.0
	height, width, channel = depth.shape
	# depth = np.zeros((width, height))
	# depth_padded = np.pad(depth, ((1, 1), (1, 1)), 'edge')
	# s01 = depth[:-2, 1:-1]
	# s21 = depth[2:, 1:-1]
	# s10 = depth[1:-1, :-2]
	# s12 = depth[1:-1, 2:]
	# print(height, width)
	normal = np.zeros_like(depth)

	focal = .5 * width / np.tan(0.5 * camera_angle_x)
	K = np.array([
		[focal, 0, 0.5 * width],
		[0, focal, 0.5 * height],
		[0, 0, 1]
	]).astype(np.float32)

	position = depth_to_position(height, width, K, pose, 1.0 / (depth[:,:,0:1] + 1e-10))
	# plt.imshow(position[:, :, 0])
	# plt.show()
	# plt.imshow(position[:, :, 1])
	# plt.show()
	# plt.imshow(position[:, :, 2])
	# plt.show()

	def get_value(x, y):
		n_i = np.clip(x, 0, width - 1)
		n_j = np.clip(y, 0, height - 1)
		return position[n_j, n_i, :]

	kernel_size = 16

	for i in range(width):
		for j in range(height):
			s01 = get_value(i - 1, j)
			s21 = get_value(i + 1, j)
			s10 = get_value(i, j - 1)
			s12 = get_value(i, j + 1)

			va = s21 - s01
			vb = s12 - s10

			va = normalize(va)
			vb = normalize(vb)

			vc = np.cross(vb, va)
			vc = normalize(vc)

			normal[j,i,:] = vc
	plt.imshow((normal + 1) * 0.5)
	plt.show()

	binormal, tangent = get_TBN(normal)
	#binormal = torch.Tensor(binormal)
	#tangent = torch.Tensor(tangent)

	TBNs = np.stack([tangent, binormal, normal], axis=-1)

	samples = []
	for i in range(kernel_size):
		samples.append(cosine_sample_hemisphere())
	samples = np.array(samples)
	#position = torch.Tensor(position)

	radius = 0.01
	print(position.shape)

	samplePos = position + radius * TBNs * samples
	print(samplePos.shape)
	# for i in range(width):
	# 	for j in range(height):
	# 		TBN = TBNs[j, i, :]
	# 		fragPos = position[j, i, :]
	# 		for k in range(kernel_size):
	# 			samplePos = fragPos + radius * TBN * samples[i]


	print(normal.shape)
	plt.imshow((normal + 1) * 0.5)
	plt.show()

import random
def cosine_sample_hemisphere():
	u1 = random.random()
	u2 = random.random()
	r = np.sqrt(u1)
	phi = 2 * np.pi * u2
	x = r * np.cos(phi)
	y = r * np.sin(phi)
	z = np.sqrt(max(0, 1-x*x-y*y))
	return np.array([x, y, z], dtype=np.float32)


def get_TBN(normal):
	binormal = np.zeros_like(normal)
	binormal[..., 0] = np.where(normal[...,0] > normal[...,2], -normal[..., 1], 0)
	binormal[..., 1] = np.where(normal[...,0] > normal[...,2], normal[..., 0], -normal[...,2])
	binormal[..., 2] = np.where(normal[..., 0] > normal[..., 2], 0, normal[..., 1])
	binormal = binormal / np.linalg.norm(binormal, axis=-1, keepdims=True)
	tangent = np.cross(binormal, normal, axis=-1)

	return binormal, tangent

import os
import json
if __name__ == "__main__":
	target = "kitchen"
	basedir = '../../data/mitsuba/%s' % target
	with open(os.path.join(basedir, 'transforms_test.json'), 'r') as fp:
		meta = json.load(fp)
	skip = 10
	poses = []
	for frame in meta['frames'][::skip]:
		# (3) load pose information
		pose = np.array(frame['transform']).astype(np.float32)
		# Mitsuba --> camera forward is +Z !!
		pose[:3, 0] *= -1
		pose[:3, 2] *= -1
		poses.append(pose)
	camera_angle_x = float(meta['frames'][0]['fov_degree']) / 180.0 * np.pi

	for i in range(1):
		path = "../../logs_20211101/specular_ibl_normal_oracle/%s/infer_normal/testset_100000/disp_00%d.png" % (target, i)
		#path = "../../logs/specular_ibl_no_normalize/%s/infer_normal/testset_100000/disp_00%d.png" % (
		#target, i)
		#path = "../../logs/specular_ibl/%s/not_infer_normal/testset_015000/disp_00%d.png" % (
		# target, i)

		depth_to_normal(path, poses[i], camera_angle_x)