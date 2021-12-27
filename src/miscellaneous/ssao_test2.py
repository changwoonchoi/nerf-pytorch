import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn

def depth_to_normal(depth_path, pose, camera_angle_x):
	depth = cv2.imread(depth_path)
	depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
	depth = np.asarray(depth, dtype=np.float32) / 255.0
	height, width, channel = depth.shape

	depth = 1.0 / (depth[:,:,0:1] + 1e-10)
	# depth = depth[:,:,0:1]
	depth = torch.Tensor(depth)
	depth = torch.permute(depth, (2, 0, 1))
	print(depth.shape)

	kn = torch.Tensor([[-1 ,-1,-1],[-1, 8 ,-1], [-1, -1, -1]]).unsqueeze(0).unsqueeze(0)
	conv = nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1, bias=False)
	conv.weight = nn.Parameter(kn)

	depth_conv = conv(depth.unsqueeze(0))
	depth_conv = torch.clip(depth_conv, 0, 1)

	depth_conv = torch.permute(depth_conv[0], (1, 2, 0))
	print(depth_conv.shape)
	img = depth_conv.detach().numpy()[...,0]
	stacked_img = np.stack((img,) * 3, axis=-1)
	plt.imshow(stacked_img)
	plt.show()


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