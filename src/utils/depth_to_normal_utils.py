import torch
import numpy as np
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
	rays_p = c2w[:3,-1] + rays_d * d[...,None]
	return rays_p


def depth_to_normal(depth_map, pose, K):
	height, width = depth_map.shape
	position = depth_to_position(height, width, K, pose, depth_map)
	normal = torch.zeros((*depth_map.shape, 3))

	def get_value(x, y):
		n_i = np.clip(x, 0, width - 1)
		n_j = np.clip(y, 0, height - 1)
		return position[n_j, n_i, :]

	for i in range(width):
		for j in range(height):
			s01 = get_value(i - 1, j)
			s21 = get_value(i + 1, j)
			s10 = get_value(i, j - 1)
			s12 = get_value(i, j + 1)

			va = s21 - s01
			vb = s12 - s10

			va = F.normalize(va, dim=-1)
			vb = F.normalize(vb, dim=-1)

			vc = torch.cross(vb, va)
			vc = F.normalize(vc, dim=-1)
			normal[j,i,:] = vc
	return normal
