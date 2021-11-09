import torch
import torch.nn.functional as F


def get_normal_from_sigma_gradient_surface(x_surface, network_query_fn, network_fn):
	x_surface.requires_grad = True
	sigma_x_surface = network_query_fn(x_surface, None, network_fn)
	sigma_x_surface = F.relu(sigma_x_surface)
	sigma_x_surface.backward(torch.ones_like(sigma_x_surface))
	normal_map_from_sigma_gradient_surface = -F.normalize(x_surface.grad, dim=-1)
	normal_map_from_sigma_gradient_surface.detach_()
	x_surface.requires_grad = False
	return normal_map_from_sigma_gradient_surface


def get_normal_from_sigma_gradient(pts, weights, network_query_fn, network_fn):
	pts.requires_grad = True
	sigma_x = network_query_fn(pts, None, network_fn)
	sigma_x = F.relu(sigma_x)
	sigma_x.backward(torch.ones_like(sigma_x))
	normal_from_sigma_gradient = -F.normalize(pts.grad, dim=-1)
	normal_from_sigma_gradient.detach_()
	pts.requires_grad = False
	normal_map_from_sigma_gradient = torch.sum(weights[..., None] * normal_from_sigma_gradient, -2)
	normal_map_from_sigma_gradient.detach_()
	return normal_map_from_sigma_gradient


