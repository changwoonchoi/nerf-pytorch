import torch
import numpy as np
from functools import reduce


def label2color(label, color_list):
	"""
	params:
		label: (H, W, 6) one-hot encoded torch tensor
		color_list: colors of labels
	returns:
		colored_mask: (H, W, 3)
	"""
	colored_mask = torch.zeros([label.shape[0], label.shape[1], 3])
	mask = torch.argmax(label, dim=-1)
	for i in range(len(color_list)):
		mask_i = mask == i
		colored_mask[mask_i] = color_list[i].float().to(colored_mask.device)

	return colored_mask


def color2label(colored_mask, color_list):
	"""
	params:
		colored_mask: (H, W, 3)
		color_list: colors of labels
	returns:
		label: (H, W) stores instance num
	"""
	label = torch.zeros([colored_mask.shape[0], colored_mask.shape[1]])

	for i, color in enumerate(color_list):
		one_hot = torch.zeros(len(color_list))
		one_hot[i] = 1
		mask_i = (colored_mask.view([-1, 3]) == color).reshape([label.shape[0], label.shape[1], 3])
		mask_i = torch.logical_and(torch.logical_and(mask_i[:,:,0], mask_i[:,:,1]), mask_i[:,:,2])
		label[mask_i] = i
	return label

def colored_mask_to_label_map_np(colored_mask, color_list):
	"""
	params:
		colored_mask: (H, W, 3)
		color_list: colors of labels
	returns:
		label: (H, W) stores instance num
	"""
	f = lambda label, i: np.where(np.all(colored_mask == color_list[i], axis=-1), i, label)
	label_init = np.zeros(colored_mask.shape[:-1], dtype=np.int32)
	return reduce(f, list(range(len(color_list))), label_init)


def encode_mask_np(mask, instance_number, method, instance_color_list=None):
	"""
	params:
		mask: (H, W, 1)
		instance_number: N
		method: one of scale, one_hot, mask_color
	returns:
		label: (H, W) stores instance num
	"""
	if method == "scalar":
		encoded_mask = mask / instance_number
	elif method == "one_hot":
		encoded_mask = np.eye(instance_number)[mask]
	elif method == "mask_color":
		encoded_mask = instance_color_list[mask]

	return encoded_mask

if __name__ == "__main__":
	mask = np.array([1, 3, 2, 1, 2, 0])
	instance_number = 4
	instance_color_list = np.array(
		[
			[1,0,1],
			[0,0,1],
			[0,1,1],
			[1,1,1]
		]
	)

	colored_mask = np.array(
		[
			[[1,0,1],
			[1,0,1],
			[1,0,1]],
			[[0,0,1],
			[0,1,1],
			[1,1,1]]
		]
	)

	# result = encode_mask_np(mask, 4, "mask_color", instance_color_list)
	# print(result)

	result = colored_mask_to_indexed_mask_np(colored_mask, instance_color_list)
	print(result)
