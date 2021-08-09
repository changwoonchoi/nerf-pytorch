import torch

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
		label_onehot: (H, W, N_instance) one-hot encoded torch tensor
		label: (H, W) stores instance num
	"""
	label_onehot = torch.zeros([colored_mask.shape[0], colored_mask.shape[1], len(color_list)])
	label = torch.zeros([colored_mask.shape[0], colored_mask.shape[1]])

	for i, color in enumerate(color_list):
		one_hot = torch.zeros(len(color_list))
		one_hot[i] = 1
		mask_i = (colored_mask.view([-1, 3]) == color).reshape([label.shape[0], label.shape[1], 3])
		mask_i = torch.logical_and(torch.logical_and(mask_i[:, :, 0], mask_i[:, :, 1]), mask_i[:, :, 2])
		label_onehot[mask_i] = one_hot
		label[mask_i] = i
	return label_onehot, label
