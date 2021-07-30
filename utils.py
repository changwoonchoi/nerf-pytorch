import torch

def label2color(label, color_list):
	"""
	params:
		label: (H, W) int torch tensor
		color_list: colors of labels
	returns:
		colored_mask: (H, W, 3)
	"""
	colored_mask = torch.zeros([label.shape[0], label.shape[1], 3])
	for i in range(int(torch.max(label).item())):
		mask_i = label == i
		colored_mask[mask_i] = color_list[i]
	# TODO: need to verify
	return colored_mask


def color2label(colored_mask, color_list):
	"""
	params:
		colored_mask: (H, W, 3)
		color_list: colors of labels
	returns:
		label: (H, W) torch tensor
	"""
	label = torch.zeros([colored_mask.shape[0], colored_mask.shape[1]])
	for i, color in enumerate(color_list):
		mask_i = torch.sum((colored_mask.view(-1, 3) == color), dim=-1).reshape([label.shape[0], label.shape[1]]).to(torch.bool)
		label[mask_i] = i
	return label.to(torch.uint8)
