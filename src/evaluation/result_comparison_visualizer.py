import os
from utils.image_utils import *
import matplotlib.pyplot as plt


def load_image_target(folder, target, index):
	file_path = os.path.join(folder, "testset_099999", target + "_{:03d}.png".format(index))
	return load_image_from_path(file_path, scale=1)


def load_image_target_gt(scene, target, index, scale):
	if target=="rgb":
		file_path = os.path.join("../../data/mitsuba", scene, "test", "%d.png" % index)
	else:
		file_path = os.path.join("../../data/mitsuba", scene, "test", "%d_%s.png" % (index, target))
	return load_image_from_path(file_path, scale=scale)


def visualize_comparison(basedir, scene_name, index=1, exp_names=None, compare_targets=None, skip=1, scale=1):
	exp_names_dict = {
		"ours": "Ours",
		"ours_hdr": "Ours",
		"monte_carlo_nerf_surface": "MC",
		"monte_carlo_env_map": "MC + Env",
		"gt": "GT"
	}

	if exp_names is None:
		exp_names = ["monte_carlo_nerf_surface", "monte_carlo_env_map", "ours_hdr", "gt"]
	if compare_targets is None:
		compare_targets = ["diffuse", "specular", "rgb"]
		# compare_targets = ["albedo", "irradiance", "roughness", "diffuse", "specular", "rgb"]

	n_row = len(exp_names)
	n_col = len(compare_targets)
	fig = plt.figure(figsize=(2 * n_col + 2, 2 * n_row))
	fig_index = 1

	for i_exp, exp_name in enumerate(exp_names):
		path = os.path.join(basedir, scene_name, exp_name)

		for i_target, compare_target in enumerate(compare_targets):

			if exp_name == "gt":
				image = load_image_target_gt(scene_name, compare_target, skip * index + 1, 1/scale)
			else:
				image = load_image_target(path, compare_target, index)
			ax = fig.add_subplot(n_row, n_col, fig_index)
			# plt.axis('off')
			plt.xticks([])
			plt.yticks([])
			if i_exp == 0:
				ax.set_xlabel(compare_target)
				ax.xaxis.set_label_position('top')
			if i_target == 0:
				ax.set_ylabel(exp_names_dict[exp_name])

			ax.imshow(image)
			fig_index += 1

	plt.suptitle("Index: %d"% index)
	fig.tight_layout()
	plt.show()

for i in range(100):
	visualize_comparison("../../logs_eval/final_config_lindisp_equal_sample/", "kitchen", index=i+1)