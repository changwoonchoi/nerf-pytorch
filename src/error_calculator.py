from utils.image_utils import *
import os
from dataset.dataset_interface import load_dataset
from piq import ssim, psnr
import torch
import pandas as pd


def load_eval_images(basedir, scene_name, exp_name, target_n=-1):
	path = os.path.join(basedir, scene_name, exp_name)
	if target_n == -1:
		load_folder = sorted(next(os.walk(path))[1])[-1]
	else:
		load_folder = 'testset_{:06d}'.format(target_n)
	load_folder = os.path.join(path, load_folder)

	mitsuba_eval = load_dataset("mitsuba_eval", load_folder)
	mitsuba_eval.load_all_data(4)
	mitsuba_eval.to_tensor("cpu")
	return mitsuba_eval


def eval_error(ground_truth, pred, target="diffuse", metric=None):
	if target == "diffuse":
		dataset_gt = ground_truth.diffuses
		dataset_prd = pred.diffuses
	elif target == "specular":
		dataset_gt = ground_truth.speculars
		dataset_prd = pred.speculars
	else:
		dataset_gt = ground_truth.images
		dataset_prd = pred.images
	dataset_gt = torch.permute(dataset_gt, (0, 3, 1, 2))
	dataset_prd = torch.permute(dataset_prd, (0, 3, 1, 2))

	if metric == "ssim":
		metric_f = ssim
	elif metric == "psnr":
		metric_f = psnr
	else:
		metric_f = torch.nn.MSELoss()
	value = metric_f(dataset_gt, dataset_prd)

	return value


def calculate_error_whole(basedir, scene_names=None, exp_names=None):
	if scene_names is None:
		scene_names = sorted(next(os.walk(basedir))[1])
	if exp_names is None:
		exp_names = sorted(next(os.walk(os.path.join(basedir, scene_names[0])))[1])
	# scene_names = ["bathroom", "beroom", "kitchen", "living-room-2", "living-room-3", "staircase", "veach-ajar", "veach_door_simple"]
	# scene_names = ["kitchen", "bathroom2"]
	# exp_names = ["monte_carlo_env_map", "monte_carlo_nerf_surface", "ours", "ours_hdr", "ours_smooth", "ours_smooth_hdr"]
	# exp_names = ["ours", "ours_with_gt_depth"]
	compare_targets = ["image", "diffuse", "specular"]
	metrics = ["ssim", "psnr", "mse"]
	df = pd.DataFrame()

	for scene in scene_names:
		load_params = {
			"load_diffuse_specular": True,
			"image_scale": 1/4,
			"skip": 10,
			"split": "test"
		}
		scene_gt_dataset = load_dataset("mitsuba", "../data/mitsuba/%s" % scene, **load_params)
		scene_gt_dataset.load_all_data(4)
		scene_gt_dataset.to_tensor("cpu")

		for exp_name in exp_names:
			exp_dataset = load_eval_images(basedir, scene, exp_name)

			for compare_target in compare_targets:
				metric_errors = {}
				for metric in metrics:
					error = eval_error(scene_gt_dataset, exp_dataset, compare_target, metric)
					metric_errors[metric] = float(error)
				df = df.append({"scene": scene, "exp_name": exp_name, "compare_target": compare_target, **metric_errors}, ignore_index=True)

	average = df.groupby(['compare_target', 'exp_name']).mean()
	average.to_csv(os.path.join(basedir, "error.csv"))
	df.to_csv(os.path.join(basedir, "error_total.csv"), index=False)

calculate_error_whole("../logs/final_config/", scene_names=["kitchen"])