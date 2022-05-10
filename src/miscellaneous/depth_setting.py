import sys
sys.path.append("../")
from utils.image_utils import *
import os
import json

def find_min_max_depth_helper(basedir):
	min_value = 100000
	max_value = 0
	splits = ["train", "val", "test"]

	for split in splits:
		for i in range(100):
			depth_file_path = os.path.join(basedir, split, "%d_depth.npy" % (i + 1))
			depth_map = load_numpy_from_path(depth_file_path)
			min_value = min(min_value, np.min(depth_map))
			max_value = max(max_value, np.max(depth_map))
	print(basedir, min_value, max_value)
	path = os.path.join(basedir, "min_max_depth.json")
	with open(str(path), "w") as f:
		data = {
			"min_depth": float(min_value),
			"max_depth": float(max_value)
		}
		json.dump(data, f)

def find_min_max_depth():
	# targets = ["bathroom2", "bedroom", "kitchen", "living-room-2", "living-room-3", "staircase", "veach-ajar", "veach_door_simple"]
	targets = ["dining-room", "bathroom", "classroom", "living-room"]

	for target in targets:
		basedir = "../../data/mitsuba/%s" % target
		find_min_max_depth_helper(basedir)

find_min_max_depth()