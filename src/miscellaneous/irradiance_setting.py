import numpy as np
import glob
import imageio
import json


def find_representative_irradiance_value(dataset_type: str, room_name: str):
	if dataset_type == 'mitsuba':
		bell_irradiance_files = glob.glob(
			'../../data/mitsuba_no_transparent_with_prior/{}/train/*_bell_s.png'.format(
				room_name))
		ting_irradiance_files = glob.glob(
			'../../data/mitsuba_no_transparent_with_prior/{}/train/*_ting_s.png'.format(
				room_name))
	elif dataset_type == 'falcor':
		bell_irradiance_files = glob.glob(
			'../../data/falcor/{}/*_bell_s.png'.format(room_name))
		ting_irradiance_files = glob.glob(
			'../../data/falcor/{}/*_ting_s.png'.format(room_name))
	elif dataset_type == 'replica':
		bell_irradiance_files = glob.glob(
			'../../data/replica/{}/train/*_bell_s.png'.format(room_name))
		ting_irradiance_files = glob.glob(
			'../../data/replica/{}/train/*_bell_s.png'.format(room_name))
	elif dataset_type == 'real':
		bell_irradiance_files = glob.glob(
			'../../data/real_data/{}/images/*_bell_s.png'.format(room_name))
		print("(bell) {} files in {}".format(len(bell_irradiance_files), room_name))
		ting_irradiance_files = glob.glob(
			'../../data/real_data/{}/images/*_ting_s.png'.format(room_name))
		print("(ting) {} files in {}".format(len(bell_irradiance_files), room_name))
	elif dataset_type == 'bespoke':
		bell_irradiance_files = glob.glob(
			'../../data/Bespoke_Images/{}/images/*_bell_s.png'.format(room_name))
		print("(bell) {} files in {}".format(len(bell_irradiance_files), room_name))
		ting_irradiance_files = glob.glob(
			'../../data/Bespoke_Images/{}/images/*_ting_s.png'.format(room_name))
		print("(ting) {} files in {}".format(len(bell_irradiance_files), room_name))
	elif dataset_type == 'nerfing_mvs':
		bell_irradiance_files = glob.glob(
			'../../data/NerfingMVS/{}/images/*_bell_s.png'.format(room_name))
		print("(bell) {} files in {}".format(len(bell_irradiance_files), room_name))
		ting_irradiance_files = glob.glob(
			'../../data/NerfingMVS/{}/images/*_ting_s.png'.format(room_name))
		print("(ting) {} files in {}".format(len(ting_irradiance_files), room_name))
	elif dataset_type == 'scannet':
		bell_irradiance_files = glob.glob(
			'../../data/scannet/{}/images/*_bell_s.png'.format(room_name))
		print("(bell) {} files in {}".format(len(bell_irradiance_files), room_name))
		ting_irradiance_files = glob.glob(
			'../../data/scannet/{}/images/*_ting_s.png'.format(room_name))
		print("(ting) {} files in {}".format(len(ting_irradiance_files), room_name))
	else:
		raise ValueError

	bell_irradiances = []
	ting_irradiances = []

	for irradiance_file in bell_irradiance_files:
		bell_irradiances.append(imageio.imread(irradiance_file) / 255.)
	for irradiance_file in ting_irradiance_files:
		ting_irradiances.append(imageio.imread(irradiance_file) / 255.)

	bell_irradiances = np.stack(bell_irradiances, axis=0)
	ting_irradiances = np.stack(ting_irradiances, axis=0)

	mean_bell = np.mean(bell_irradiances)
	mean_ting = np.mean(ting_irradiances)
	# median_bell = np.median(bell_irradiances)
	# median_ting = np.median(ting_irradiances)

	mean = {'bell': mean_bell, 'ting': mean_ting}
	# median = {'bell': median_bell, 'ting': median_ting}
	return mean  # , median

import os

if __name__ == "__main__":
	#replica scenes
	# replica_scenes = os.listdir('../../data/replica')
	# for room in replica_scenes:
	# 	print("replica {} processing".format(room))
	# 	irradiance_mean = find_representative_irradiance_value('replica', room)
	# 	with open('../../data/replica/{}/avg_irradiance.json'.format(room), "w") as f:
	# 		data = {
	# 			"mean_bell": float(irradiance_mean['bell']),
	# 			"mean_ting": float(irradiance_mean['ting'])
	# 		}
	# 		json.dump(data, f)
	# mitsuba scenes
	# mitsuba_rooms = ['bathroom', 'bathroom2', 'bedroom', 'classroom', 'dining-room', 'kitchen', 'living-room', 'living-room-2', 'living-room-3', 'staircase', 'veach-ajar', 'veach_door_simple']
	# # rooms = ['kitchen']
	# for room in mitsuba_rooms:
	# 	print("mitsuba {} processing".format(room))
	# 	irradiance_mean = find_representative_irradiance_value('mitsuba', room)
	# 	with open('../../data/mitsuba_no_transparent_with_prior/{}/avg_irradiance.json'.format(room), "w") as f:
	# 		data = {
	# 			"mean_bell": float(irradiance_mean['bell']),
	# 			"mean_ting": float(irradiance_mean['ting'])
	# 		}
	# 		json.dump(data, f)

	# falcor_rooms = ['kitchen', 'living-room-2']
	#
	# # falcor scenes
	# for room in falcor_rooms:
	# 	print("falcor {} processing".format(room))
	# 	irradiance_mean = find_representative_irradiance_value('falcor', room)
	# 	with open('../../data/falcor/{}/avg_irradiance.json'.format(room), "w") as f:
	# 		data = {
	# 			"mean_bell": float(irradiance_mean['bell']),
	# 			"mean_ting": float(irradiance_mean['ting'])
	# 		}
	# 		json.dump(data, f)

	# real_rooms = ['951_new', '951-2_new']

	# # real scenes
	# for room in real_rooms:
	# 	print('real {} processing'.format(room))
	# 	irradiance_mean = find_representative_irradiance_value('real', room)
	# 	with open('../../data/real_data/{}/avg_irradiance.json'.format(room), "w") as f:
	# 		data = {
	# 			"mean_bell": float(irradiance_mean['bell']),
	# 			"mean_ting": float(irradiance_mean['ting'])
	# 		}
	# 		json.dump(data, f)

	# bespoke_rooms = ['Kitchen_colmap_new']

	# for room in bespoke_rooms:
	# 	print('bespoke {} processing'.format(room))
	# 	irradiance_mean = find_representative_irradiance_value('bespoke', room)
	# 	with open('../../data/Bespoke_Images/{}/avg_irradiance.json'.format(room), "w") as f:
	# 		data = {
	# 			"mean_bell": float(irradiance_mean['bell']),
	# 			"mean_ting": float(irradiance_mean['ting'])
	# 		}
	# 		json.dump(data, f)


	# nerfing_mvs_rooms = ['scene0000_01', 'scene0079_00', 'scene0158_00', 'scene0316_00', 'scene0521_00', 'scene0553_00', 'scene0616_00', 'scene0653_00']

	# for room in nerfing_mvs_rooms:
	# 	print('nerfing_mvs {} processing'.format(room))
	# 	irradiance_mean = find_representative_irradiance_value('nerfing_mvs', room)
	# 	with open('../../data/NerfingMVS/{}/avg_irradiance.json'.format(room), "w") as f:
	# 		data = {
	# 			"mean_bell": float(irradiance_mean['bell']),
	# 			"mean_ting": float(irradiance_mean['ting'])
	# 		}
	# 		json.dump(data, f)

	scannet_rooms = ['scene0126_02', 'scene0134_00']

	for room in scannet_rooms:
		print('scannet {} processing'.format(room))
		irradiance_mean = find_representative_irradiance_value('scannet', room)
		with open('../../data/scannet/{}/avg_irradiance.json'.format(room), "w") as f:
			data = {
				"mean_bell": float(irradiance_mean['bell']),
				"mean_ting": float(irradiance_mean['ting'])
			}
			json.dump(data, f)
