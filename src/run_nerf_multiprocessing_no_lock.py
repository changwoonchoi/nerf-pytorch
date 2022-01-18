import subprocess
import json
from multiprocessing import Queue, Lock, Process, Pool
import os
from pathlib import Path
from pprint import pprint
import sys
import natsort
import torch
torch.multiprocessing.set_sharing_strategy('file_system')


def find_all_configs(directory):
	configs = []
	if isinstance(directory, list):
		for single_directory in directory:
			configs += find_all_configs(single_directory)
	else:
		if os.path.isdir(directory):
			for path in Path(directory).rglob('*.txt'):
				path = str(path)
				if 'common' not in path and 'default' not in path:
					configs.append(path)
		else:
			for path in Path().rglob(directory):
				path = str(path)
				if 'common' not in path and 'default' not in path:
					configs.append(path)
	return configs


def run_single_process(gpu_id, configs):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
	# print(gpu_id, configs)
	for config_file in configs:

		command = "python run_nerf_decomp.py --config %s" % config_file

		print("----------------GPU ID-----------", gpu_id)
		print(command)
		subprocess.run(command.split(" "))

		# try:
		#
		# except Exception:
		# 	pass
	print("Process finished! at %d GPU" % gpu_id)


class MultiProcessingRenderer:
	def __init__(self, config_file):
		self.json_config_file = config_file

	def run(self):
		# load json file
		with open(self.json_config_file, "r") as f:
			multiprocessing_configs = json.load(f)

		# find all configs
		config_files = find_all_configs(multiprocessing_configs["config_lists"])
		config_files = natsort.natsorted(config_files) #config_files.sort()
		pprint(config_files)
		gpu_ids = multiprocessing_configs.pop("available_gpus")

		procs = []
		gpu_configs = [[] for _ in range(len(gpu_ids))]
		index = 0
		for config in config_files:
			gpu_configs[index].append(config)
			index += 1
			index %= len(gpu_ids)

		if len(gpu_ids) > 1:
			for i in range(len(gpu_ids)):
				gpu_id = gpu_ids[i]
				proc = Process(target=run_single_process, args=(gpu_id, gpu_configs[i]))
				procs.append(proc)
				proc.start()

			for proc in procs:
				proc.join()
		else:
			run_single_process(gpu_ids[0], config_files)


if __name__ == '__main__':
	m = MultiProcessingRenderer(sys.argv[1])
	m.run()
