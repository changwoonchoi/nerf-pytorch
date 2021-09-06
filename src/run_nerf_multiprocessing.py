import subprocess
import json
from multiprocessing import Queue, Lock, Process, Pool
import os
from pathlib import Path
from pprint import pprint
import sys


def find_all_configs(directory):
	configs = []
	if isinstance(directory, list):
		for single_directory in directory:
			configs += find_all_configs(single_directory)
	else:
		if os.path.isfile(directory):
			configs.append(directory)
		else:
			for path in Path(directory).rglob('*.txt'):
				path = str(path)
				if 'common' not in path and 'default' not in path:
					configs.append(path)
	return configs


def run_single_process(gpu_id, queue, lock):
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

	while True:
		with lock:
			if queue.empty():
				break
			else:
				config_file = queue.get()

		command = "python run_nerf.py --config %s" % config_file

		print("----------------GPU ID-----------", gpu_id)
		print(command)
		subprocess.run(command.split(" "))


class MultiProcessingRenderer:
	def __init__(self, config_file):
		self.json_config_file = config_file

	def run(self):
		# load json file
		with open(self.json_config_file, "r") as f:
			multiprocessing_configs = json.load(f)

		# find all configs
		config_files = find_all_configs(multiprocessing_configs["config_lists"])
		config_files.sort()
		pprint(config_files)

		gpu_ids = multiprocessing_configs.pop("available_gpus")

		queue = Queue()
		for config in config_files:
			queue.put(config)

		procs = []
		lock = Lock()

		if len(gpu_ids) > 1:
			for i in range(len(gpu_ids)):
				gpu_id = gpu_ids[i]
				proc = Process(target=run_single_process, args=(gpu_id, queue, lock))
				procs.append(proc)
				proc.start()

			for proc in procs:
				proc.join()
		else:
			print("Single!")
			run_single_process(gpu_ids[0], queue, lock)


if __name__ == '__main__':
	m = MultiProcessingRenderer(sys.argv[1])
	m.run()
