from dataset.dataset_interface import NerfDataset
from dataset.dataset_clevr import ClevrDataset


def load_dataset(dataset_type, basedir, **kwargs) -> NerfDataset:
	if dataset_type == "clevr":
		return ClevrDataset(basedir, **kwargs)
