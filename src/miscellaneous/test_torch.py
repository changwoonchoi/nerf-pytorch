import torch
import numpy as np


def test_torch():
	a = torch.Tensor(np.zeros((5, 4)))
	b = torch.Tensor(np.zeros((1, 4)))
	c = b - a
	print(c.shape)

test_torch()