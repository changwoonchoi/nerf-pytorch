import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
	def __init__(self,D=8, W=256, input_ch=3, skips=[4]):
		super().__init__()
		self.D = D
		self.W = W
		self.input_ch = input_ch
		self.skips = skips

		self.positions_linears = nn.ModuleList(
			[nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in
										range(D - 1)]
		)