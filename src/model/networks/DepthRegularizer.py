import torch.nn as nn
import torch
__all__ = ['DepthRegularizer']

class DepthRegularizer(nn.Module):
	"""docstring for DepthRegularizer"""
	def __init__(self, **kwargs):
		super(DepthRegularizer, self).__init__()

		layers = [
			nn.ConvTranspose2d(in_channels= 63, out_channels=512, kernel_size=4, stride=4), #  4
			nn.ReLU(),
			# nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2), #  4
			# nn.ReLU(),
			nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2), #  8
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2), # 16
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2), # 32
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels=128, out_channels= 64, kernel_size=2, stride=2), # 64
			nn.ReLU(),
			nn.ConvTranspose2d(in_channels= 64, out_channels=  1, kernel_size=2, stride=2), #128
			nn.ReLU()
			# nn.ConvTranspose2d(in_channels= 16, out_channels=  1, kernel_size=2, stride=2), #256
			# nn.ReLU()
			]
		for layer in layers:
			if isinstance(layer, nn.ConvTranspose2d):
				layer.weight.data.uniform_(-.1, .1)
				layer.bias.data.zero_()
		self.deconv = nn.Sequential(*layers)

	def forward(self, x):
		return self.deconv(x)
