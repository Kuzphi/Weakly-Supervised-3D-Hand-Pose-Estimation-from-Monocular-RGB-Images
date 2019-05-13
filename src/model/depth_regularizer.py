from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch import nn

from src.utils.misc import to_cuda
from src.model.BaseModel import BaseModel
from src.model.utils.loss import CPMMSELoss
from src.model.utils.evaluation import get_preds_from_heatmap
from src.model.utils.data_reprocess import reprocess
__all__ = ['depth_regularizer']

class depth_regularizer(BaseModel):
	"""docstring for depth_regularizer"""
	def __init__(self, cfg):
		super(depth_regularizer, self).__init__(cfg)

	def forward(self):
		self.outputs = self.networks['Regression'](to_cuda(self.batch['input']))

		root_depth = self.batch['root_depth']
		index_bone_length = self.batch['index_bone_length']	
		depthmap_max = self.batch['depthmap_max']
		depthmap_range = self.batch['depthmap_range']

		reg_input = torch.zeros(self.batch['coor3d'].shape).cuda()
		reg_input[:, :, :2] = self.batch['coor2d'][:, :, :2].cuda().clone()
		reg_input[:, :, 2] = self.outputs['depth'] * index_bone_length.unsqueeze(1).cuda() + root_depth.unsqueeze(1).cuda()
		reg_input[:, :, 2] = (depthmap_max.unsqueeze(1).cuda() - reg_input[:, :, 2]) / depthmap_range.unsqueeze(1).cuda()
		reg_input = reg_input.view(reg_input.size(0), -1, 1, 1)
		self.outputs = self.networks['DepthRegularizer'](reg_input).squeeze()

		self.loss = self.criterion()
	def step(self):
		self.forward()
		for optimizer in self.optimizers.values():
			optimizer.zero_grad()

		self.loss.backward()

		for optimizer in self.optimizers.values():
			optimizer.step()
	
	def reprocess(self, input, reprocess_cfg):
		return reprocess(input, reprocess_cfg)

	def get_batch_result(self):
		return {}

	def eval_batch_result(self):
		return {}
		
	def eval_epoch_result(self):
		return {}

	def criterion(self):
		criterion = nn.L1Loss()
		return criterion(self.outputs, self.batch['depthmap'].float().cuda())