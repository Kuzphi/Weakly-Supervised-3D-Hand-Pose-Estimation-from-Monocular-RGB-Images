from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import random
from torch import nn
from torch.nn.functional import interpolate
from src.model.BaseModel import BaseModel
from src.utils.misc import to_cuda, to_cpu, to_torch
from src.utils.imutils import resize, draw_heatmap
from src.model.utils.loss import CPMMSELoss
from src.model.utils.evaluation import get_preds_from_heatmap, calc_auc

__all__ = ['Weakly_direct_regression_with_depth']
class Weakly_direct_regression_with_depth(BaseModel):
	"""docstring for Weakly_direct_regression_with_depth"""
	def __init__(self, cfg):
		super(Weakly_direct_regression_with_depth, self).__init__(cfg)
	
	def transforms(self, cfg, img, depth, coor2d, matrix):
		# resize
		if cfg.has_key('RESIZE'):
			xscale, yscale = 1. * cfg.RESIZE / img.size(1), 1. * cfg.RESIZE / img.size(2) 
			coor2d[:, 0] *= xscale
			coor2d[:, 1] *= yscale
			scale =[[xscale,    0,  0],
					[0,    yscale,  0],
					[0,         0,  1]]
			matrix = np.matmul(scale, matrix)

			img = resize(img, cfg.RESIZE, cfg.RESIZE)
			depth = depth.unsqueeze(0)
			depth = interpolate(depth, (128, 128), mode = 'bilinear', align_corners = True)[0,...]

		if cfg.COLOR_NORISE:
			img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
			img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
			img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

		# assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 256
		return img, depth, coor2d, matrix

	def reprocess(self, input, data_cfg):
		img = input['img']
		depthmap = input['depthmap'].float()

		coor2d = input['coor2d']
		# assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 320
		matrix = input['matrix']
		meta = input['meta']
		#apply transforms into image and calculate cooresponding coor and camera instrict matrix
		if data_cfg.TRANSFORMS:
			img, depthmap, coor2d, matrix = self.transforms(data_cfg.TRANSFORMS, img, depthmap, coor2d, matrix)

		# if depthmap_max - depthmap_min < 1e-6:
		#     print(name, ": ", depthmap_max - depthmap_min)
		# depthmap = (depthmap.max() - depthmap) / (depthmap_max - depthmap_min)
		# print(depthmap)
		
		

		matrix = np.linalg.inv(matrix) #take the inversion of matrix

		coor3d = coor2d.clone()
		coor3d[:,:2] *= coor3d[:, 2:]
		coor3d = torch.matmul(coor3d, to_torch(matrix).transpose(0, 1))

		root_depth = coor2d[0, 2].clone()
		index_bone_length = torch.norm(coor3d[9,:] - coor3d[10,:]).float()
		
		relative_depth = (coor2d[:,2] - root_depth) / index_bone_length

		depthmap *= float(2**16 - 1)
		depthmap = (depthmap - root_depth) / index_bone_length
		dpethmap = depthmap.float()

		depthmap_max = depthmap.max()
		depthmap_min = depthmap.min()
		depthmap = (depthmap - depthmap_min) / (depthmap_max - depthmap_min)
		
		heatmap = torch.zeros(data_cfg.NUM_JOINTS, img.size(1), img.size(2))
		depth   = torch.zeros(data_cfg.NUM_JOINTS, img.size(1), img.size(2))

		for i in range(21):
			heatmap[i] = draw_heatmap(heatmap[i], coor2d[i], data_cfg.HEATMAP.SIGMA)
			depth[i]   = heatmap[i] * (coor2d[i, 2] - coor2d[0, 2]) / index_bone_length


		return {'input': {'img':img,
						  'depthmap': depthmap,
						  },
				'heatmap': heatmap,
				'matrix': to_torch(matrix),
				'color_hm': heatmap,
				'depth_hm' :  depth,
				'depthmap': depthmap,
				'depthmap_max': depthmap_max,
				'depthmap_range': depthmap_max - depthmap_min,
				'coor3d': coor3d,
				'coor2d': coor2d,
				'root_depth': root_depth,
				'index_bone_length': index_bone_length,
				'relative_depth': relative_depth,
				'weight': 1,
				'meta': meta,
				}

	def forward(self):
		self.outputs = self.networks['Regression'](to_cuda(self.batch['input']))

		root_depth = self.batch['root_depth']
		index_bone_length = self.batch['index_bone_length']	
		depthmap_max = self.batch['depthmap_max']
		depthmap_range = self.batch['depthmap_range']

		reg_input = torch.zeros(self.batch['coor3d'].shape)
		reg_input[:, :, :2] = self.batch['coor2d'][:, :, :2].cpu().clone()
		reg_input[:, :, 2] = self.outputs['depth'].cpu() * index_bone_length.unsqueeze(1) + root_depth.unsqueeze(1)
		reg_input[:, :, 2] = (depthmap_max.unsqueeze(1) - reg_input[:, :, 2]) / depthmap_range.unsqueeze(1)
		reg_input = reg_input.view(reg_input.size(0), -1, 1, 1)

		self.outputs['depthmap'] = self.networks['DepthRegularizer'](reg_input.cuda()).squeeze()
		self.loss = self.criterion()

		self.outputs = to_cpu(self.outputs)

	def eval_batch_result(self):
		dis2d = self.batch_result['dis2d'].mean()
		dis3d = self.batch_result['dis3d'].mean()
		return {"dis2d": dis2d, "dis3d": dis3d, "loss": self.loss.item()}

	def get_batch_result(self, type):
		preds2d = get_preds_from_heatmap(self.outputs['heatmap'][-1])
		# preds2d = get_preds_from_heatmap(self.batch['heatmap'])
		preds3d = torch.zeros((preds2d.size(0), 21, 3))

		root_depth = self.batch['root_depth']
		index_bone_length = self.batch['index_bone_length']

		preds3d[:,:,:2] = preds2d.clone()
		preds3d[:,:,2]  = self.outputs['depth'] * index_bone_length.unsqueeze(1) + root_depth.unsqueeze(1)
		# preds3d[:,:,2]  = self.batch['relative_depth'] * index_bone_length.unsqueeze(1) + root_depth.unsqueeze(1)
		

		preds3d[:, :, :2] *= preds3d[:, :, 2:]

		for i in range(preds3d.size(0)):
			preds3d[i, :, :] = torch.matmul(preds3d[i, :, :], self.batch['matrix'][i].transpose(0,1))

		dis2d = torch.norm(self.batch['coor2d'][..., :2] - preds2d, dim = -1)
		dis3d = torch.norm(self.batch['coor3d'] - preds3d, dim = -1)
		
		
		self.batch_result = {'coor2d': preds2d, 
							 'coor3d': preds3d,
							 'dis3d' : dis3d,
							 'dis2d' : dis2d}

		return self.batch_result

	def eval_epoch_result(self):
		AUC_20_50 = calc_auc(self.epoch_result['dis3d'].view(-1), 20, 50)
		return {
			"AUC_20_50":AUC_20_50}
			
	def criterion(self):
		L1Loss = nn.L1Loss()
		loss = torch.zeros(1).cuda()
		loss += CPMMSELoss(self.outputs, self.batch)
		loss += nn.functional.smooth_l1_loss(self.outputs['depth'], self.batch['relative_depth'].cuda()) * 0.1 
		loss += L1Loss(self.outputs['depthmap'], self.batch['depthmap'].float().cuda())
		return loss

