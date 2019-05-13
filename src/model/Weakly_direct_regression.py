from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
from torch import nn

from src.model.BaseModel import BaseModel
from src.model.utils.loss import CPMMSELoss
from src.model.utils.evaluation import get_preds_from_heatmap, calc_auc
from src.model.utils.data_reprocess import transforms
from src.utils.misc import to_torch
from src.utils.imutils import draw_heatmap
__all__ = ['Weakly_direct_regression']
class Weakly_direct_regression(BaseModel):
	"""docstring for Weakly_direct_regression"""
	def __init__(self, cfg):
		super(Weakly_direct_regression, self).__init__(cfg)
		if cfg.STAGE == 1: #train heatmap
			pass
			# self.set_requires_grad(self.networks[].module.depth)			
		if cfg.STAGE == 2: #train only depth network
			for name, net in self.networks.iteritems():
				for net_name in net.module.__dict__['_modules']:
					if 'depth' in net_name:
						continue
					self.set_requires_grad(net.module.__dict__['_modules'][net_name])
			# for param in self.network.module.stage6.parameters():
			# 		print(param.requires_grad)

		if cfg.STAGE == 3: #train both networks
			pass

	def reprocess(self, input, reprocess_cfg):
		img = input['img']
		coor2d = input['coor2d']
		matrix = input['matrix']
		meta = input['meta']
		#apply transforms into image and calculate cooresponding coor and camera instrict matrix
		if reprocess_cfg.TRANSFORMS:
			img, _, coor2d, matrix = transforms(reprocess_cfg.TRANSFORMS, img, None, coor2d, matrix)

		matrix = np.linalg.inv(matrix) #take the inversion of matrix

		coor3d = coor2d.clone()
		coor3d[:,:2] *= coor3d[:, 2:]
		coor3d = torch.matmul(coor3d, to_torch(matrix).transpose(0, 1))

		root_depth = coor2d[0, 2].clone()
		index_bone_length = torch.norm(coor3d[9,:] - coor3d[10,:])
		relative_depth = (coor2d[:,2] - root_depth) / index_bone_length

		heatmap = torch.zeros(reprocess_cfg.NUM_JOINTS, img.size(1), img.size(2))

		for i in range(21):
			heatmap[i] = draw_heatmap(heatmap[i], coor2d[i], reprocess_cfg.HEATMAP.SIGMA)


		return {'input': {'img':img
						  },
				'heatmap': heatmap,
				'matrix': to_torch(matrix),
				'color_hm': heatmap,
				'coor3d': to_torch(coor3d),
				'coor2d': to_torch(coor2d),
				'root_depth': root_depth,
				'index_bone_length': index_bone_length,
				'relative_depth': relative_depth,
				'weight': 1,
				'meta': meta,
				}


	def eval_batch_result(self):
		dis2d = self.batch_result['dis2d'].mean()
		dis3d = self.batch_result['dis3d'].mean()
		return {"dis2d": dis2d, "dis3d": dis3d, "loss": self.loss.item()}

	def get_batch_result(self, type):
		preds2d = get_preds_from_heatmap(self.outputs['heatmap'][-1])
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
		dis3d = torch.norm(self.batch['coor3d'] - preds3d, dim = -1) #* 1000
		
		
		self.batch_result = {'coor2d': preds2d, 
							 'coor3d': preds3d,
							 'dis3d' : dis3d,
							 'dis2d' : dis2d}

		return self.batch_result

	def eval_epoch_result(self):
		median = torch.median(self.epoch_result['dis3d'].view(-1))
		AUC_20_50 = calc_auc(self.epoch_result['dis3d'].view(-1), 20, 50)
		return {
			"median": median,
			"AUC_20_50":AUC_20_50}

	def criterion(self):
		self.loss = torch.zeros(1).cuda()
		if self.cfg.STAGE == 1 or self.cfg.STAGE == 3:
			self.loss  += CPMMSELoss(self.outputs, self.batch)

		if self.cfg.STAGE == 2 or self.cfg.STAGE == 3:
			# bs    = self.outputs['depthmap'].size(0)
			# index = self.batch['index'].long().cuda()
			
			# depth = self.outputs['depthmap'].view(bs, -1).gather(1, index).view(bs, 21)
			# print (self.outputs.keys())
			self.loss += nn.functional.smooth_l1_loss(self.outputs['depth'], self.batch['relative_depth'].cuda())
		return self.loss

