# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random

import os
import cv2
import json
import pickle
import torch
import numpy as np
import scipy.io as sio
from easydict import EasyDict as edict

from src.dataset import JointsDataset, BaseDataset
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image, resize, im_to_numpy
from src.model.utils.evaluation import get_preds_from_heatmap
from src.model.utils.evaluation import calc_auc, AUC, calc_auc
__all__ = ['STB']
# class STB(BaseDataset):
#     """docstring for TencentHand"""
#     def __init__(self, cfg):
#         super(STB, self).__init__(cfg)
#         self.upsampler = torch.nn.Upsample(scale_factor = 8, mode = 'bilinear', align_corners = True)
#     def _get_db(self):
#         self.db = []
#         self.name = []
#         self.all = 0
#         for name in sorted(os.listdir(self.cfg.DATA_JSON_PATH)):
#             if name[:-7] in self.cfg.PICK:
#                 matPath = os.path.join(self.cfg.DATA_JSON_PATH, name)
#                 self.db.append(sio.loadmat(matPath)['handPara'])
#                 self.all += 1500
#                 self.name.append(name[:-4])
#         print(self.all)
#         return self.db
	
#     def __len__(self):
#         return self.all

#     def transforms(self, cfg, img, coor):

#         if cfg.RESIZE:
#             img = resize(img, cfg.RESIZE, cfg.RESIZE)

#         if self.is_train:
			
#             # Color
#             if cfg.COLOR_NORISE:
#                 img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
#                 img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
#                 img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

#         return img, coor

#     def __getitem__(self, idx):

#         name = self.name[idx // 1500]
#         coor = self.db[idx // 1500][:,:,idx % 1500]

#         coor = coor.transpose(1,0) / 1000. #milionmeter to meter
#         coor[1:, ] = coor[1:, :].reshape(5,4,-1)[::-1,::-1,:].reshape(20, -1)
#         coor = np.array(coor)

#         coor = to_torch(coor)
#         index_bone_length = torch.norm(coor[12,:] - coor[11,:])
#         coor = coor - coor[:1,:].repeat(21,1)

#         name = name.split("_")
#         if name[1] == 'BB':
#             image_path   = os.path.join(self.cfg.ROOT, name[0], "_".join(['left', str(idx % 1500)]) + '.png')
#         elif name[1] == 'SK':
#             image_path   = os.path.join(self.cfg.ROOT, name[0], "_".join(['SK', str(idx % 1500)]) + '.png')
#         else:
#             raise Exception("Unrecognized name {}".format(name))

#         img = load_image(image_path)

#         #apply transforms into image and calculate cooresponding coor
#         if self.cfg.TRANSFORMS:
#             img, coor = self.transforms(self.cfg.TRANSFORMS, img , coor)

#         meta = edict({'name': name})
#         isleft = 1

#         return {'input': {'img':img, 
#                           'hand_side': torch.tensor([isleft, 1 - isleft]).float(),
#                           # 'heatmap': heatmap
#                           },
#                 "index_bone_length": index_bone_length,
#                 'coor': coor, 
#                 'weight': 1,
#                 'meta': meta}
class STB(BaseDataset):
	"""docstring for STB"""
	def __init__(self, cfg):
		super(STB, self).__init__(cfg)

	def _get_db(self):
		self.name = self.cfg.PICK
		self.db = pickle.load(open(self.cfg.DATA_JSON_PATH))
		self.all = 1500 * len(self.cfg.PICK)
		return self.db
	
	def __len__(self):
		# return 100
		return self.all
	def __getitem__(self, idx):
		name = self.name[idx // 1500]
		coor2d = self.db[name]['sk']['coor2d'][idx % 1500,:,:]
		matrix = self.db[name]['sk']['matrix'][idx % 1500]

		name = name.split('_')
		image_path   = os.path.join(self.cfg.ROOT, name[0], 'SK_' + str(idx % 1500) + '.png')
		img = load_image(image_path, mode = 'RGB')
		depth_path = os.path.join(self.cfg.ROOT, name[0], 'SK_depth_{}.pickle'.format(idx % 1500))
		depthmap = pickle.load(open(depth_path)).unsqueeze(0)

		# coor2d = label['project']
		# assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 320
		

		# coor2d = torch.matmul(coor3d, to_torch(matrix).transpose(0, 1))
		meta = edict({'name': name})
		return {'img':img,
				'depthmap': depthmap,
				# 'index': index, 
				'matrix': to_torch(matrix),
				'coor2d': to_torch(coor2d),
				'meta': meta,
				}

	