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
__all__ = ['MultiView']

from src.dataset import RHD
from src.dataset import STB

class MultiView(BaseDataset):
	"""docstring for MultiView"""
	def __init__(self, cfg):
		super(MultiView, self).__init__(cfg)
		# self.depth_daset = RHD(cfg.DEPTH_CFG)		
	def _get_db(self):
		self.db = []
		for idx in self.cfg.PICK:			
			self.db += [str(idx) + '/' + x[:-4] for x in os.listdir('data/multiview/2dCrop/%s/image' % idx)]
		self.all = len(self.db)
		self.depth_db = os.listdir('data/RHD/cropped/training/depth')
		return self.db
	
	def __len__(self):
		# return 100
		return self.all
	def __getitem__(self, idx):
		tok = self.db[idx].split('/')

		image_path   = os.path.join(self.cfg.ROOT, tok[0], 'image', tok[1] + '.jpg')
		img = load_image(image_path, mode = 'RGB')
		label_path = os.path.join(self.cfg.ROOT, tok[0], 'annotation', tok[1] + '.torch')
		label = torch.load(label_path)
		depth_path = 'data/RHD/cropped/training/depth/' + self.depth_db[random.randint(0, len(self.depth_db) - 1)]
		depthmap = torch.load(depth_path).unsqueeze(0)
		meta = edict({'name': self.db[idx]})
		return {'img':img,
				'depthmap': depthmap,
				'matrix': to_torch(label['matrix']).float(),
				'coor2d': label['coor2d'].float(),
				'meta': meta,
				}
	