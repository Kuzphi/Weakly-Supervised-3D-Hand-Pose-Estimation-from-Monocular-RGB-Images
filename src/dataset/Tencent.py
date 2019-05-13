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
import torch
import numpy as np
from easydict import EasyDict as edict

from src.dataset import BaseDataset
from src.utils.imutils import im_to_torch, load_image
from src.utils.misc import to_torch
__all__ =['TencentHand']
class TencentHand(BaseDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(TencentHand, self).__init__(cfg)

    def _get_db(self):
        return json.load(open(self.cfg.DATA_JSON_PATH))
        
    # def transforms(self, cfg, img, coor):
    #     # resize
    #     if cfg.has_key('RESIZE'):
    #         coor[:, 0] = coor[:, 0] / img.size(1) * cfg.RESIZE
    #         coor[:, 1] = coor[:, 1] / img.size(2) * cfg.RESIZE
    #         img = resize(img, cfg.RESIZE, cfg.RESIZE)

    #     if self.is_train:

    #         # Flip
    #         if cfg.FLIP and random.random() <= 0.5:
    #             img = torch.flip(img, dims = [1])
    #             coor[:, 1] = img.size(1) - coor[:, 1]

    #         # Color 
    #         if cfg.COLOR_NORISE:
    #             img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
    #             img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)
    #             img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(-0.5, 0.5)

    #     return img, coor
    def __getitem__(self, idx):
        w = self.db[idx]

        image_path   = os.path.join(self.cfg.ROOT, w[1], w[1] + w[2], 'image', w[0] + '.png')
        label_path = os.path.join(self.cfg.ROOT, w[1], w[1] + w[2], 'label', w[0] + '.json')

        img = load_image(image_path, mode = 'GBR') # C * H * W
        
        label = json.load(open(label_path))

        #calculate ground truth coordination
        coor = np.array(label['perspective'])
        coor[:, 0] = coor[:, 0] * img.size(1)
        coor[:, 1] = (1 - coor[:, 1]) * img.size(2)
        coor = coor[:, :2]

        #apply transforms into image and calculate cooresponding coor
        if self.cfg.TRANSFORMS:
            img, coor = self.transforms(self.cfg.TRANSFORMS, img , coor)

        #openpose require 22 channel, discard the last one
        # heatmap = np.zeros((self.cfg.NUM_JOINTS, img.shape[1], img.shape[2]))
        # for i in range(self.cfg.NUM_JOINTS - 1):
        #     heatmap[i, :, :] = draw_heatmap(heatmap[i], coor[i], self.cfg.HEATMAP.SIGMA, type = self.cfg.HEATMAP.TYPE) 


        meta = edict({'name': w[1] + ' ' + w[2] + ' ' + w[0]})

        assert coor.min() > 0, label_path
        
        return { 'img':img,
                 'coor': to_torch(coor),
                 'meta': meta}

    def __len__(self):
        return 20000