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
import  matplotlib.pyplot as plt
from easydict import EasyDict as edict

from src.dataset import BaseDataset
from src.utils.imutils import im_to_torch, im_to_numpy, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image, resize, plot_hand

__all__ = ['RealHandRealBG2D']
class RealHandRealBG2D(BaseDataset):
    """docstring for TencentHand"""
    def __init__(self, cfg):
        super(RealHandRealBG2D, self).__init__()

    def _get_db(self):
        self.names = ['Model1Tap','Model1Fist', 'Model1ArmRotate', 'Model1WristRotate']
        self.db = {}
        self.anno = {}
        for name in self.names:
            self.db[name] = []
            self.anno[name] = []
        
        for name in self.names:
            ipath = 'data/TencentHand/RealHandRealBG/%s/edge2hand_newD_newG_pix2pix_newfake_20k/test_latest/images'%name
            lpath = 'data/TencentHand/Model1/%s/label/'%name
            for file in os.listdir(ipath):
                if file.endswith('fake_B.png'):
                    self.db[name].append(os.path.join(ipath, file))
                    idx = file[:7]
                    label = json.load(open(os.path.join(lpath, idx + '.json'),"r"))
                    self.anno[name].append(label['perspective'])

        # self.anno = pickle.load(open(self.cfg.DATA_JSON_PATH))
        # return sorted(self.anno.keys())
        return self.db
        
    # def transforms(self, cfg, img, coor):
    #     # resize
    #     if cfg.has_key('RESIZE'):
    #         coor[:, 0] = coor[:, 0] / img.size(1) * cfg.RESIZE
    #         coor[:, 1] = coor[:, 1] / img.size(2) * cfg.RESIZE
    #         img = resize(img, cfg.RESIZE, cfg.RESIZE)

    #     if self.is_train:
    #         # s = s*torch.randn(1).mul_(sf).add_(1).clamp(1-sf, 1+sf)[0]
    #         # r = torch.randn(1).mul_(rf).clamp(-2*rf, 2*rf)[0] if random.random() <= 0.6 else 0
            
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
        name = self.names[idx // 5220]
        path = self.db[name][idx % 5220]
        coor = self.anno[name][idx % 5220]

        img = load_image(path)

        #calculate ground truth coordination
        coor = torch.tensor(coor)
        coor[:, 0] = coor[:, 0] * img.size(1)
        coor[:, 1] = (1 - coor[:, 1]) * img.size(2)
        coor = coor[:, :2]

        #apply transforms into image and calculate cooresponding coor
        # if self.cfg.TRANSFORMS:
        #     img, coor = self.transforms(self.cfg.TRANSFORMS, img , coor)

        #openpose require 22 channel, discard the last one
        # heatmap = np.zeros((self.cfg.NUM_JOINTS, img.shape[1], img.shape[2]))
        # for i in range(self.cfg.NUM_JOINTS - 1):
        #     heatmap[i, :, :] = draw_heatmap(heatmap[i], coor[i], self.cfg.HEATMAP.SIGMA, type = self.cfg.HEATMAP.TYPE) 


        meta = edict({'name': path})

        assert coor.min() > 0, path
        
        return { 'img':img,
                 'coor': to_torch(coor),
                 'meta': meta}

    def __len__(self):
        return 5220 * 4