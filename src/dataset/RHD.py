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
import scipy

import os
import cv2
import json
import pickle
import torch
import numpy as np
from easydict import EasyDict as edict

from src.dataset import BaseDataset
from src.utils.imutils import load_image
from src.utils.misc import to_torch
__all__  = ['RHD']
class RHD(BaseDataset):

    def __init__(self, cfg):
        super(RHD, self).__init__(cfg)

    def _get_db(self):
        self.anno = pickle.load(open(self.cfg.DATA_JSON_PATH))
        return sorted(self.anno.keys())

    def __len__(self):
        # return 100
        return len(self.db)

    def __getitem__(self, idx):
        name = self.db[idx]
        label = self.anno[name]

        image_path   = os.path.join(self.cfg.ROOT, 'color', name + '.png')
        img = load_image(image_path)# already / 255 with C * W * H
        
        depth_path = os.path.join(self.cfg.ROOT, 'depth', name + '.torch')
        depthmap = torch.load(depth_path).unsqueeze(0)

        coor2d = label['project']
        matrix = label['K']
        assert coor2d[:, :2].min() >= 0 and coor2d[:, :2].max() < 320
        meta = edict({'name': name})

        coor2d[1:,:] = coor2d[1:,:].reshape(5,4,-1)[:,::-1,:].reshape(20, -1)
        coor2d = to_torch(np.array(coor2d))
        coor2d[:,:2] = coor2d[:,:2].long().float()
        return {"img": img,
                "matrix": matrix,
                "coor2d": coor2d,
                "depthmap": depthmap,
                "meta": meta
                }

