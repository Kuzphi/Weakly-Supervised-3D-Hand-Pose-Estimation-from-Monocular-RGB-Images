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
from src.utils.imutils import im_to_torch, draw_heatmap
from src.utils.misc import to_torch
from src.utils.imutils import load_image

__all__ = ['CMU']
class CMU(BaseDataset):
    """docstring for CMU"""
    def __init__(self, cfg):
        super(CMU, self).__init__(cfg)

    def _get_db(self):
        self.anno = pickle.load(open(self.cfg.DATA_JSON_PATH))
        return sorted(self.anno.keys())

    def __getitem__(self, idx):
        name = self.db[idx]
        label = self.anno[name]

        image_path   = os.path.join(self.cfg.ROOT, name + '.png')
        img = load_image(image_path, mode = 'RGB') # already / 255
        coor = label['uv_coor']
        coor = np.array(coor)
        coor = to_torch(coor)
        meta = edict({'name': name})

        return {'img':img,
                'coor': to_torch(coor[:,:2]),                
                'meta': meta}


