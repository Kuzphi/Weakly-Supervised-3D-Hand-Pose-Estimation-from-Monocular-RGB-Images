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
import scipy
import torch
import matplotlib.pyplot as plt
import numpy as np

from easydict import EasyDict as edict
from mpl_toolkits.mplot3d import Axes3D

from src.dataset import InferenceDataset
from src.utils.imutils import draw_joint2d, draw_joint3d, plot_hand, plot_hand_3d
from src.utils.imutils import load_image, im_to_torch, im_to_numpy, draw_heatmap
from src.utils.misc import to_torch
from PIL import Image
from src.utils.misc import to_numpy
import io
__all__ = ['Test', 'OriginImage']
class Test(InferenceDataset):
    def __init__(self, cfg):
        super(Test, self).__init__(cfg)
        self.db = self._get_db()

    def __getitem__(self, idx):
        img_path = self.db[idx]
        img = load_image(img_path, mode = 'GBR') # already / 255
        meta = {'idx': idx}

        return { 'input': { "img": img},
                'weight': 1,
                 'meta': meta}

    def __len__(self):
        # return len(self.db['left_hand']) + len(self.db['right_hand'])
        return 100
        return len(self.db)

class OriginImage(Test):
    """docstring for origin_image"""
    def __init__(self, cfg):
        super(OriginImage, self).__init__(cfg)
        
    def __getitem__(self, idx):
        # isleft = 1 if idx < len(self.db['left_hand']) else 0

        # if isleft:
        #     img = self.db['left_hand'][idx]
        # else:
        #     img = self.db['right_hand'][idx - len(self.db['left_hand'])]
        isleft = 0
        img_path = self.db[idx]
        img = scipy.misc.imread(img_path, mode='RGB') 
        img = scipy.misc.imresize(img, (256, 256))
        # import matplotlib.pyplot as plt
        # fig = plt.figure(1)
        # ax1 = fig.add_subplot(111)
        # ax1.imshow(img)
        # plt.show()

        img = (img.astype('float') / 255.0) - 0.5
        img = im_to_torch(img)
        meta = {'isleft': isleft,
                'idx': self.db[idx]}

        return { 'input': { "img": img,
                            "hand_side": torch.tensor([isleft, 1 - isleft]).float()},
                 'meta': meta}
    def __len__(self):
        return len(self.db)