# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from src.utils.misc import to_torch



# def DistanceLoss(outputs, batch):
# 	gt_coor = batch['coor3d']
# 	pred_coor = outputs['pose3d'].cpu() * batch['index_bone_length'].view(-1,1,1).repeat(1,21,3)

# 	dis = torch.norm(gt_coor - pred_coor, dim = -1)
# 	dis = torch.mean(dis) 
# 	return dis

def CPMMSELoss(outputs, batch):
	criterion = nn.MSELoss()
	loss = torch.zeros(1).cuda()
	target = batch['color_hm'].cuda()
	for pred in outputs['heatmap']:
		loss += criterion(pred, target)
	return loss

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
# def DistanceLoss2D(outputs, batch):
# 	preds = get_preds_from_heatmap(outputs['heatmap'][-1], softmax = True)
# 	diff = batch['coor'].cuda() - preds
# 	dis = torch.norm(diff, dim = -1).mean()
# 	return dis

# class JointsMSELoss(nn.Module):
#     def __init__(self, use_target_weight):
#         super(JointsMSELoss, self).__init__()
#         self.criterion = nn.MSELoss(size_average=True)
#         self.use_target_weight = use_target_weight

#     def forward(self, output, target, target_weight):
#         batch_size = output.size(0)
#         num_joints = output.size(1)
#         heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
#         heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
#         loss = 0

#         for idx in range(num_joints):
#             heatmap_pred = heatmaps_pred[idx].squeeze()
#             heatmap_gt = heatmaps_gt[idx].squeeze()
#             if self.use_target_weight:
#                 loss += 0.5 * self.criterion(
#                     heatmap_pred.mul(target_weight[:, idx]),
#                     heatmap_gt.mul(target_weight[:, idx])
#                 )
#             else:
#                 loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

#         return loss / num_joints
