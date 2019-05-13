from __future__ import absolute_import

import math
import matplotlib.pyplot as plt
import numpy as np
import torch

from random import randint
from src.utils.transforms import transform, transform_preds
from src.utils.misc import to_torch, to_numpy
__all__ = ['accuracy', 'AverageMeter']

def get_preds_from_heatmap(scoremaps, softmax = False, return_type = 'torch'):
    """ Performs detection per scoremap for the hands keypoints. """
    s = scoremaps.shape
    assert len(s) == 4, "This function was only designed for 4D Scoremaps(B * C * H * W)."
    # assert (s[2] < s[1]) and (s[2] < s[0]), "Probably the input is not correct, because [H, W, C] is expected."

    keypoint_coords = torch.zeros((s[0], s[1], 2))

    if softmax:
        x = scoremaps.sum(dim = 3)
        weight = torch.arange(s[2]).view(1,1,-1).expand_as(x).float()
        keypoint_coords[:,:,1] = torch.sum(x * weight, dim = 2) / x.sum(dim = 2)

        y = scoremaps.sum(dim = 2)
        weight = torch.arange(s[3]).view(1,1,-1).expand_as(y).float()
        keypoint_coords[:,:,0] = torch.sum(y * weight, dim = 2) / y.sum(dim = 2)

    else:

        for idx in range(s[0]):
            for i in range(s[1]):
                v, u = np.unravel_index(np.argmax(scoremaps[idx, i, :, :]), (s[2], s[3]))
                keypoint_coords[idx, i, 0] = u
                keypoint_coords[idx, i, 1] = v #do not know why but need reverse it !
    
    if return_type == 'numpy':
        return to_numpy(keypoint_coords[:,:21,:])

    return keypoint_coords[:,:21,:]

def get_preds(heatmap, depthmap, batch, return_type = 'torch'):
    preds_2d = get_preds_from_heatmap(heatmap)
    preds_3d = torch.zeros((preds_2d.shape[0], preds_2d.shape[1], 3))

    matrix = batch['matrix']
    root_depth = batch['root_depth']
    index_bone_length = batch['index_bone_length']

    for i in range(preds_2d.shape[0]):
        for j in range(preds_2d.shape[1]):
            pt = preds_2d[i, j].clone().long()
            preds_3d[i, j,  2] = depthmap[i, j, pt[1], pt[0]] * index_bone_length[i] + root_depth[i]
            preds_3d[i, j, :2] = preds_2d[i, j].clone()
            
    preds_3d[:,:,:2] *=preds_3d[:,:,2:]
    for i in range(preds_3d.shape[0]):
        preds_3d[i, :, :] = torch.matmul(preds_3d[i, :, :], matrix[i].transpose(0, 1))

    if return_type == 'numpy':
        return to_numpy(preds_2d), to_numpy(preds_3d)

    return preds_2d, preds_3d

def calc_dists(preds, target):
    preds = preds.float()
    target = target.float()
    dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if target[n,c,0] > 1 and target[n, c, 1] > 1:
                dists[c, n] = torch.dist(preds[n,c,:], target[n,c,:])
            else:
                dists[c, n] = -1
    return dists

def dist_acc(dists, thr=0.5):
    ''' Return percentage below threshold while ignoring values with a -1 '''
    if dists.ne(-1).sum() > 0:
        return dists.le(thr).eq(dists.ne(-1)).sum()*1.0 / dists.ne(-1).sum()
    else:
        return -1

def avg_dist(dists):
    if dists.ne(-1).sum() > 0:
        return (dists * dists.ne(-1).float()).sum()*1.0 / dists.ne(-1).sum().float()
    else:
        return -1

def eval_result(output, batch, num_joints, thr=0.5):
    ''' Calculate accuracy according to PCK, but uses ground truth heatmap rather than x,y locations
        First value to be returned is average accuracy across 'idxs', followed by individual accuracies
    '''
    preds   = get_preds(output)
    gts     = batch['coor']
    norm    = torch.ones(preds.size(0)) * output.size(3) / 10
    dists   = calc_dists(preds, gts[:,:,:2])
    normed_dists = torch.zeros(preds.size(1), preds.size(0))
    for n in range(preds.size(0)):
        for c in range(preds.size(1)):
            if dists[c, n] >= 0:
                normed_dists[c,n] = dists[c, n] / norm[n]
            else:
                normed_dists[c,n] = -1
                
    acc = torch.zeros(num_joints+1)
    dis = torch.zeros(num_joints+1)
    avg_acc = 0
    avg_dis = 0
    cnt = 0

    for i in range(num_joints):
        acc[i+1] = dist_acc(normed_dists[i])
        dis[i+1] = avg_dist(dists[i])
        if acc[i+1] >= 0: 
            avg_dis = avg_dis + dis[i+1]
            avg_acc = avg_acc + acc[i+1]
            cnt += 1
            
    if cnt != 0:  
        acc[0] = avg_acc / cnt
        dis[0] = avg_dis / cnt
    return acc, dis

def final_preds(output, center, scale, res):
    coords = get_preds(output) # float type

    # pose-processing
    for n in range(coords.size(0)):
        for p in range(coords.size(1)):
            hm = output[n][p]
            px = int(math.floor(coords[n][p][0]))
            py = int(math.floor(coords[n][p][1]))
            if px > 1 and px < res[0] and py > 1 and py < res[1]:
                diff = torch.Tensor([hm[py - 1][px] - hm[py - 1][px - 2], hm[py][px - 1]-hm[py - 2][px - 1]])
                coords[n][p] += diff.sign() * .25
    coords += 0.5
    preds = coords.clone()

    # Transform back
    for i in range(coords.size(0)):
        preds[i] = transform_preds(coords[i], center[i], scale[i], res)

    if preds.dim() < 3:
        preds = preds.view(1, preds.size())

    return preds

def AUC(dist):
    x = np.array(sorted(list(dist)))
    y = np.array([1. * (i + 1)/ len(x)  for i in range(len(x))])
    return (x, y)

def calc_auc(dist, limx = -1, limy = 1e99):
    """ Given x and y values it calculates the approx. integral and normalizes it: area under curve"""
    x, y = AUC(dist)
    # x = x * 1000 #meter to million meter
    # print (x.min(), x.max())
    l, r = 0, 0
    for i in range(len(x)):
        if limx > x[i]:
            l = i + 1
        if limy > x[i]:
            r = i + 1
    # print("l: ",l, x[l], "r: ", r, x[r - 1])
    tx = x[l:r]
    ty = y[l:r]    
    integral = np.trapz(ty, tx)
    norm = np.trapz(np.ones_like(ty), tx)
    # norm = x[-1] - x[]
    # if limy < 1e98 and limy > x[-1]:
    #     add = limy - x[-1]
    #     integral += add
    #     norm += add
    # print(integral , norm)
    return integral / norm