from __future__ import absolute_import

import numpy as np
import os
import shutil
import torch 
import time
import yaml
import pickle
import warnings
from easydict import EasyDict as edict

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        val = float(val)
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0

class MetricMeter(object):
    """docstring for MetricMeter"""
    def __init__(self, metric_item):
        super(MetricMeter, self).__init__()
        self.metric = {}
        for name in metric_item:
            self.metric[name] = AverageMeter()

    def update(self, metric_update, size):
        for name in metric_update:
            if self.metric.has_key(name):
                self.metric[name].update(metric_update[name], size)
            else:
                warnings.warn("{} does not found in metric".format(name), RuntimeWarning)

    def __getitem__(self, idx):
        return self.metric[idx]

    def names(self):
        return self.metric.keys()

    def to_dict(self):
        return {name: self.metric[name].avg for name in self.metric}
def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor

def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray).float()
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray.float()

def to_cpu(outputs):
    if isinstance(outputs,dict):
        return {key: to_cpu(outputs[key]) for key in outputs}
    if isinstance(outputs,list):
        return [to_cpu(output) for output in outputs]
    if isinstance(outputs, torch.Tensor):
        return outputs.detach().cpu() if outputs.is_cuda else outputs        
    # return outputs
    raise Exception("Unrecognized type {}".format(type(output)))

def to_cuda(outputs):
    if isinstance(outputs,dict): 
        return {key: to_cuda(outputs[key]) for key in outputs}        
    if isinstance(outputs,list):
        return [to_cuda(output) for output in outputs]
    if isinstance(outputs, torch.Tensor):
        return outputs if outputs.is_cuda else outputs.cuda()

    raise Exception("Unrecognized type {}".format(type(output)))
    # return outputs

def combine(x , y):
    assert type(x) == type(y), 'combine two different type items {} and {}'.format(type(x), type(y))
    if isinstance(x, dict):
        assert x.keys() == y.keys()
        return {kx: combine(x[kx],y[kx]) for kx in x}
    if isinstance(x, list):
        assert len(x) == len(y), 'lists size does not match'
        return [combine(a,b) for a, b in zip(x, y)]
    if isinstance(x, torch.Tensor):
        return torch.cat([x,y], 0)
    if isinstance(x, np.ndarray):
        return np.concatenate([x,y], 0)
    raise Exception("Unrecognized type {}".format(type(x)))

def get_dataset_name(cfg):
    dataset_name = ','.join(cfg.CONTAINS.keys())
    return dataset_name

def get_config(fpath, type = 'train'):
    cfg = yaml.load(open(fpath))
    cfg = edict(cfg)

    tag = time.asctime(time.localtime(time.time()))
    tag = tag[4:-5] #remove day of the week and year
    tag = tag.replace("  "," ").replace(" ","_")
    if type == 'train':
        train_dataset_name = get_dataset_name(cfg.TRAIN.DATASET)
        valid_dataset_name = get_dataset_name(cfg.VALID.DATASET)
        cfg.TAG = "_".join([tag, cfg.MODEL.NAME, 'train:'+train_dataset_name, 'valid:' + valid_dataset_name])
        cfg.START_EPOCH = cfg.CURRENT_EPOCH #if resuming training
        cfg.SAVE_PATH = os.path.join(cfg.OUTPUT_DIR, cfg.TAG)
        if cfg.LOG.MONITOR_ITEM is None:
            cfg.LOG.MONITOR_ITEM = []
        for name in cfg.MODEL.OPTIMIZERS:
            cfg.LOG.MONITOR_ITEM.append(name + '_lr')
            
    if type == 'infer':
        valid_dataset_name = get_dataset_name(cfg.DATASET)
        cfg.TAG = "_".join([tag, cfg.MODEL.NAME, 'valid:' + valid_dataset_name])
        cfg.CHECKPOINT = os.path.join(cfg.OUTPUT_DIR, cfg.TAG)
        cfg.IMG_RESULT = os.path.join(cfg.CHECKPOINT, 'img_result')
    return cfg

def save_config(cfg, fpath, tag):
    cfg.RESUME_TRAIN = 1
    cfg.CHECKPOINT = os.path.join(cfg.OUTPUT_DIR, cfg.TAG, tag)
    configpath = os.path.join(fpath, tag, 'config.yml')
    yaml.dump(cfg, open(configpath,"w"))

def save_checkpoint(model, preds, cfg, log, is_best, fpath, snapshot=None):
    # preds = to_numpy(preds)
    latest_filepath = os.path.join(fpath, 'latest')

    if not os.path.exists(latest_filepath):
        os.makedirs(latest_filepath)

    log.save(latest_filepath)
    save_config(cfg, fpath, 'latest')
    model.save(latest_filepath)
    pickle.dump(preds, open(os.path.join(latest_filepath, 'preds.pickle'), 'w'))

    if snapshot and cfg.CURRENT_EPOCH % snapshot == 0:
        shutil.copytree(latest_filepath, os.path.join(fpath, str(cfg.CURRENT_EPOCH)))
        save_config(cfg, fpath, str(cfg.CURRENT_EPOCH))

    if is_best:
        best_path = os.path.join(fpath, "best")
        if os.path.exists(best_path):
            shutil.rmtree(best_path)
        shutil.copytree(latest_filepath, best_path)
        save_config(cfg, fpath, 'best')

def save_infer_result(result, metric = None, checkpoint='checkpoint', filename='preds.pickle'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    if metric is not None:
        pickle.dump(metric.to_dict(), open(os.path.join(checkpoint, 'metric.json'),"w"))
    filepath = os.path.join(checkpoint, filename)
    pickle.dump(result, open(filepath,"w"))

