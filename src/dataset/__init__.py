# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from abc import abstractmethod
__all__ = ['JointsDataset', 'InferenceDataset']

class BaseDataset(object):

    def __init__(self, cfg):
        self.cfg = cfg
        self.is_train = cfg.IS_TRAIN
        self.db = self._get_db()

    def __len__(self):
        return len(self.db)

    @abstractmethod
    def _get_db(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

class JointsDataset(BaseDataset):
    """docstring for JointsDataset"""
    def __init__(self, cfg, reprocess):
        super(JointsDataset, self).__init__(cfg)
        self.datasets = []
        self.reprocess = reprocess
        self.len = 0
        self.cfg = cfg
        for key in cfg.CONTAINS:
            cfg.CONTAINS[key].REPROCESS = cfg.REPROCESS
            cfg.CONTAINS[key].IS_TRAIN = cfg.IS_TRAIN
            self.datasets.append( eval(key)(cfg.CONTAINS[key]))
            self.len += len(self.datasets[-1])            

    def __len__(self):
        # return 10
        return self.len

    def _get_db(self):
        pass

    def __getitem__(self, idx):
        for name, dataset in zip(self.cfg.CONTAINS, self.datasets):
            if idx < len(dataset):
                return self.reprocess(dataset[idx], self.cfg.CONTAINS[name].REPROCESS)
            idx -= len(dataset)

class InferenceDataset(BaseDataset):
    """docstring for InferenceDataset"""
    def __init__(self, cfg):
        super(InferenceDataset, self).__init__()
        self.data_form = cfg.DATA_FORM
        self.path = cfg.DATA_PATH
        self.db = self._get_db()

    def _get_db(self):
        if self.data_form == 'img_root':
            result = []
            for img in os.listdir(self.path):
                if img.endswith('jpg') or img.endswith('png'):
                    result.append(os.path.join(self.path,img))
            return sorted(result)
        elif self.data_form == 'block_file':
            if self.path.endswith('pickle'):
                import pickle
                img = pickle.load(open(self.path,'r'))

            elif self.path.endswith('h5'):
                import h5py                
                img = h5py.File(self.path) 
            else:
                raise RuntimeError('can not load {}'.format(self.path))
            return img
        raise RuntimeError('could not recognize {}'.format(self.data_form))

    def __len__(self,):
        return len(self.db)

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    @abstractmethod        
    def get_preds(outputs):
        pass 

    @abstractmethod
    def eval_result(self, outputs, batch, cfg = None,  **kwargs):
        #should be same as self.cfg.metric item
        pass

import os
for name in os.listdir(os.path.dirname(__file__)):
    if name[0] == '.' or name == '__init__.py' or name[-3:] != '.py':
        continue

    module = __import__(name[:-3], locals(), globals(), ['*'])
    for key in module.__all__:
        locals()[key] = getattr(module, key)
