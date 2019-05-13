# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(Kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
import src.core.loss as loss

from src import model
from src import dataset
from src.core import validate
from src.utils.misc import MetricMeter, get_config, save_infer_result

def main(args):
    print("Reading configuration file")
    cfg = get_config(args.cfg, type = 'infer')

    print("Creating Model")
    model = eval('model.' + cfg.MODEL.NAME)(cfg.MODEL)

    print("Loading Training Data")
    infer_data = eval('dataset.' + cfg.DATASET.NAME)(cfg.DATASET, model.reprocess)

    infer_loader = DataLoader(
        infer_data,
        batch_size=cfg.DATASET.BATCH_SIZE * len(cfg.MODEL.GPUS),
        num_workers=cfg.WORKERS)

    print("Starting Inference")
    if cfg.IS_VALID:
        metric = MetricMeter(cfg.METRIC_ITEMS)
        preds  = validate(cfg, infer_loader, model, metric) # validate(cfg, infer_loader, model, criterion)
        save_infer_result(preds, metric, cfg.CHECKPOINT)
    else:
        preds = validate(cfg, infer_loader, model)
        save_infer_result(preds, None, cfg.CHECKPOINT)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument('--cfg',
                        help='configure file',
                        type=str)
    args = parser.parse_args()
    main(args)