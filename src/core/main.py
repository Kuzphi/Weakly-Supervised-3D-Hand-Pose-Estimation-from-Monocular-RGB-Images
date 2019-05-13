# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import torch

from src import Bar
from src.utils.misc import AverageMeter

def train(cfg, train_loader, model, metric, log):
    data_time = AverageMeter()
    batch_time = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    bar = Bar('Processing', max=len(train_loader))
    model.collections = []
    for i, batch in enumerate(train_loader):
        # print(i)
        size = batch['weight'].size(0)
        # measure data loading time
        data_time.update(time.time() - end)

        model.set_batch(batch)
        model.step()

        # debug, print intermediate result
        if cfg.DEBUG:
            model.debug()

        #calculate the result
        model.get_batch_result(type = 'train')

        #put the result of this batch into collection for epoch result eval
        model.collect_batch_result()

        # measure accuracy and record loss
        metric_ = model.eval_batch_result()
        metric.update(metric_, size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        suffix = '({batch}/{size}) Data:{data:.1f}s Batch:{bt:.1f}s Total:{total:} ETA:{eta:} '.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.avg,
                    total=bar.elapsed_td,
                    eta=bar.eta_td)
        for name in metric_.keys():
            suffix += '{}: {:.4f} '.format(name, metric[name].avg)
        bar.suffix  = suffix
        bar.next()

    log.info(bar.suffix)
    bar.finish()

    model.get_epoch_result()
    metric_ = model.eval_epoch_result()
    metric.update(metric_, 1)

    
    print("".join(["%s : %.4f"%(key, metric_[key]) for key in metric_]))
    return model.epoch_result

def validate(cfg, val_loader, model, metric = None, log = None):
    data_time = AverageMeter()    
    batch_time = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_loader.dataset)
    model.collections = []

    idx = 0
    bar = Bar('Processing', max=len(val_loader))
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):

            data_time.update(time.time() - end)

            size = batch['weight'].size(0)

            # measure data loading time
            data_time.update(time.time() - end)
            
            # compute output
            model.set_batch(batch)
            model.forward()
            # debug, print intermediate result
            if cfg.DEBUG:
                model.debug()

            model.get_batch_result(type = 'valid')

            #put the result of this batch into collection for epoch result eval
            model.collect_batch_result()
            
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            suffix = '({batch}/{size}) Data:{data:.1f}s Batch:{bt:.1f}s Total:{total:} ETA:{eta:} '.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td)

            if cfg.IS_VALID:
                metric_ = model.eval_batch_result()
                metric.update(metric_, size)
                for name in metric_.keys():
                    suffix += '{}: {:.4f} '.format(name, metric[name].avg)

            bar.suffix  = suffix
            bar.next()

        if log:
            log.info(bar.suffix)
        bar.finish()

    model.get_epoch_result()
    metric_ = model.eval_epoch_result()
    metric.update(metric_, 1)
    print("".join(["%s : %.4f"%(key, metric_[key]) for key in metric_]))
    return model.epoch_result