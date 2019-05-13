# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Liangjian Chen(Kuzphi@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import torch
from torch.utils.data import DataLoader

from src import model
from src import dataset
from src.core import train, validate
from src.core.log import Log
from src.utils.misc import MetricMeter, get_config, save_checkpoint
def main(args):
	print("Reading configuration file")
	cfg = get_config(args.cfg)
	cfg.DEBUG = args.debug
	print("Creating Model")
	model = eval('model.' + cfg.MODEL.NAME)(cfg.MODEL)

	print("Creating Log")
	log = Log(monitor_item = cfg.LOG.MONITOR_ITEM, metric_item = cfg.METRIC_ITEMS, title = cfg.TAG)

	print("Loading Training Data")
	train_data = eval('dataset.' + cfg.TRAIN.DATASET.NAME)(cfg.TRAIN.DATASET, model.reprocess)
	print("Loading Valid Data")
	valid_data = eval('dataset.' + cfg.VALID.DATASET.NAME)(cfg.VALID.DATASET, model.reprocess)
	print ("Train Data Size: ", len(train_data))
	print ("Valid Data Size: ", len(valid_data))
	train_loader = DataLoader(
		train_data,
		batch_size=cfg.TRAIN.DATASET.BATCH_SIZE * len(cfg.MODEL.GPUS),
		shuffle=cfg.TRAIN.DATASET.SHUFFLE,
		num_workers=cfg.WORKERS)

	valid_loader = DataLoader(
		valid_data,
		batch_size=cfg.VALID.DATASET.BATCH_SIZE * len(cfg.MODEL.GPUS),
		num_workers=cfg.WORKERS)

	best = 1e99
	sgn = -1 if cfg.MAIN_METRIC.endswith('Acc') else 1

	if cfg.RESUME_TRAIN:
		print("Resuming data from checkpoint")
		log.resume(cfg.CHECKPOINT)
		model.resume(cfg.CHECKPOINT)
		best = log.log['valid_' + cfg.MAIN_METRIC][-1]

	for epoch in range(cfg.START_EPOCH, cfg.END_EPOCH):
		LR = ' '.join("{}:{}".format(k, v.param_groups[0]['lr']) for k,v in model.optimizers.items())

		print('Epoch: %d |LR %s' % (epoch, LR))

		# train for one epoch
		train_metric = MetricMeter(cfg.METRIC_ITEMS)
		train(cfg.TRAIN, train_loader, model, train_metric, log)

		# evaluate on validation set
		valid_metric = MetricMeter(cfg.METRIC_ITEMS)
		predictions = validate(cfg.VALID,valid_loader, model, valid_metric, log)

		#value of the appended logger file should be basic type for json serializing

		monitor_metric = None #add monitor item if there is one
		log.append(train = train_metric.to_dict(), valid = valid_metric.to_dict(), monitor = monitor_metric)

		# remember best acc and save checkpoint
		new_metric = valid_metric[cfg.MAIN_METRIC].avg
		is_best = sgn * new_metric < best
		best = min(best, sgn * new_metric)
		cfg.CURRENT_EPOCH = epoch
		save_checkpoint(model, predictions, cfg, log, is_best, fpath=cfg.SAVE_PATH, snapshot = 5)
		model.update_learning_rate()

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Train keypoints network')
	parser.add_argument('--cfg',
						help='configure file',
						type=str)
	parser.add_argument('-d', '--debug', dest='debug', action='store_true',
						help='show intermediate results')

	args = parser.parse_args()
	main(args)