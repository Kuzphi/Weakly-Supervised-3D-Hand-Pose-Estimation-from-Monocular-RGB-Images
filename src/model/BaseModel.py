from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import itertools
from functools import reduce
from src.model.utils import loss
from src.model.networks import *
from src.utils.misc import to_torch, to_numpy, to_cuda, to_cpu, combine
__all__ = ['BaseModel']
class BaseModel(object):
	"""docstring for BaseModel"""
	def __init__(self, cfg):
		super(BaseModel, self).__init__()
		self.cfg = cfg
		self.name = cfg.NAME
		self.define_network(cfg.NETWORKS)
		self.define_optimizers_and_schedulers(cfg.OPTIMIZERS)

	def define_network(self, cfg):
		
		self.networks = {}
		for name in cfg:
			print("Setting up network %s"%name)
			self.networks[name] = eval(cfg[name].TYPE)(**cfg[name])
			self.networks[name] = torch.nn.DataParallel(self.networks[name], device_ids=self.cfg.GPUS).cuda()
			if cfg[name].PRETRAINED_WEIGHT_PATH:
				print("Loading %s net's pretrained weight from %s"% (name, cfg[name].PRETRAINED_WEIGHT_PATH))
				weight = torch.load(cfg[name].PRETRAINED_WEIGHT_PATH)
				self.networks[name].load_state_dict(weight, strict = False)

	def define_optimizers_and_schedulers(self, cfg):
		# assert len(self.networks) == 1, 'undefined optimizers method'
		print("Setting up optimizer and schedulers")
		self.optimizers = {}
		self.schedulers = {}
		for name in cfg:			
			paras = [self.networks[net_name].parameters() for net_name in cfg[name].NETWORKS]
			paras = itertools.chain(*paras)
			optimizer = eval('torch.optim.' + cfg[name].TYPE)(paras,**cfg[name].PARAMETERS)
			self.optimizers[name] = optimizer

			if cfg[name].has_key('SCHEDULER'): #define schduler if exists
				scheduler_cfg = cfg[name].SCHEDULER
				scheduler = eval('torch.optim.lr_scheduler.' + scheduler_cfg.TYPE)(optimizer, **scheduler_cfg.PARAMETERS)
				scheduler.step()
				self.schedulers[name] = scheduler

	def criterion(self):
		if self.cfg.has_key('criterion'):
			return eval('loss.' + self.cfg.CRITERION)(model.batch, model.output)
		raise NotImplementedError

	def train(self):
		for name in self.networks:
			self.networks[name].train()

	def eval(self):
		for name in self.networks:
			self.networks[name].eval()

	def set_batch(self, batch):
		self.batch = batch

	def forward(self):
		assert len(self.networks) == 1, 'undefined forward method'
		# xx = to_cuda(self.batch['input'])
		# print("cuda", xx['img'].cuda().is_cuda)
		name = self.networks.keys()[0]
		self.outputs = self.networks[name](to_cuda(self.batch['input']))
		self.loss 	 = self.criterion()
		self.outputs = to_cpu(self.outputs)

	def step(self):
		self.forward()
		for optimizer in self.optimizers.values():
			optimizer.zero_grad()

		self.loss.backward()

		for optimizer in self.optimizers.values():
			optimizer.step()

	def update_learning_rate(self):
		for scheduler in self.schedulers.values():
			scheduler.step()

	# set requies_grad=Fasle to avoid computation
	def set_requires_grad(self, nets, requires_grad=False):
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad

	def save(self, path):
		path = os.path.join(path, 'model')
		if not os.path.exists(path):
			os.makedirs(path)
		for name, net in self.networks.iteritems():
			fpath = os.path.join(path, 'net_' + name + '.torch')
			torch.save(net.state_dict(), fpath)

		for name, optimizer in self.optimizers.iteritems():
			fpath = os.path.join(path, 'optimizer_' + name + '.torch')
			torch.save(optimizer.state_dict(), fpath)

		for name, scheduler in self.schedulers.iteritems():
			fpath = os.path.join(path, 'scheduler_' + name + '.torch')
			torch.save(scheduler.state_dict(), fpath)

	def resume(self, path):
		path = os.path.join(path, 'model')
		if not os.path.exists(path):
			raise IOError, '%s does not exists' % path
		for name, net in self.networks.iteritems():
			fpath = os.path.join(path, 'net_' + name + '.torch')
			net.load_state_dict(torch.load(fpath))

		for name, optimizer in self.optimizers.iteritems():
			fpath = os.path.join(path, 'optimizer_' + name + '.torch')
			optimizer.load_state_dict(torch.load(fpath))

		for name, scheduler in self.schedulers.iteritems():
			fpath = os.path.join(path, 'scheduler_' + name + '.torch')
			scheduler.load_state_dict(torch.load(fpath))

	def get_batch_result(self):
		raise NotImplementedError

	def collect_batch_result(self):
		self.collections.append(self.batch_result)

	def get_epoch_result(self):
		self.epoch_result = reduce(combine, self.collections)
		return self.epoch_result

	def define_evaluation(self):
		raise NotImplementedError
		
	def eval_batch_result(self):
		raise NotImplementedError

	def eval_epoch_result(self):
		raise NotImplementedError
