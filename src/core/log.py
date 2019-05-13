# A simple torch style log
from __future__ import absolute_import

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['Logger', 'LoggerMonitor', 'savefig']

def savefig(fname, dpi=None):
	dpi = 150 if dpi == None else dpi
	plt.savefig(fname, dpi=dpi)
	
def plot_overlap(logger, names=None):
	names = logger.names if names == None else names
	numbers = logger.numbers
	for _, name in enumerate(names):
		x = np.arange(len(numbers[name]))
		plt.plot(x, np.asarray(numbers[name]))
	return [logger.title + '(' + name + ')' for name in names]

class Log(object):
	'''Save training process to log file with simple plot function.'''
	def __init__(self, monitor_item, metric_item, title = None): 
		self.log = dict()
		self.title = title        
		self.log['info'] = ''

		self.metric_item = metric_item
		self.monitor_item = monitor_item
		
		for item in monitor_item:
			self.log[item] = []

		for item in metric_item:
			self.log['train_' + item] = []
			self.log['valid_' + item] = []

	def resume(self, fpath):
		fpath = os.path.join(fpath,'log.json')
		if not os.path.exists(fpath):
			raise IOError, '%s does not exist' % fpath
		print('Resuming log from checkpoint')
		self.ori_log = json.load(open(fpath))
		for item in self.monitor_item:
			assert self.log.has_key(item), "Loadding Error: {} does not exist in original log file".format(item)
			self.log[item] = self.ori_log[item]

	def append(self, train = None, valid = None, monitor = None):
		
		def update(required_keys, updates, prefix = ''):
			for key in required_keys:
				assert updates.has_key(key), "{} does not found in update".format(key)
				self.log[prefix + key].append(updates[key])

		if train is not None:
			update(self.metric_item, train, 'train_')
		if valid is not None:
			update(self.metric_item, valid, 'valid_')
		if monitor is not None:
			update(self.monitor_item, monitor)
		

	def plot(self, plot_item=None):
		if plot_item == None:
			plot_item = []
			for key in self.log.keys():
				if isinstance(self.log[key], list):
					plot_item.append(key)

		for key in plot_item:
			assert self.log.has_key(key), "log file does not contain {}".format(key)            
			assert isinstance(self.log[key], list), "unable to plot {}, whose value is not a list".format(key)
			x = range(len(self[key]))
			y = np.array(self[key]).astype(np.float)
			plt.plot(x, y)
			
		plt.legend([self.title + '(' + key + ')' for key in plot_item])
		plt.grid(True)

	def save(self, fpath):
		fpath = os.path.join(fpath, 'log.json')
		json.dump(self.log, open(fpath,"w"))

	def info(self, newline):
		self.log['info'] += newline + '\n'
		
class LogMonitor(object):
	'''Load and visualize multiple logs.'''
	def __init__ (self, paths):
		'''paths is a distionary with {name:filepath} pair'''
		self.loggers = []
		for title, path in paths.items():
			logger = Logger(path, title=title, resume=True)
			self.loggers.append(logger)

	def plot(self, names=None):
		plt.figure()
		plt.subplot(121)
		legend_text = []
		for logger in self.loggers:
			legend_text += plot_overlap(logger, names)
		plt.legend(legend_text, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
		plt.grid(True)
					
if __name__ == '__main__':
	# # Example
	# logger = Logger('test.txt')
	# logger.set_names(['Train loss', 'Valid loss','Test loss'])

	# length = 100
	# t = np.arange(length)
	# train_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
	# valid_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1
	# test_loss = np.exp(-t / 10.0) + np.random.rand(length) * 0.1

	# for i in range(0, length):
	#     logger.append([train_loss[i], valid_loss[i], test_loss[i]])
	# logger.plot()

	# Example: logger monitor
	paths = {
	'resadvnet20':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet20/log.txt', 
	'resadvnet32':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet32/log.txt',
	'resadvnet44':'/home/wyang/code/pytorch-classification/checkpoint/cifar10/resadvnet44/log.txt',
	}

	field = ['Valid Acc.']

	monitor = LoggerMonitor(paths)
	monitor.plot(names=field)
	savefig('test.eps')