#!/usr/bin/env python

import torch
import numpy as np
import sys
from terminal_print import *
from torch.autograd import Variable
from classifier import *
from format_conversion import *
import collections



def basic_optimizer(model, db, data_loader_name, loss_callback='compute_loss', epoc_loop=5000, zero_is_min=False):
	#	zero_is_min	: This means that the objective cannot go below 0. Cause an earlier exit

	optimizer = model.get_optimizer()
	avgLoss_cue = collections.deque([], 400)

	for epoch in range(epoc_loop):
		running_avg = []
		running_avg_grad = []
		for i, data in enumerate(db[data_loader_name], 0):
			[inputs, labels, indices] = data
			inputs = Variable(inputs.type(db['dataType']), requires_grad=False)
			labels = Variable(labels.type(db['dataType']), requires_grad=False)

			loss_method = getattr(model, loss_callback)
			loss = loss_method(inputs, labels, indices)
			#loss_before = loss.data.item()

			model.zero_grad()	
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			#	size of the gradient norm
			grad_norm = 0	
			for i, param in enumerate(model.parameters()):
				try: grad_norm += param.grad.data.norm()
				except: import pdb; pdb.set_trace()


			running_avg_grad.append(grad_norm)
			running_avg.append(loss.data.item())

			#loss = loss_method(inputs, labels, indices)
			#loss_after = loss.data.item()
			#print('\t\tloss before %.3f , loss after %.3f'%(loss_before, loss_after))
			#import pdb; pdb.set_trace();

		#avgLoss = np.mean(np.array(running_avg))		#/db['num_of_output']
		maxLoss = np.max(np.array(running_avg))		#/db['num_of_output']
		avgGrad = np.mean(np.array(running_avg_grad))
	
		avgLoss_cue.append(maxLoss)
		progression_slope = get_slope(avgLoss_cue)
		loss_optimization_printout(db, epoch, maxLoss, avgGrad, epoc_loop, progression_slope)

		if len(avgLoss_cue) > 300 and progression_slope >= 0: break;
		if zero_is_min: 
			if maxLoss < 0.00001: break;

	clear_current_line()
	return [maxLoss, avgGrad, progression_slope]

