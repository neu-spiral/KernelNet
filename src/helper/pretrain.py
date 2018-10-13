#!/usr/bin/env python

import sys
import torch
from rbm import *
from basic_optimizer import *
from format_conversion import *
from torch.utils.data import Dataset, DataLoader


def pretrain(model, db, loader_id, sparcity=False, sparcity_percentage=0.002):
	def quick_error_check(model, db, data_loader_name):
		X = numpy2Variable(db[data_loader_name].dataset.X, db['dataType'])
		x_out = model(X)
		return (torch.norm(X-x_out).item())


	#	Read the network, for each layer, create a RBM, train it and set the weights to the value of the RBM
	tmpDB = db.copy()	
	pps = True
	tmpX = np.copy(db[loader_id].dataset.X)


	count = 0
	for layer in model.children():
		count += 1; 
		if count > model.num_of_linear_layers: break
		if pps: print('\tLayer ' , count, layer)

		hiddenDim = layer.weight.shape[0]
		inDim = layer.weight.shape[1]

		smallest_loss = 100
		best_rbm = None
		pl = 1

		for l in range(db['pretrain_repeats']):		#	Run rbm 5 times and pick the lowest cost as best_rbm
			if pps: print('\t\tCurrently running ' , l , ' out of ' , db['pretrain_repeats'], ' iteration')

			if(db['cuda']): rbmLayer = rbm(inDim, hiddenDim, activation=layer.activation, sparse=sparcity).cuda()
			else: rbmLayer = rbm(inDim, hiddenDim, activation=layer.activation, sparse=sparcity)

			error_before = quick_error_check(rbmLayer, db, loader_id)
			[avgLoss, avgGrad, progression_slope] = basic_optimizer(rbmLayer, db, data_loader_name=loader_id)
			error_after = quick_error_check(rbmLayer, db, loader_id)

			if False: study_model_output(rbmLayer, tmpDB)
			if avgLoss < smallest_loss or type(best_rbm) == type(None): 
				smallest_loss = avgLoss
				best_rbm = rbmLayer
				pl = l

			if pps: clear_previous_line()

		if pps: print('\t\trun ' , pl, ' was chosen, with loss : %.3f,  error before : %.3f, error after : %.3f'%(smallest_loss, error_before, error_after), '\n')

		x = numpy2Variable(tmpDB['train_data'].X, db['dataType'])

		tmpDB['train_data'].X = ensure_matrix_is_numpy(best_rbm.innerLayer_out(x))
		tmpDB['data_loader'] = DataLoader(dataset=tmpDB['train_data'], batch_size=tmpDB['batch_size'], shuffle=True)

		layer.weight = best_rbm.l1.weight
		layer.bias = best_rbm.l1.bias
		clear_previous_line()

		##	debug
		#tmpX_var = numpy2Variable(tmpX, db['dataType'])
		#outX = model(tmpX_var)


	db['train_data'].X = tmpX
