#!/usr/bin/env python

import sys
import torch
from rbm import *
from basic_optimizer import *
from format_conversion import *
from torch.utils.data import Dataset, DataLoader


def pretrain(model, db, sparcity=False, sparcity_percentage=0.002):
	#	Read the network, for each layer, create a RBM, train it and set the weights to the value of the RBM
	tmpDB = db.copy()	
	pps = True
	tmpX = np.copy(db['train_data'].X)


	count = 0
	for layer in model.children():
		count += 1; 
		if count > model.num_of_linear_layers: break
		if pps: print('\tLayer ' , count, layer)

		hiddenDim = layer.weight.data.numpy().shape[0]
		inDim = layer.weight.data.numpy().shape[1]

		smallest_loss = 100
		best_rbm = None
		pl = 1

		for l in range(db['pretrain_repeats']):		#	Run rbm 5 times and pick the lowest cost as best_rbm
			if pps: print('\t\tCurrently running ' , l , ' out of ' , db['pretrain_repeats'], ' iteration')
			rbmLayer = rbm(inDim, hiddenDim, activation=layer.activation, sparse=sparcity)
			[avgLoss, avgGrad, progression_slope] = basic_optimizer(rbmLayer, db, data_loader_name='train_loader')

			if False: study_model_output(rbmLayer, tmpDB)
			if avgLoss < smallest_loss or type(best_rbm) == type(None): 
				smallest_loss = avgLoss
				best_rbm = rbmLayer
				pl = l

			if pps: clear_previous_line()

		if pps: print('\t\trun ' , pl, ' was chosen, with loss : ', smallest_loss, '\n')

		x = numpy2Variable(tmpDB['train_data'].X, db['dataType'])
		tmpDB['train_data'].X = (best_rbm.innerLayer_out(x)).data.numpy()
		tmpDB['data_loader'] = DataLoader(dataset=tmpDB['train_data'], batch_size=tmpDB['batch_size'], shuffle=True)

		layer.weight = best_rbm.l1.weight
		layer.bias = best_rbm.l1.bias
		clear_previous_line()

		##	debug
		#tmpX_var = numpy2Variable(tmpX, db['dataType'])
		#outX = model(tmpX_var)


	db['train_data'].X = tmpX
