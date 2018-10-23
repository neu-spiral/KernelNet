#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import time 

class MLP_autoencoder(torch.nn.Module):
	def __init__(self, db, add_decoder=True, learning_rate=0.001):
		super(MLP_autoencoder, self).__init__()
		self.db = db
		self.add_decoder = add_decoder
		self.learning_rate = learning_rate
		self.input_size = db['net_input_size']
		self.output_dim = db['net_input_size']
		self.mlp_width = db['mlp_width']
		self.net_depth = db['net_depth']
		self.dataType = db['dataType']
		in_out_list = []


		for l in range(1, self.net_depth+1):
			if l == self.net_depth:
				in_out_list.append((self.output_dim ,self.mlp_width))
				lr = 'self.l' + str(l) + ' = torch.nn.Linear(' + str(self.mlp_width) + ', ' + str(self.output_dim) + ' , bias=True)'
				exec(lr)
				exec('self.l' + str(l) + '.activation = "none"')		#softmax, relu, tanh, sigmoid, none
			elif l == 1:
				in_out_list.append((self.mlp_width ,self.input_size))
				lr = 'self.l' + str(l) + ' = torch.nn.Linear(' + str(self.input_size) + ', ' + str(self.mlp_width) + ' , bias=True)'
				exec(lr)
				exec('self.l' + str(l) + '.activation = "relu"')		#softmax, relu, tanh, sigmoid, none
			else:
				in_out_list.append((self.mlp_width, self.mlp_width))
				lr = 'self.l' + str(l) + ' = torch.nn.Linear(' + str(self.mlp_width) + ', ' + str(self.mlp_width) + ' , bias=True)'
				exec(lr)
				exec('self.l' + str(l) + '.activation = "relu"')


		if add_decoder:
			in_out_list.reverse()
	
			for l, item in enumerate(in_out_list):
				c = l + self.net_depth + 1
				if c == self.net_depth*2:
					lr = 'self.l' + str(c) + ' = torch.nn.Linear(' + str(item[0]) + ', ' + str(item[1]) + ' , bias=True)'
					exec(lr)
					exec('self.l' + str(c) + '.activation = "none"')		#softmax, relu, tanh, sigmoid, none
				else:
					lr = 'self.l' + str(c) + ' = torch.nn.Linear(' + str(item[0]) + ', ' + str(item[1]) + ' , bias=True)'
					exec(lr)
					exec('self.l' + str(c) + '.activation = "relu"')


		self.initialize_network()
		self.output_network()

	def output_network(self):
		print('\tConstructing Kernel Net')
		for i in self.children():
			try:
				print('\t\t%s , %s'%(i,i.activation))
			except:
				print('\t\t%s '%(i))


	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

	def mse_loss(self, x, y):
		return ((x-y)**2).mean()
	    #return torch.sum((x - y) ** 2)
		

	def initialize_network(self):
		db = self.db

		for param in self.parameters():
			if(len(param.data.numpy().shape)) > 1:
				torch.nn.init.kaiming_normal_(param.data , a=0, mode='fan_in')	
			else:
				pass
				#param.data = torch.zeros(param.data.size())

		self.num_of_linear_layers = 0
		for m in self.children():
			if type(m) == torch.nn.Linear:
				self.num_of_linear_layers += 1


		#	If using L21 regularizer
		#hsic_cost = HSIC_AE_objective(self, db)
		#each_L1 = np.sum(np.abs(z), axis=1)
		#L12_norm = np.sqrt(np.sum(each_L1*each_L1))
		#db['λ_0_ratio'] = np.abs(hsic_cost/L12_norm)
		#db['λ'] = float(db['λ_ratio'] * db['λ_0_ratio'])



	def gaussian_kernel(self, x, σ):			#Each row is a sample
		bs = x.shape[0]
		K = self.db['dataType'](bs, bs)
		K = Variable(K.type(self.db['dataType']), requires_grad=False)		

		for i in range(bs):
			dif = x[i,:] - x
			K[i,:] = torch.exp(-torch.sum(dif*dif, dim=1)/(2*σ*σ))

		return K

	def forward(self, y0):
		for m, layer in enumerate(self.children(),1):
			if m == self.net_depth*2:
				cmd = 'self.y_pred = self.l' + str(m) + '(y' + str(m-1) + ')'
				exec(cmd)
				break;
			elif m == self.net_depth:
				if self.add_decoder:
					var = 'y' + str(m)
					cmd = var + ' = self.l' + str(m) + '(y' + str(m-1) + ')'
					exec(cmd)
				else:
					cmd = 'self.y_pred = self.l' + str(m) + '(y' + str(m-1) + ')'
					exec(cmd)
					return [self.y_pred, self.y_pred]

			else:
				var = 'y' + str(m)
				cmd = var + ' = F.relu(self.l' + str(m) + '(y' + str(m-1) + '))'
				#cmd2 = var + '= F.dropout(' + var + ', training=self.training)'
				exec(cmd)
				#exec(cmd2)

		exec('self.fx = y' + str(self.net_depth))
		return [self.y_pred, self.fx]

