#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
import numpy.matlib
from kernel_lib import *
from distances import *
from format_conversion import *
import torch.nn.functional as F
import time 
import scipy.linalg

class identity_net(torch.nn.Module):
	def __init__(self, db, add_decoder=True, learning_rate=0.001):
		super(identity_net, self).__init__()

		self.db = db
		self.add_decoder = add_decoder
		self.learning_rate = learning_rate
		self.input_size = db['net_input_size']
		self.output_dim = db['net_input_size']
		self.net_depth = db['kernel_net_depth']
		self.width_scale = db['width_scale']		#	1 is double of input size, 2 is 4 times
		self.net_width = 2*db['net_input_size']*self.width_scale
		

		self.l0 = torch.nn.Linear(self.input_size, self.net_width , bias=True)
		for l in range(self.net_depth-1):
			lr = 'self.l' + str(l+1) + ' = torch.nn.Linear(' + str(self.net_width) + ', ' + str(self.net_width) + ' , bias=True)'
			exec(lr)

		lr = 'self.l' + str(self.net_depth) + ' = torch.nn.Linear(' + str(self.net_width) + ', ' + str(self.input_size) + ' , bias=True)'
		exec(lr)
		exec('self.l' + str(self.net_depth) + '.activation = "none"')		#softmax, relu, tanh, sigmoid, none


		if add_decoder:
			lr = 'self.l' + str(self.net_depth+1) + ' = torch.nn.Linear(' + str(self.input_size) + ', ' + str(self.net_width) + ' , bias=True)'
			exec(lr)

			for l in range(self.net_depth-1):
				lr = 'self.l' + str(self.net_depth+2+l) + ' = torch.nn.Linear(' + str(self.net_width) + ', ' + str(self.net_width) + ' , bias=True)'
				exec(lr)
	

			lr = 'self.l' + str(2*self.net_depth+1) + ' = torch.nn.Linear(' + str(self.net_width) + ', ' + str(self.input_size) + ' , bias=True)'
			exec(lr)
			exec('self.l' + str(2*self.net_depth+1) + '.activation = "none"')		#softmax, relu, tanh, sigmoid, none


		self.output_network()
		self.initialize_network()

	def output_network(self):
		print('\tDimension Reduction Network')
		for i in self.children():
			try:
				print('\t\t%s , %s'%(i,i.activation))
			except:
				print('\t\t%s '%(i))


	def initialize_variables(self, db):
		self.db = db
			
		[x_hat, φ_x] = self.forward(db['train_data'].X_Var)
		φ_x = ensure_matrix_is_numpy(φ_x)
		self.φ_x_mpd = float(median_of_pairwise_distance(φ_x))
		self.σ = float(self.φ_x_mpd*db['σ_ratio'])

		N = db['train_data'].N
		self.H = np.eye(N) - (1.0/N)*np.ones((N, N))
		self.mlp_width = db['mlp_width']

	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

	def mse_loss(self, x, y):
	    return torch.sum((x - y) ** 2)
		
	def get_current_state(self, db, in_x):
		[x_hat, φ_x] = self.forward(in_x)
		φ_x = ensure_matrix_is_numpy(φ_x)

		[DKxD, Dinv] = normalized_rbk_sklearn(φ_x, self.σ)
		HDKxDH = center_matrix(db, DKxD)
		[U, U_normalized] = L_to_U(db, HDKxDH)
		Ku = U.dot(U.T)

		current_hsic = -float(np.sum(HDKxDH*Ku))
		current_AE_loss = float(ensure_matrix_is_numpy(self.autoencoder_loss(in_x, None, None)))

		#db['λ'] = float(db["λ_ratio"]*db['λ_obj_ratio'])
		#current_loss = float(current_hsic + db['λ']*current_AE_loss)

		if 'λ_obj_ratio' not in db:
			db['λ_obj_ratio'] = float(np.abs(current_hsic/current_AE_loss))

		db['λ'] = float(db["λ_ratio"]*db['λ_obj_ratio'])
		current_loss = float(current_hsic + db['λ']*current_AE_loss)

		return [current_loss, current_hsic, current_AE_loss, φ_x, U, U_normalized]

	def autoencoder_loss(self, x, label, indices):
		db = self.db
		[x_hat, φ_x] = self.forward(x)
		auto_cost = self.mse_loss(x_hat, x)

		return auto_cost

	def set_Y(self, Y):
		self.Y = Y

	def compute_loss(self, x, label, indices):
		db = self.db
		[x_hat, φ_x] = self.forward(x)
		Kx = self.gaussian_kernel(φ_x, self.σ)

		PP = self.Y[indices, :]
		Ysmall = PP[:, indices]
		obj_loss = -torch.sum(Kx*Ysmall)
		AE_loss = self.mse_loss(x_hat, x)

		#loss2 = self.mse_loss(φ_x, x)
		#loss = obj_loss + db['λ']*(AE_loss+loss2)

		loss = obj_loss + db['λ']*(AE_loss)
		return loss

	def initialize_network(self):
		db = self.db
		self.pretrain_time = 0
		self.end2end_time = 0

		atom = np.array([[1],[-1]])
		col = np.matlib.repmat(atom, self.width_scale, 1)
		z = np.zeros(((self.input_size-1)*2*self.width_scale, 1))
		one_column = np.vstack((col, z))
		original_column = np.copy(one_column)

		eyeMatrix = torch.eye(self.net_width)
		eyeMatrix = Variable(eyeMatrix.type(self.db['dataType']), requires_grad=False)		

		for i in range(self.input_size-1):
			one_column = np.roll(one_column, 2*self.width_scale)
			original_column = np.hstack((original_column, one_column))

		original_column = torch.tensor(original_column)
		original_column = Variable(original_column.type(self.db['dataType']), requires_grad=False)		


		for i, param in enumerate(self.parameters()):
			if len(param.data.shape) == 1:
				param.data = torch.zeros(param.data.size())
			else:
				if param.data.shape[1] == self.input_size:
					param.data = (1.0/self.width_scale)*original_column
				elif param.data.shape[0] == self.input_size:
					param.data = original_column.t()
				else:
					param.data = eyeMatrix

		#for i, param in enumerate(self.parameters()):
		#	print(param.data)


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
		self.y0 = y0

		for m, layer in enumerate(self.children(),0):
			if m == self.net_depth:
				cmd = 'self.y' + str(m+1) + ' = self.fx = self.l' + str(m) + '(self.y' + str(m) + ')'
				exec(cmd)
				#print(m , layer)
				#print(cmd)
				#import pdb; pdb.set_trace()
			elif m == self.net_depth*2+1:
				cmd = 'self.y' + str(m+1) + ' = self.y_pred = self.l' + str(m) + '(self.y' + str(m) + ')'
				exec(cmd)
				#print(m , layer)
				#print(cmd)
				#import pdb; pdb.set_trace()
			else:
				cmd = 'self.y' + str(m+1) + ' = F.relu(self.l' + str(m) + '(self.y' + str(m) + '))'
				exec(cmd)
				#print(m , layer)
				#print(cmd)
				#import pdb; pdb.set_trace()

		#self.fx	#this is the output of AE
		return [self.y_pred, self.fx]


