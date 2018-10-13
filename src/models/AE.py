#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
from kernel_lib import *
from autoencoder import *
from format_conversion import *
from distances import *
from RFF import *
import torch.nn.functional as F
import time 

class AE(autoencoder):
	def __init__(self, db, add_decoder=True, learning_rate=0.001):
		super(AE, self).__init__(db, add_decoder, learning_rate)


	def initialize_variables(self):
		db = self.db
		X = numpy2Variable(db['train_data'].X, db['dataType'])
			
		[x_hat, φ_x] = self.forward(X)
		φ_x = ensure_matrix_is_numpy(φ_x)
		db['φ_x_mpd'] = float(median_of_pairwise_distance(φ_x))

		N = db['train_data'].N
		self.H = np.eye(N) - (1.0/N)*np.ones((N, N))

	def set_Y(self, Y):
		self.Y = Y

	def get_current_state(self):
		db = self.db
		X = numpy2Variable(db['train_data'].X, db['dataType'])
			
		[x_hat, φ_x] = self.forward(X)
		φ_x = ensure_matrix_is_numpy(φ_x)

		[DKxD, db['D_inv']] = normalized_rbk_sklearn(φ_x, db['φ_x_mpd'])
		HDKxDH = center_matrix(db, DKxD)
		Ku = db['U'].dot(db['U'].T)
		current_hsic = -np.sum(HDKxDH*Ku)
		current_AE_loss = ensure_matrix_is_numpy(self.autoencoder_loss(X, None, None))

		if 'λ_obj_ratio' not in db:
			db['λ_obj_ratio'] = np.abs(current_hsic/current_AE_loss)

		db['λ'] = db["λ_ratio"]*db['λ_obj_ratio']
		current_loss = current_hsic + db['λ']*current_AE_loss

		print('\tCurrent obj loss : %.5f from %.5f +  %.3f[%.5f]'%(current_loss, current_hsic, db['λ'], current_AE_loss))
		return [current_loss, current_hsic, current_AE_loss]

	def autoencoder_loss(self, x, label, indices):
		db = self.db
		[x_hat, φ_x] = self.forward(x)
		auto_cost = self.mse_loss(x_hat, x)
		
		return auto_cost

	def compute_loss(self, x, label, indices):
		db = self.db
		[x_hat, φ_x] = self.forward(x)


		[x_hat1, z] = db['dim_reducer_obj'](x)
		expanded_z = torch.mm(z, self.dim_expansion_matrix)
		[z_hat, φ_z] = self.forward(expanded_z)


		#	using RFF
		Kx = self.rff.get_rbf(φ_x, db['φ_x_mpd'], True, db['dataType'])
		Kz = self.rff.get_rbf(φ_z, db['φ_z_mpd'], True, db['dataType'])

		
		#Kx = self.gaussian_kernel(φ_x, db['φ_x_mpd'])
		#Kz = self.gaussian_kernel(φ_z, db['φ_z_mpd'])
		#print(Kx[0:5,0:5])
		#import pdb; pdb.set_trace()

		KxH = torch.mm(Kx, self.H)
		KzH = torch.mm(Kz, self.H)

		hsic_cost = -torch.sum(KxH*KzH)

		##	adding autoencoder regularizer
		#auto_cost = self.mse_loss(self.x_hat, x)
		#loss = hsic_cost + db['λ']*auto_cost
		return hsic_cost




