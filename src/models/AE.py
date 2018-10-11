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

		self.initialize()

	def initialize(self):
		db = self.db
		X = numpy2Variable(db['train_data'].X, db['dataType'])
		[x_hat, φ_x] = self.forward(X)
		db['φ_x_mpd'] = float(median_of_pairwise_distance(φ_x.data.numpy()))

		N = db['train_data'].N

		db = self.db
		dim_diff = self.input_size - db['output_dim']
		dim_expansion_matrix = np.hstack(( np.eye(db['output_dim']) , np.zeros((db['output_dim'], dim_diff))))
		self.dim_expansion_matrix = numpy2Variable(dim_expansion_matrix, db['dataType'])

		self.rff = RFF()
		self.rff.initialize_RFF(db['train_data'].X, db['φ_x_mpd'], True, db['dataType'])

		self.H = np.eye(N) - (1.0/N)*np.ones((N, N))
		self.H = numpy2Variable(self.H, db['dataType'])


	def autoencoder_loss(self, x, label, indices):
		db = self.db
		[x_hat, φ_x] = self.forward(x)
		auto_cost = self.mse_loss(x_hat, x)
		return auto_cost

	def compute_loss(self, x, label, indices):
		db = self.db

		[x_hat1, z] = db['dim_reducer_obj'](x)
		expanded_z = torch.mm(z, self.dim_expansion_matrix)
		[z_hat, φ_z] = self.forward(expanded_z)

		[x_hat2, φ_x] = self.forward(x)

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




