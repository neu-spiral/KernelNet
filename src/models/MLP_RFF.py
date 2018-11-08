#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
from kernel_lib import *
from MLP_autoencoder import *
from format_conversion import *
from distances import *
from RFF import *
import torch.nn.functional as F
import time 

class MLP_RFF(MLP_autoencoder):
	def __init__(self, db, add_decoder=True, learning_rate=0.001):
		super(MLP_RFF, self).__init__(db, add_decoder, learning_rate)


	def initialize_variables(self, db):
		self.db = db
			
		[x_hat, φ_x] = self.forward(db['train_data'].X_Var)
		φ_x = ensure_matrix_is_numpy(φ_x)
		self.φ_x_mpd = float(median_of_pairwise_distance(φ_x))
		self.σ = float(self.φ_x_mpd*db['σ_ratio'])

		N = db['train_data'].N
		d = db['train_data'].d
		self.H = np.eye(N) - (1.0/N)*np.ones((N, N))
		self.mlp_width = db['mlp_width']


		self.sample_num=20000
		b = 2*np.pi*np.random.rand(1, self.sample_num)
		self.phase_shift = np.matlib.repmat(b, N, 1)	
		self.rand_proj = np.random.randn(d, self.sample_num)/(self.σ)

		self.phase_shift = torch.from_numpy(self.phase_shift)
		self.phase_shift = Variable(self.phase_shift.type(db['dataType']), requires_grad=False)

		self.rand_proj = torch.from_numpy(self.rand_proj)
		self.rand_proj = Variable(self.rand_proj.type(db['dataType']), requires_grad=False)

		mask = 1 - torch.eye(N)
		self.mask = Variable(mask.type(db['dataType']), requires_grad=False)

	def compute_RFF_Gaussian(self, x):
		db = self.db

		if type(x) == np.ndarray: x = numpy2Variable(x, db['dataType'], need_grad=False)

		P = torch.cos(torch.mm(x,self.rand_proj) + self.phase_shift)
		K = torch.mm(P, P.transpose(0,1))
		K = (2.0/self.sample_num)*K
		K = F.relu(K)
		K = K*self.mask

		ds = 1.0/torch.sqrt(torch.sum(K, dim=0))
		D = torch.ger(ds,ds)
		DKD = K*D
		return [DKD, K]

	def set_Y(self, Y):
		self.Y = Y

	def get_current_state(self, db, in_x):
		[x_hat, φ_x] = self.forward(in_x)
		φ_x = ensure_matrix_is_numpy(φ_x)

		[DKxD, K] = self.compute_RFF_Gaussian(φ_x)
		HDKxDH = center_matrix(db, DKxD)
		[U, U_normalized] = L_to_U(db, HDKxDH)
		Ku = U.dot(U.T)

		current_hsic = -float(np.sum(HDKxDH*Ku))
		current_AE_loss = float(ensure_matrix_is_numpy(self.autoencoder_loss(in_x, None, None)))

		if 'λ_obj_ratio' not in db:
			db['λ_obj_ratio'] = float(np.abs(current_hsic/current_AE_loss))
			db['λ'] = float(db["λ_ratio"]*db['λ_obj_ratio'])

		current_loss = float(current_hsic + db['λ']*current_AE_loss)
		return [current_loss, current_hsic, current_AE_loss, φ_x, U, U_normalized]



	def autoencoder_loss(self, x, label, indices):
		db = self.db
		[x_hat, φ_x] = self.forward(x)
		auto_cost = self.mse_loss(x_hat, x)

		if self.input_size == self.output_dim:
			mid_cost = self.mse_loss(x_hat, φ_x)
			total_cost = auto_cost + mid_cost
			return total_cost
		else:
			return auto_cost

	def compute_loss(self, x, label, indices):
		db = self.db
		[x_hat, φ_x] = self.forward(db['train_data'].X_Var)

		[DKD, K] = self.compute_RFF_Gaussian(φ_x)
		#Kx = self.gaussian_kernel(φ_x, self.σ)

		obj_loss = -torch.sum(DKD*self.Y)
		AE_loss = self.mse_loss(x_hat, db['train_data'].X_Var)
		loss = obj_loss + db['λ']*AE_loss

		return loss

