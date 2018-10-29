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


	def initialize_variables(self, db):
		self.db = db
			
		[x_hat, φ_x] = self.forward(db['train_data'].X_Var)
		φ_x = ensure_matrix_is_numpy(φ_x)
		self.φ_x_mpd = float(median_of_pairwise_distance(φ_x))
		self.σ = float(self.φ_x_mpd*db['σ_ratio'])

		N = db['train_data'].N
		self.H = np.eye(N) - (1.0/N)*np.ones((N, N))
		self.mlp_width = db['mlp_width']

	def set_Y(self, Y):
		self.Y = Y

	def get_current_state(self, db, in_x):
		[x_hat, φ_x] = self.forward(in_x)
		φ_x = ensure_matrix_is_numpy(φ_x)

		[DKxD, Dinv] = normalized_rbk_sklearn(φ_x, self.σ)
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
		[x_hat, φ_x] = self.forward(x)
		Kx = self.gaussian_kernel(φ_x, self.σ)

		PP = self.Y[indices, :]
		Ysmall = PP[:, indices]
		obj_loss = -torch.sum(Kx*Ysmall)
		AE_loss = self.mse_loss(x_hat, x)
		loss = obj_loss + db['λ']*AE_loss

		return loss

