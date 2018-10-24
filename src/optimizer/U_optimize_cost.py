#!/usr/bin/env python

import numpy as np
import sklearn.metrics

class U_optimize_cost:
	def __init__(self, X, L, sigma):
		self.L = L
		self.X = X
		self.sigma = sigma

	def compute_cost(self, U):
		Ku = U.dot(U.T)
		cost = -np.sum(self.L*Ku)
		return cost

	def compute_gradient(self, U):
		grad = -2*self.L.dot(U)
		return grad
