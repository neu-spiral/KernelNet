#!/usr/bin/env python

import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F


class rbm(torch.nn.Module):
	def __init__(self, num_input, num_hidden, activation, use_denoising=True, sparse=False, learning_rate=0.001):
		super(rbm, self).__init__()

		self.l1 = torch.nn.Linear(num_input, num_hidden, bias=True)
		self.l2 = torch.nn.Linear(num_hidden, num_input, bias=True)

		self.use_denoising = use_denoising
		self.sparsity = sparse
		self.criterion = torch.nn.MSELoss(size_average=False)
		#self.criterion = torch.nn.CrossEntropyLoss(size_average=False)

		self.learning_rate = learning_rate
		for param in self.parameters():
			if(len(param.data.numpy().shape)) > 1:
				try:
					torch.nn.init.kaiming_normal_(param.data , a=0, mode='fan_in')	
				except:
					torch.nn.init.kaiming_normal(param.data , a=0, mode='fan_in')	
			else:
				pass
				#param.data = torch.zeros(param.data.size())
		self.activation = activation

	def set_ratio(self, ratio):
		self.beta = torch.from_numpy(np.array([ratio]))
		self.beta = Variable(self.beta.type(torch.FloatTensor), requires_grad=False)

	def compute_ratio(self, inputs, labels, indices):
		if self.activation == 'relu':
			y1 = F.relu(self.l1(inputs))
		elif self.activation == 'tanh':
			y1 = F.tanh(self.l1(inputs))
		elif self.activation == 'sigmoid':
			y1 = F.sigmoid(self.l1(inputs))
		elif self.activation == 'softmax':
			y1 = F.softmax(self.l1(inputs))

		xout = self.l2(y1)

		norm_mag = torch.sum(torch.abs(y1))		#	L1 regularizer
		ratio = torch.abs(self.criterion(xout, inputs)/norm_mag)

		if float(norm_mag) < 0.00001:
			print('\n\nObserve inside rbm that norm_mag is nearly zero, how is this possible???\n\n')
			import pdb; pdb.set_trace()

		return ratio

	def get_optimizer(self):
		return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
		

	def innerLayer_out(self, x):
		if self.activation == 'relu':
			y1 = F.relu(self.l1(x))
		elif self.activation == 'sigmoid':
			y1 = F.sigmoid(self.l1(x))
		elif self.activation == 'tanh':
			y1 = F.tanh(self.l1(x))
#		elif self.activation == 'softmax':
#			y1 = F.softmax(self.l1(x), dim=None)
		elif self.activation == 'none':
			y1 = self.l1(x)
		else:
			print('Unknow activation function ' + self.activation + '\n\n')
			raise

		return y1

	def compute_loss(self, inputs, labels, indices):
		if self.sparsity:
			y1 = self.innerLayer_out(inputs)
			xout = self.l2(y1)
			norm_mag = self.beta*torch.max(torch.sum(torch.abs(y1), dim=0))		#	L1 regularizer
			cost = self.criterion(xout, inputs)
			loss = cost + norm_mag
		else:
			xout = self.forward(inputs)
			loss = self.criterion(xout, inputs)

		#l1_reg = Variable( torch.FloatTensor(1), requires_grad=False)
		#for W in self.parameters(): 
		#	l1_reg = l1_reg + W.norm(2)

		return loss
		#return self.criterion(xout, inputs) + 0.01*l1_reg

	def forward(self, x):	
		if self.use_denoising: 
			x = F.dropout(x, p=0.2, training=self.training)	

		y1 = self.innerLayer_out(x)
		y_pred = self.l2(y1)
		return y_pred
