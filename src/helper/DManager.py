#!/usr/bin/env python3

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from distances import *
from terminal_print import *
import numpy as np

class DManager(Dataset):
	def __init__(self, data_path, label_path, dataType, center_data=True):
		print('\tLoading : %s'%data_path)
		self.dtype = np.float64				#np.float32
		self.array_format = 'numpy'			# numpy, pytorch
		self.label_path = label_path
		if data_path == '': return

		self.X = np.loadtxt(data_path, delimiter=',', dtype=self.dtype)			
		if label_path != '': 
			self.Y = np.loadtxt(label_path, delimiter=',', dtype=np.int32)
		if center_data: self.X = preprocessing.scale(self.X)

		self.N = self.X.shape[0]
		self.d = self.X.shape[1]
		

		self.X_Var = torch.tensor(self.X)
		if label_path != '': self.Y_Var = torch.tensor(self.Y)

		self.X_Var = Variable(self.X_Var.type(dataType), requires_grad=False)
		if label_path != '': self.Y_Var = Variable(self.Y_Var.type(dataType), requires_grad=False)

		print('\t\tData of size %dx%d was loaded ....'%(self.N, self.d))

	def __getitem__(self, index):
		if self.label_path == '':
			return self.X[index], 0, index
		else:
			return self.X[index], self.Y[index], index


	def __len__(self):
		try: return self.X.shape[0]
		except: return 0

