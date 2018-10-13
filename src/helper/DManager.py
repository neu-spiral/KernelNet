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

		self.X = np.loadtxt(data_path, delimiter=',', dtype=self.dtype)			
		self.Y = np.loadtxt(label_path, delimiter=',', dtype=np.int32)
		if center_data: self.X = preprocessing.scale(self.X)

		self.N = self.X.shape[0]
		self.d = self.X.shape[1]
		

		self.X_Var = torch.tensor(self.X)
		self.Y_Var = torch.tensor(self.Y)
		self.X_Var = Variable(self.X_Var.type(dataType), requires_grad=False)
		self.Y_Var = Variable(self.Y_Var.type(dataType), requires_grad=False)


	def __getitem__(self, index):
		return self.X[index], self.Y[index], index


	def __len__(self):
		return self.X.shape[0]

