#!/usr/bin/env python3

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from distances import *
from terminal_print import *
import numpy as np

class DManager(Dataset):
	def __init__(self, db):
		self.dtype = np.float64				#np.float32
		self.array_format = 'numpy'			# numpy, pytorch

		self.X = np.loadtxt(db['orig_data_file_name'], delimiter=',', dtype=self.dtype)			
		self.Y = np.loadtxt(db['orig_label_file_name'], delimiter=',', dtype=np.int32)
		if db['center_and_scale']: self.X = preprocessing.scale(self.X)

		db['N'] = self.X.shape[0]					# num of samples
		db['d']  = self.X.shape[1]					# num of Dims
		db['mpd'] = float(median_of_pairwise_distance(self.X))
		self.db = db

	def __getitem__(self, index):
		return self.X[index], self.Y[index], index


	def __len__(self):
		return self.X.shape[0]

