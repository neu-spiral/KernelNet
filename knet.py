#!/usr/bin/env python

import sys
sys.path.append('./src')
sys.path.append('./tests')
sys.path.append('./src/models')
sys.path.append('./src/helper')

import matplotlib
import numpy as np
import random
import itertools
import socket
from dataset_manipulate import *
from AE import *
from DManager import *

if socket.gethostname().find('login') != -1:
	print('\nError : you cannot run program on login node.......\n\n')
	sys.exit()


np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

def initialize_data(db):
	gen_train_validate_data(db)
	db['train_data'] = DManager(db['train_path'], db['train_label_path'])
	db['valid_data'] = DManager(db["valid_path"], db['valid_label_path'])

def initialize_network(db):
	db['net_input_size'] = db['train_data'].d
	db['net_depth'] = db['kernel_net_depth']
	db['net_output_dim'] = db['output_dim']


	db['knet'] = db['kernel_model'](db)

	import pdb; pdb.set_trace()

if __name__ == "__main__":
	db = {}
	# Data info
	db["data_name"]="wine"
	db["center_and_scale"]=True
	db["data_path"]="./datasets/wine/"
	db["orig_data_file_name"]="./datasets/wine/wine.csv"
	db["orig_label_file_name"]="./datasets/wine/wine_label.csv"
	db["data_folder"]="./datasets/wine/10_fold/split_0/"
	db["train_data_file_name"]="./datasets/wine/10_fold/split_0/train.csv"
	db["train_label_file_name"]="./datasets/wine/10_fold/split_0/train_label.csv"
	db["test_data_file_name"]="./datasets/wine/10_fold/split_0/test.csv"
	db["test_label_file_name"]="./datasets/wine/10_fold/split_0/test_label.csv"
	db['dataType'] = torch.FloatTensor				

	# hyperparams
	db["output_dim"]=3
	db["kernel_net_depth"]=7
	db["σ_ratio"]=1.0
	db["λ_ratio"]=0.0

	# knet info
	db['kernel_model'] = AE
else:
	db = load_db()



initialize_data(db)
initialize_network(db)
#train_kernel_net(db)


