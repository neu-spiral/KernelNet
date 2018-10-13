#!/usr/bin/env python

import sys
sys.path.append('./src')
sys.path.append('./tests')
sys.path.append('./src/models')
sys.path.append('./src/helper')
sys.path.append('./src/optimizer')


import matplotlib
import numpy as np
import random
import itertools
import socket
from dataset_manipulate import *
from pretrain import *
from AE import *
from storage import *
from DManager import *
from opt_Kernel import *

if socket.gethostname().find('login') != -1:
	print('\nError : you cannot run program on login node.......\n\n')
	sys.exit()


np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)

def initialize_data(db):
	print('\nRunning %s\n\tLoading datasets...'%(db["data_name"]))
	db['cuda'] = torch.cuda.is_available()
	if(db['cuda']): db['dataType'] = torch.cuda.FloatTensor
	else: db['dataType'] = torch.FloatTensor				
	#db['dataType'] = torch.FloatTensor

	gen_train_validate_data(db)
	#db['orig_data'] = DManager(db['orig_data_file_name'], db['orig_label_file_name'], db['dataType'])
	db['train_data'] = DManager(db['train_path'], db['train_label_path'], db['dataType'])
	db['valid_data'] = DManager(db["valid_path"], db['valid_label_path'], db['dataType'])

	#db['orig_data_loader'] = DataLoader(dataset=db['orig_data'], batch_size=db['batch_size'], shuffle=True)
	db['train_loader'] = DataLoader(dataset=db['train_data'], batch_size=db['batch_size'], shuffle=True)
	db['valid_loader'] = DataLoader(dataset=db['valid_data'], batch_size=db['batch_size'], shuffle=True)

def initialize_embedding(db):
	print('\tComputing initial U for Spectral Clustering...')
	N = db['train_data'].N
	H = np.eye(N) - (1.0/N)*np.ones((N, N))

	X = db['train_data'].X
	db['x_mpd'] = float(median_of_pairwise_distance(X))
	L = getLaplacian(db, X, db['x_mpd'], H=H)
	[db['U'], U_normalized] = L_to_U(db, L)
	[allocation, db['init_spectral_nmi']] = kmeans(db['num_of_clusters'], db['U'], Y=db['train_data'].Y)
	print('\t\tInitial Spectral Clustering NMI on raw data : %.3f'%db['init_spectral_nmi'])
	

def initialize_network(db):
	db['net_input_size'] = db['train_data'].d
	db['net_depth'] = db['kernel_net_depth']
	db['net_output_dim'] = db['output_dim']

	if(db['cuda']): db['knet'] = db['kernel_model'](db).cuda()
	else: db['knet'] = db['kernel_model'](db)

	dataLoader = 'train_loader'
	if not import_pretrained_network(db, 'knet', 'rbm'):
		pretrain(db['knet'], db, dataLoader)
		export_pretrained_network(db, 'knet', 'rbm')

	print('\tRunning End to End Autoencoder training...')
	prev_loss = db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)
	[avgLoss, avgGrad, progression_slope] = basic_optimizer(db['knet'], db, loss_callback='autoencoder_loss', data_loader_name=dataLoader)
	post_loss = db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)
	print('\n\tError of End to End AE , Before %.3f, After %.3f'%(prev_loss.item(), post_loss.item()))

	db['knet'].initialize_variables()
	db['knet'].get_current_state()


def train_kernel_net(db):
	db['opt_K'] = db['opt_K_class'](db)
	db['opt_U'] = db['opt_U_class'](db)
	db['exit_cond'] = db['exit_cond_class']

	start_time = time.time() 
	for count in itertools.count():
		db['opt_K'].run(count)
		db['opt_U'].run(count)
		count = db['exit_cond'](db)
		if count > 99: break;



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

	# debug tracking
	db['objective_tracker'] = True

	# hyperparams
	db["output_dim"]=5
	db["kernel_net_depth"]=7
	db["σ_ratio"]=1.0
	db["λ_ratio"]=0.0
	db['pretrain_repeats'] = 1
	db['batch_size'] = 5
	db['num_of_clusters'] = 3
	db['use_Degree_matrix'] = True

	# objs
	db['opt_K_class'] = opt_K
	db['opt_U_class'] = opt_U
	db['exit_cond_class'] = exit_cond

	# knet info
	db['kernel_model'] = AE
else:
	db = load_db()



initialize_data(db)
initialize_embedding(db)
initialize_network(db)
#train_kernel_net(db)
import pdb; pdb.set_trace()

