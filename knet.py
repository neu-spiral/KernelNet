#!/usr/bin/env python

import sys
sys.path.append('./src')
sys.path.append('./tests')
sys.path.append('./src/models')
sys.path.append('./src/helper')
sys.path.append('./src/optimizer')
sys.path.append('./src/validation')
sys.path.append('./default_settings')

import matplotlib
import numpy as np
import random
import itertools
import socket
import time
import debug
import warnings
from dataset_manipulate import *
from pretrain import *
from AE import *
from AE_validate import *
from storage import *
from DManager import *
from opt_Kernel import *
from wine_raw_data import *
from moon_raw_data import *

if socket.gethostname().find('login') != -1:
	print('\nError : you cannot run program on login node.......\n\n')
	sys.exit() 

np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
warnings.filterwarnings("ignore")

def initialize_data(db):
	print('\nRunning %s with cuda=%s\n\tLoading datasets...'%(db["data_name"], str(db['cuda'])))

	if(db['cuda']): db['dataType'] = torch.cuda.FloatTensor
	else: db['dataType'] = torch.FloatTensor				
	#db['dataType'] = torch.FloatTensor

	if db["test_data_file_name"] == '':
		db['train_data'] = DManager(db["train_data_file_name"], db["train_label_file_name"], db['dataType'])
		db['train_loader'] = DataLoader(dataset=db['train_data'], batch_size=db['batch_size'], shuffle=True)
	else:
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

	σ = float(db['x_mpd']*db["σ_ratio"])
	[L, db['D_inv']] = getLaplacian(db, X, σ, H=H)
	[db['U'], U_normalized] = L_to_U(db, L)

	[allocation, db['init_spectral_nmi']] = kmeans(db['num_of_clusters'], U_normalized, Y=db['train_data'].Y)
	print('\t\tInitial Spectral Clustering NMI on raw data : %.3f'%db['init_spectral_nmi'])


def initialize_network(db, pretrain_knet=True):
	db['net_input_size'] = db['train_data'].d
	db['net_depth'] = db['kernel_net_depth']
	db['net_output_dim'] = db['output_dim']

	if(db['cuda']): db['knet'] = db['kernel_model'](db).cuda()
	else: db['knet'] = db['kernel_model'](db)
	

	if pretrain_knet:
		dataLoader = 'train_loader'
		if not import_pretrained_network(db, 'knet', 'rbm'):
			start_time = time.time() 
			pretrain(db['knet'], db, dataLoader)
			db['knet'].pretrain_time = time.time() - start_time
			export_pretrained_network(db, 'knet', 'rbm')
	
	
		if not import_pretrained_network(db, 'knet', 'end2end'):
			print('\tRunning End to End Autoencoder training...')
			prev_loss = db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)
			start_time = time.time() 
	
			basic_optimizer(db['knet'], db, loss_callback='autoencoder_loss', data_loader_name=dataLoader)
	
			db['knet'].end2end_time = time.time() - start_time
			post_loss = db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)
			print('\n\tError of End to End AE , Before %.3f, After %.3f'%(prev_loss.item(), post_loss.item()))
			export_pretrained_network(db, 'knet', 'end2end')

		#debug.end2end(db)

	db['knet'].initialize_variables(db)
	[db['initial_loss'], db['initial_hsic'], db['initial_AE_loss'], φ_x, U, U_normalized] = db['knet'].get_current_state(db, db['train_data'].X_Var)
	[allocation, db['init_AE+Kmeans_nmi']] = kmeans(db['num_of_clusters'], φ_x, Y=db['train_data'].Y)
	[allocation, db['init_AE+Spectral_nmi']] = kmeans(db['num_of_clusters'], U_normalized, Y=db['train_data'].Y)

	print('\t\tInitial AE + Kmeans NMI : %.3f, AE + Spectral : %.3f'%(db['init_AE+Kmeans_nmi'], db['init_AE+Spectral_nmi']))



def train_kernel_net(db):
	db['opt_K'] = db['opt_K_class'](db)
	db['opt_U'] = db['opt_U_class'](db)
	db['converge_list'] = []


	if not import_pretrained_network(db, 'knet', 'last'):
		start_time = time.time() 
		for count in range(100):
			db['opt_K'].run(count)
			db['opt_U'].run(count)
			if db['exit_cond'](db, count) > 99: break;
	
		db['knet'].train_time = time.time() - start_time
		export_pretrained_network(db, 'knet', 'last')


	db['validate_function'](db)



def define_settings():
	#db = wine_raw_data()
	db = moon_raw_data()


	db = load_db(db)
	return db


db = define_settings()
initialize_data(db)
initialize_embedding(db)
initialize_network(db, pretrain_knet=True)
train_kernel_net(db)

