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
from matplotlib import pyplot as plt
from dataset_manipulate import *
from pretrain import *
from AE import *
from AE_validate import *
from storage import *
from DManager import *
from opt_Kernel import *
from wine_raw_data import *
from wine_sm import *
from moon_raw_data import *
from moon_raw_data_sm import *
from spiral_raw_data import *
from face_raw_data import *
from face_raw_data_sm import *
from rcv_raw_data import *
from cancer_raw_data import *



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

	db['train_data'] = DManager(db["train_data_file_name"], db["train_label_file_name"], db['dataType'])
	db['train_loader'] = DataLoader(dataset=db['train_data'], batch_size=db['batch_size'], shuffle=True)

	if 'train_test_dataset' in db:
		db['test_data'] = DManager(db["test_data_file_name"], db["test_label_file_name"], db['dataType'])
		db['test_loader'] = DataLoader(dataset=db['test_data'], batch_size=db['batch_size'], shuffle=True)
	elif 'using_10_fold_dataset' in db:
		gen_train_validate_data(db)
		db['valid_data'] = DManager(db["valid_path"], db['valid_label_path'], db['dataType'])
		db['valid_loader'] = DataLoader(dataset=db['valid_data'], batch_size=db['batch_size'], shuffle=True)

def initialize_embedding(db):
	print('\tComputing initial U for Spectral Clustering...')
	N = db['train_data'].N
	H = np.eye(N) - (1.0/N)*np.ones((N, N))

	X = db['train_data'].X
	db['x_mpd'] = float(median_of_pairwise_distance(X))

	σ = float(db['x_mpd']*db["σ_ratio"])
	[L, db['D_inv']] = getLaplacian(db, X, σ, H=H)
	[db['U'], db['U_normalized']] = L_to_U(db, L)
	
	[allocation, db['init_spectral_nmi']] = kmeans(db['num_of_clusters'], db['U_normalized'], Y=db['train_data'].Y)
	print('\t\tInitial Spectral Clustering NMI on raw data : %.3f'%db['init_spectral_nmi'])


def initialize_network(db, pretrain_knet=True, ignore_in_batch=False):
	db['net_input_size'] = db['train_data'].d
	db['net_depth'] = db['kernel_net_depth']
	db['net_output_dim'] = db['output_dim']

	if(db['cuda']): db['knet'] = db['kernel_model'](db).cuda()
	else: db['knet'] = db['kernel_model'](db)
	

	if pretrain_knet:
		dataLoader = 'train_loader'
		if not import_pretrained_network(db, 'knet', 'rbm', ignore_in_batch):
			start_time = time.time() 
			pretrain(db['knet'], db, dataLoader)
			db['knet'].pretrain_time = time.time() - start_time
			export_pretrained_network(db, 'knet', 'rbm', ignore_in_batch)
	
	
		if not import_pretrained_network(db, 'knet', 'end2end', ignore_in_batch):
			print('\tRunning End to End Autoencoder training...')
			prev_loss = db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)
			start_time = time.time() 
	
			basic_optimizer(db['knet'], db, loss_callback='autoencoder_loss', data_loader_name=dataLoader)
	
			db['knet'].end2end_time = time.time() - start_time
			db['knet'].end2end_error = (db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)).item()
			print('\n\tError of End to End AE , Before %.3f, After %.3f'%(prev_loss.item(), db['knet'].end2end_error))
			export_pretrained_network(db, 'knet', 'end2end', ignore_in_batch)
		else:
			print('\t\tError of End to End AE : %.3f'%(db['knet'].end2end_error))

		#debug.end2end(db)

	db['knet'].initialize_variables(db)

	#db['RFF'] = RFF(sample_num=20)
	#db['RFF'].initialize_RFF(db['train_data'].X, db['knet'].σ, False, None)
	#ϕ_x = db['RFF'].np_feature_map(db['train_data'].X)
	#[allocation, db['init_ϕ_x_nmi']] = kmeans(db['num_of_clusters'], ϕ_x, Y=db['train_data'].Y)
#[db['U'], db['U_normalized']]

	[db['initial_loss'], db['initial_hsic'], db['initial_AE_loss'], ψ_x, U, U_normalized] = db['knet'].get_current_state(db, db['train_data'].X_Var)
	[allocation, db['init_AE+Kmeans_nmi']] = kmeans(db['num_of_clusters'], ψ_x, Y=db['train_data'].Y)
	[allocation, db['init_AE+Spectral_nmi']] = kmeans(db['num_of_clusters'], U_normalized, Y=db['train_data'].Y)
	db['U'] = Allocation_2_Y(allocation)


	extra_info = ''
	if 'test_data' in db:
		[db['test_initial_loss'], db['test_initial_hsic'], db['test_initial_AE_loss'], test_ψ_x, test_U, test_U_normalized] = db['knet'].get_current_state(db, db['test_data'].X_Var)
		[allocation, db['init_AE+Kmeans_nmi_onTest']] = kmeans(db['num_of_clusters'], test_U_normalized, Y=db['test_data'].Y)
		extra_info = ', AE+Kmeans on Test NMI : %.3f'%db['init_AE+Kmeans_nmi_onTest']

	print('\t\tInitial AE + Kmeans NMI : %.3f, AE + Spectral : %.3f%s'%(db['init_AE+Kmeans_nmi'], db['init_AE+Spectral_nmi'], extra_info))
	#import pdb; pdb.set_trace()



def train_kernel_net(db):
	db['opt_K'] = db['opt_K_class'](db)
	db['opt_U'] = db['opt_U_class'](db)
	db['converge_list'] = []


	if not import_pretrained_network(db, 'knet', 'last', True):
		start_time = time.time() 
		for count in range(100):
			db['opt_K'].run(count)
			db['opt_U'].run(count)
			if db['exit_cond'](db, count) > 99: break;

		db['λ'] = 0
		db['λ_ratio'] = 0
		for count in range(1):
			db['opt_K'].run(count)
			db['opt_U'].run(count)
			if db['exit_cond'](db, count) > 99: break;

		db['knet'].train_time = time.time() - start_time
		export_pretrained_network(db, 'knet', 'last', True)


	[db['train_loss'], db['train_hsic'], db['train_AE_loss'], φ_x, U, U_normalized] = db['knet'].get_current_state(db, db['train_data'].X_Var)
	db['validate_function'](db)
	#debug.plot_output(db)


#---------------------------------------------------------------------


def define_settings():
	db = wine_raw_data()
	#db = cancer_raw_data()
	#db = wine_sm()
	#db = moon_raw_data()
	#db = moon_raw_data_sm()
	#db = spiral_raw_data()
	#db = face_raw_data()
	#db = face_raw_data_sm()
	#db = rcv_raw_data()

	db = load_db(db)
	return db


def check_σ():
	db = define_settings()
	for i in np.arange(0.1,4,0.1):
		print(i)
		db["σ_ratio"] = i
		initialize_data(db)
		initialize_embedding(db)


def discover_lowest_end2end_error():
	db = define_settings()
	initialize_data(db)
	initialize_embedding(db)
	initialize_network(db, pretrain_knet=True, ignore_in_batch=True)
	save_to_lowest_end2end(db)

def default_run():
	db = define_settings()
	initialize_data(db)
	initialize_embedding(db)
	initialize_network(db, pretrain_knet=True)
	train_kernel_net(db)





default_run()
#discover_lowest_end2end_error()
