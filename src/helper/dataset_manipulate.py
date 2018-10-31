#!/usr/bin/env python
from path_tools import *
from numpy import genfromtxt
from kernel_lib import *
from DManager import *
from distances import *
import numpy as np
import sys
import os



def gen_subset_and_rest(db):
	ensure_path_exists('%s/subset_rest'%(db['data_path']))
	train_path = ('%s/subset_rest/train.csv'%(db['data_path']))
	test_path = ('%s/subset_rest/test.csv'%(db['data_path']))
	train_label_path = ('%s/subset_rest/train_label.csv'%(db['data_path']))
	test_label_path = ('%s/subset_rest/test_label.csv'%(db['data_path']))
	if (db['recompute_data_split'] == False) and os.path.exists(train_path): return

	orig_data = DManager(db['orig_data_file_name'], db['orig_label_file_name'], torch.FloatTensor)
	σ = median_of_pairwise_distance(orig_data.X)
	K_orig = rbk_sklearn(orig_data.X, σ)
	[D,V] = np.linalg.eigh(K_orig)
	scaled_cumsum_D = np.cumsum(np.flip(D)/np.sum(D))
	eigLen = len(scaled_cumsum_D[scaled_cumsum_D < 0.95])
	largest_eigs = np.flip(D)[0:eigLen]
	largest_eigs = largest_eigs/np.sum(largest_eigs)

	N = orig_data.N
	for test_percent in np.arange(0.05,0.8,0.05):
		for rep in range(10):
			inc = int(np.floor(test_percent*N))
			if inc < eigLen: continue
	
			rp = np.random.permutation(N).tolist()
			test_set_id = rp[0:inc]
			sample_X = orig_data.X[test_set_id,:]
	
			K_new = rbk_sklearn(sample_X, σ)
			[D,V] = np.linalg.eigh(K_new)
			small_eigs = np.flip(D)[0:eigLen]
			small_eigs = small_eigs/np.sum(small_eigs)
	
			Kd = np.absolute(largest_eigs - small_eigs)
	
			print(sample_X.shape)
			print(np.max(Kd))
			print(Kd)

	import pdb; pdb.set_trace()

		#train_set_id = list(set(rp) - set(test_set_id))



	np.savetxt(train_path, orig_data.X[train_set_id,:], delimiter=',', fmt='%f') 
	np.savetxt(test_path, orig_data.X[test_set_id,:], delimiter=',', fmt='%f') 
	np.savetxt(train_label_path, orig_data.Y[train_set_id], delimiter=',', fmt='%d') 
	np.savetxt(test_label_path, orig_data.Y[test_set_id], delimiter=',', fmt='%d') 



def gen_train_validate_data(db):
	db['train_validate_folder'] = db['data_folder'] + 'train_validate/'
	ensure_path_exists(db['train_validate_folder'])

	db['train_path'] = ('%strain.csv'%(db['train_validate_folder']))
	db['valid_path'] = ('%svalid.csv'%(db['train_validate_folder']))
	db['train_label_path'] = ('%strain_label.csv'%(db['train_validate_folder']))
	db['valid_label_path'] = ('%svalid_label.csv'%(db['train_validate_folder']))


	tran_valid_exist = True
	path_list = [db['train_path'], db['valid_path'], db['train_label_path'], db['valid_label_path']] 
	if not path_list_exists(path_list): tran_valid_exist = False
	if tran_valid_exist == True: return
	x = genfromtxt(db['train_data_file_name'], delimiter=',')
	y = genfromtxt(db['train_label_file_name'], delimiter=',')
	N = x.shape[0]


	rp = np.random.permutation(N).tolist()
	num_of_train = int(0.80*N)
	train_set_indx = rp[0:num_of_train]
	valid_set_indx = list(set(rp) - set(train_set_indx))

	np.savetxt(db['train_path'], x[train_set_indx,:], delimiter=',', fmt='%f') 
	np.savetxt(db['valid_path'], x[valid_set_indx,:], delimiter=',', fmt='%f') 
	np.savetxt(db['train_label_path'], y[train_set_indx], delimiter=',', fmt='%d') 
	np.savetxt(db['valid_label_path'], y[valid_set_indx], delimiter=',', fmt='%d') 


def gen_training_and_test(db, test_percent):

	ensure_path_exists('%s/train_test'%(db['data_path']))
	train_path = ('%s/train_test/train.csv'%(db['data_path']))
	test_path = ('%s/train_test/test.csv'%(db['data_path']))
	train_label_path = ('%s/train_test/train_label.csv'%(db['data_path']))
	test_label_path = ('%s/train_test/test_label.csv'%(db['data_path']))
	if (db['recompute_data_split'] == False) and os.path.exists(train_path): return


	orig_data = DManager(db['orig_data_file_name'], db['orig_label_file_name'], torch.FloatTensor)

	N = orig_data.N
	loc = 0
	inc = int(np.floor(test_percent*N))
	rp = np.random.permutation(N).tolist()

	test_set_id = rp[0:inc]
	train_set_id = list(set(rp) - set(test_set_id))

	np.savetxt(train_path, orig_data.X[train_set_id,:], delimiter=',', fmt='%f') 
	np.savetxt(test_path, orig_data.X[test_set_id,:], delimiter=',', fmt='%f') 
	np.savetxt(train_label_path, orig_data.Y[train_set_id], delimiter=',', fmt='%d') 
	np.savetxt(test_label_path, orig_data.Y[test_set_id], delimiter=',', fmt='%d') 


def gen_10_fold_data(db):
	#	Does the 10 fold path exist
	ten_fold_exist = True
	for i in range(0,10):
		train_path = ('%s/10_fold/split_%d/train.csv'%(db['data_path'],i))
		test_path = ('%s/10_fold/split_%d/test.csv'%(db['data_path'],i))
		train_label_path = ('%s/10_fold/split_%d/train_label.csv'%(db['data_path'],i))
		test_label_path = ('%s/10_fold/split_%d/test_label.csv'%(db['data_path'],i))

		path_list = [train_path, train_label_path, test_path, test_label_path] 
		if not path_list_exists(path_list): ten_fold_exist = False


	if db['recompute_data_split'] == False and ten_fold_exist == True: return


	ensure_path_exists('%s/10_fold'%(db['data_path']))
	for i in range(0,10):
		split_folder = ('%s/10_fold/split_%d'%(db['data_path'],i))
		ensure_path_exists(split_folder)

	if(db['cuda']): orig_data = DManager(db['orig_data_file_name'], db['orig_label_file_name'], torch.cuda.FloatTensor)
	else: orig_data = DManager(db['orig_data_file_name'], db['orig_label_file_name'], torch.FloatTensor)

	N = orig_data.N
	loc = 0
	inc = int(np.floor(N/10.0))
	rp = np.random.permutation(N).tolist()


	for i in range(10):
		train_path = ('%s/10_fold/split_%d/train.csv'%(db['data_path'],i))
		test_path = ('%s/10_fold/split_%d/test.csv'%(db['data_path'],i))
		train_label_path = ('%s/10_fold/split_%d/train_label.csv'%(db['data_path'],i))
		test_label_path = ('%s/10_fold/split_%d/test_label.csv'%(db['data_path'],i))


		test_set_id = rp[loc:loc+inc]
		loc = loc+inc

		train_set_id = list(set(rp) - set(test_set_id))
		np.savetxt(train_path, orig_data.X[train_set_id,:], delimiter=',', fmt='%f') 
		np.savetxt(test_path, orig_data.X[test_set_id,:], delimiter=',', fmt='%f') 
		np.savetxt(train_label_path, orig_data.Y[train_set_id], delimiter=',', fmt='%d') 
		np.savetxt(test_label_path, orig_data.Y[test_set_id], delimiter=',', fmt='%d') 



