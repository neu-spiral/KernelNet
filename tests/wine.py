#!/usr/bin/env python

from test_parent import *
from AE import *
from opt_Kernel import *
from termcolor import colored
import sklearn.metrics
import numpy as np
import random


class test_code(test_parent):
	def __init__(self):
		print(colored(('\nRunning Kernel Net : %s'%(__name__)), 'white'))
		db = {}
		#	Data settings
		db['data_name'] = 'wine'
		db['dataType'] = torch.FloatTensor				
		db['center_and_scale'] = True
		db['recompute_10_fold'] = False
		db['use_Degree_matrix'] = True
		db['pretrain_repeats'] = 1

		#	hyperparams
		db['batch_size'] = 5
		db['num_of_clusters'] = 3
		db['use_Degree_matrix'] = True

		# objs
		db['kernel_model'] = AE
		db['opt_K_class'] = opt_K
		db['opt_U_class'] = opt_U
		db['exit_cond_class'] = exit_cond

		test_parent.__init__(self, db)

	def parameter_ranges(self):
		output_dim = [3]
		kernel_net_depth = [7]
		σ_ratio = [1]
		extra_repeat = range(1)
		id_10_fold = [0]
	
		#lambda_ratio = np.arange(0.1, 3, 0.1)
		#random.shuffle(lambda_ratio)
		#lambda_ratio = [0] + list(lambda_ratio)
	
		lambda_ratio = [0]


		random.shuffle(output_dim)
		random.shuffle(kernel_net_depth)
		random.shuffle(σ_ratio)
	
		return [output_dim, kernel_net_depth, σ_ratio, extra_repeat, lambda_ratio, id_10_fold]
	





#		if data_name is None: db['data_name'] = 'wine_75.00'
#		else: db['data_name'] = data_name
#		db['data_file_name'] = '../datasets/' + db['data_name'] + '.csv'
#		db['label_file_name'] = '../datasets/' + db['data_name'] + '_label.csv'
#		db['validation_data_file_name'] = '../datasets/' + db['data_name'] + '_validation.csv'
#		db['validation_label_file_name'] = '../datasets/' + db['data_name'] + '_label_validation.csv'
#
#		db['train_on_full'] = True
#		db['use_svd'] = False
#		db['use_Degree_matrix'] = True
#		db['add_autoencoder_objective'] = True
#		db['autoencoder_objective_constant'] = 1.0		# The actual constant used during optimization
#		db['lambda_ratio'] = 1							# % ratio to bias the weight ratio
#		db['use_U_normalized'] = True
#
#		db['objective_tracker'] = np.array([])		# if commented out, it won't track it
#		db['feasibility_tracker'] = np.array([])		# if commented out, it won't track it
#		db['objective_time_tracker_time'] = np.array([])		# if commented out, it won't track it
#		db['objective_time_tracker_hsic'] = np.array([])		# if commented out, it won't track it
#
#		db['best_path'] = './pre_trained_weights/Best_pk/' 
#		db['auto_encoder_epoc_loop'] = 10000			#	Max num of epoch
#		db['learning_rate'] = 0.001
#		db['print_optimizer_loss'] = True
#		db['print_pretrain_status'] = True
#		db['percent_of_eigen_values_to_keep'] = 0.9
#
#		db['center_and_scale'] = True
#		db['use_denoising_AE'] = False
#		db['Kx_diag_to_zero'] = True
#		db['kernel_type'] = 'DAE'					# default, cnn, DAE
#		db['data_source'] = 'local_file'				# link_download, load_image, local_file
#		db['output_dim'] = 7
#		db['num_hidden'] = 12
#		db['kernel_net_depth'] = 7 
#		db['batch_size'] = 5							#	Size for each batch
#		db['num_of_clusters'] = 3
#		db['pretrain_repeats'] = 2
#		db['sigma_ratio'] = 0.3							# multiplied to the median of pairwise distance as sigma
#		db['dataType'] = torch.FloatTensor				#torch.FloatTensor
#
#		test_base.__init__(self, db)
#
#	def plot_before_n_after(self):
#		db = self.db
#		X = db['dataset'].x.numpy()
#		X_hat = db['X_hat']
#		s = db['kernel_net'].sigma
#		gamma = 1.0/(2*s*s)
#
#
#		Kx_orig = sklearn.metrics.pairwise.rbf_kernel(X, gamma=gamma)
#		Kx_orig = np.flipud(Kx_orig)
#		Kx_after = sklearn.metrics.pairwise.rbf_kernel(X_hat, gamma=gamma)
#		Kx_after = np.flipud(Kx_after)
#
#		plt.figure(1)
#		self.plot_HMap(121, Kx_orig, 'Original similarity Matrix of dataset A')
#		self.plot_HMap(122, Kx_after, 'Kernelized similarity Matrix of dataset A')
#
#		plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=0.4)
#		plt.show()
#
#
#
#
#def instantiate(data_name=None):
#	return wine(data_name)
#
#def save_run(B):
#	if B.train() == False:
#		return False
#
#	my_results = kernel_on_different_data(B.db)
#	store_best_results(B.db, my_results)
#	#store_all_results(B.db, my_results)
#	return True
#
#def run_only(B):
#	if B.train() == False:
#		return False
#	return True
#
#def run(B):
#	if B.train() == False:
#		return False
#
#	my_results = kernel_on_different_data(B.db)
#	store_best_results(B.db, my_results)
#	#store_all_results(B.db, my_results)
#
#	return True
#
#
##def parameter_ranges():
##	output_dim = [3]
##	#kernel_net_depth = [11]
##	kernel_net_depth = [6]
##	sigma_ratio = [1]
##	lambda_ratio = np.arange(0.1, 3, 0.1)
##	num_hidden = [2]
##	use_denoising_AE = [False]
##	Kx_diag_to_zero = [True]
##	extra_repeat = range(7)
##	filenames = ['wine_75.00']
##
##
##	random.shuffle(lambda_ratio)
##	lambda_ratio = [0] + list(lambda_ratio)
##
##
##	random.shuffle(output_dim)
##	random.shuffle(kernel_net_depth)
##	random.shuffle(sigma_ratio)
##	random.shuffle(num_hidden)
##	random.shuffle(use_denoising_AE)
##	random.shuffle(Kx_diag_to_zero)
##
##	return [output_dim, kernel_net_depth, sigma_ratio, use_denoising_AE, Kx_diag_to_zero, num_hidden, extra_repeat, filenames, lambda_ratio]
#
#def parameter_ranges():
#	output_dim = range(3,11)
#	kernel_net_depth = range(4,15)
#	#sigma_ratio = np.arange(0.1, 3, 0.1)
#	sigma_ratio = [1]
#	lambda_ratio = np.arange(0.1, 3, 0.1)
#	num_hidden = range(4,14)
#	use_denoising_AE = [False]
#	Kx_diag_to_zero = [True]
#	extra_repeat = range(1)
#	filenames = ['wine_75.00']
#
#	random.shuffle(lambda_ratio)
#	lambda_ratio = [0] + list(lambda_ratio)
#
#	random.shuffle(output_dim)
#	random.shuffle(kernel_net_depth)
#	random.shuffle(sigma_ratio)
#	random.shuffle(num_hidden)
#	random.shuffle(use_denoising_AE)
#	random.shuffle(Kx_diag_to_zero)
#
#	return [output_dim, kernel_net_depth, sigma_ratio, use_denoising_AE, Kx_diag_to_zero, num_hidden, extra_repeat, filenames, lambda_ratio]

