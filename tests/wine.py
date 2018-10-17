#!/usr/bin/env python

from test_parent import *
from AE import *
from opt_Kernel import *
from AE_validate import *
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
		db['exit_cond'] = exit_cond
		db['validate_function'] = AE_validate

		test_parent.__init__(self, db)

	def parameter_ranges(self):
		output_dim = [3,4,5,6,7]
		kernel_net_depth = [4,5,6,7,8]
		σ_ratio = [1]
		extra_repeat = range(20)
		id_10_fold = [0]
	
		#lambda_ratio = np.arange(0.1, 3, 0.1)
		#random.shuffle(lambda_ratio)
		#lambda_ratio = [0] + list(lambda_ratio)
	
		lambda_ratio = [0]


		random.shuffle(output_dim)
		random.shuffle(kernel_net_depth)
		random.shuffle(σ_ratio)
	
		return [output_dim, kernel_net_depth, σ_ratio, extra_repeat, lambda_ratio, id_10_fold]

