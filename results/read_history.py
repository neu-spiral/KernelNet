#!/usr/bin/env python

import pickle
import os
import matplotlib
import numpy as np
import matplotlib.pyplot as plt; plt.rcdefaults()


project_name = 'wine'
run_list = {}

for i in range(20):
	fold_id = str(i) + '_'
	if not os.path.exists('./' + project_name + '/' + fold_id + 'run_history.pk'): continue

	history = pickle.load(open('./' + project_name + '/' + fold_id + 'run_history.pk','rb'))
	
	run_list[i] = history['list_of_runs']
	num_of_runs = len(history['list_of_runs'])
	best_train_nmi = history['best_train_nmi']
	best_valid_nmi = history['best_valid_nmi']
	best_train_loss = history['best_train_loss']
	best_valid_loss = history['best_valid_loss']
	
	
	print('\nRun ID : %s with Num of runs : %d'%(fold_id, num_of_runs))
	print('\tBest nmi based on train loss')		#
	vals = (best_train_loss['train_nmi'], best_train_loss['valid_nmi'], best_train_loss['train_loss'], best_train_loss['valid_loss'])
	print('\t\ttrain nmi : %.3f, valid nmi : %.3f, train loss : %.3f, valid loss : %.3f'%vals)
	params = (best_train_loss["σ_ratio"], best_train_loss["λ_ratio"], best_train_loss['output_dim'], best_train_loss["kernel_net_depth"])
	print('\t\tσ_ratio : %.3f, λ_ratio : %.3f, output_dim : %.3f, kernel_net_depth : %.3f'%params)

	print('\tBest nmi based on valid loss')		#
	vals = (best_valid_loss['train_nmi'], best_valid_loss['valid_nmi'], best_valid_loss['train_loss'], best_valid_loss['valid_loss'])
	print('\t\ttrain nmi : %.3f, valid nmi : %.3f, train loss : %.3f, valid loss : %.3f'%vals)
	params = (best_valid_loss["σ_ratio"], best_valid_loss["λ_ratio"], best_valid_loss['output_dim'], best_valid_loss["kernel_net_depth"])
	print('\t\tσ_ratio : %.3f, λ_ratio : %.3f, output_dim : %.3f, kernel_net_depth : %.3f'%params)

	print('\tBest overall train nmi')			#
	vals = (best_train_nmi['train_nmi'], best_train_nmi['valid_nmi'], best_train_nmi['train_loss'], best_train_nmi['valid_loss'])
	print('\t\ttrain nmi : %.3f, valid nmi : %.3f, train loss : %.3f, valid loss : %.3f'%vals)
	params = (best_train_nmi["σ_ratio"], best_train_nmi["λ_ratio"], best_train_nmi['output_dim'], best_train_nmi["kernel_net_depth"])
	print('\t\tσ_ratio : %.3f, λ_ratio : %.3f, output_dim : %.3f, kernel_net_depth : %.3f'%params)

	print('\tBest overall valid nmi')			#
	vals = (best_valid_nmi['train_nmi'], best_valid_nmi['valid_nmi'], best_valid_nmi['train_loss'], best_valid_nmi['valid_loss'])
	print('\t\ttrain nmi : %.3f, valid nmi : %.3f, train loss : %.3f, valid loss : %.3f'%vals)
	params = (best_valid_nmi["σ_ratio"], best_valid_nmi["λ_ratio"], best_valid_nmi['output_dim'], best_valid_nmi["kernel_net_depth"])
	print('\t\tσ_ratio : %.3f, λ_ratio : %.3f, output_dim : %.3f, kernel_net_depth : %.3f'%params)



X_valid = []
Y_valid = []
X_train = []
Y_train = []

for i in range(20):
	if i in run_list:
		for result in run_list[i]:
			X_valid.append(result['valid_loss'])
			Y_valid.append(result['valid_nmi'])
			X_train.append(result['train_loss'])
			Y_train.append(result['train_nmi'])

			plt.figure(figsize=(14,5))
			plt.subplot(121)
			plt.plot(X_train, Y_train, 'x')
			plt.title('Train loss vs NMI')
			plt.subplot(122)
			plt.plot(X_valid, Y_valid, 'x')
			plt.title('Validation loss vs NMI')
			
			plt.show()
			plt.close()
