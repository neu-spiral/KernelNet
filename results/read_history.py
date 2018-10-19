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
	print('\tBest nmi based on train loss')
	print('\t\ttrain nmi : %.3f'%best_train_loss['train_nmi'])
	print('\t\tvalid nmi : %.3f'%best_train_loss['valid_nmi'])
	print('\t\ttrain loss : %.3f'%best_train_loss['train_loss'])
	print('\t\tvalid loss : %.3f'%best_train_loss['valid_loss'])
	print('\tBest nmi based on valid loss')
	print('\t\ttrain nmi : %.3f'%best_valid_loss['train_nmi'])
	print('\t\tvalid nmi : %.3f'%best_valid_loss['valid_nmi'])
	print('\t\ttrain loss : %.3f'%best_valid_loss['train_loss'])
	print('\t\tvalid loss : %.3f'%best_valid_loss['valid_loss'])
	print('\tBest overall train nmi')
	print('\t\ttrain nmi : %.3f'%best_train_nmi['train_nmi'])
	print('\t\tvalid nmi : %.3f'%best_train_nmi['valid_nmi'])
	print('\t\ttrain loss : %.3f'%best_train_nmi['train_loss'])
	print('\t\tvalid loss : %.3f'%best_train_nmi['valid_loss'])
	print('\tBest overall valid nmi')
	print('\t\ttrain nmi : %.3f'%best_valid_nmi['train_nmi'])
	print('\t\tvalid nmi : %.3f'%best_valid_nmi['valid_nmi'])
	print('\t\ttrain loss : %.3f'%best_valid_nmi['train_loss'])
	print('\t\tvalid loss : %.3f'%best_valid_nmi['valid_loss'])


X_valid = []
Y_valid = []
X_train = []
Y_train = []

for result in run_list[0]:
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


