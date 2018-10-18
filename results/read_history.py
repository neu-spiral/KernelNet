#!/usr/bin/env python

import pickle


project_name = 'wine'
fold_id = '0_'

history = pickle.load(open('./' + project_name + '/' + fold_id + 'run_history.pk','rb'))

num_of_runs = len(history['list_of_runs'])
best_train_nmi = history['best_train_nmi']
best_valid_nmi = history['best_valid_nmi']
best_train_loss = history['best_train_loss']
best_valid_loss = history['best_valid_loss']


print('Num of runs : %d'%num_of_runs)
print('Best train nmi : %.3f'%best_train_nmi['train_nmi'])
print('Best valid nmi : %.3f'%best_valid_nmi['valid_nmi'])
print('Best train loss : %.3f'%best_train_loss['train_loss'])
print('Best valid loss : %.3f'%best_valid_loss['valid_loss'])

