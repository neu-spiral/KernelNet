#!/usr/bin/env python
from path_tools import *
import os

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


	if db['recompute_10_fold'] == False and ten_fold_exist == True: return


	ensure_path_exists('%s/10_fold'%(db['data_path']))
	for i in range(0,10):
		split_folder = ('%s/10_fold/split_%d'%(db['data_path'],i))
		ensure_path_exists(split_folder)


	#	Split data into 10 and save them accordingly

