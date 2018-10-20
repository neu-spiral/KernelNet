#!/usr/bin/env python

import os 

def ensure_path_exists(path):
	if os.path.exists(path): return True
	os.mkdir(path)
	return False

def path_list_exists(path_list):
	for i in path_list:
		if os.path.exists(i) == False: 
			return False

	return True

def create_file(path):
	fin = open(path,'w')
	fin.close()

def delete_file(path):
	if os.path.exists(path):
		os.remove(path)

def remove_files(folder_path):
	if not os.path.exists(folder_path): return 

	file_in_tmp = os.listdir(folder_path)
	for i in file_in_tmp:
		if os.path.isfile(folder_path + i):
			os.remove(folder_path + i)


