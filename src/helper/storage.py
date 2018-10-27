
import sys
sys.path.append('./src/validation')

from path_tools import *
from opt_Kernel import *
from sm_opt_Kernel import *
from AE_validate import *
from AE import *
import numpy as np
import pickle
import sys
import shutil
import time


def import_pretrained_network(db, keyVal, stage_name):
	if 'running_batch_mode' in db: return False

	ensure_path_exists('./pretrained')
	ensure_path_exists('./pretrained/' + db['data_name'])
	path_list = ['./pretrained/' + db['data_name'] + '/' + db['data_name'] + '_' + stage_name + '.pk']

	print('\tLoading %s from %s...'%(stage_name, keyVal))
	if path_list_exists(path_list):
		list_of_networks = pickle.load( open( path_list[0], "rb" ) )

		if type(list_of_networks) == list:
			for itm in list_of_networks:
				test1 = itm.input_size == db[keyVal].input_size
				test2 = itm.output_dim == db[keyVal].output_dim
				test3 = itm.net_depth == db[keyVal].net_depth
				try: test4 = itm.mlp_width == db[keyVal].mlp_width
				except: return False
				
	
				if test1 and test2 and test3 and test4:
					db[keyVal] = itm
					print('\t\tSucessful...')
					return True
		else:
			test1 = list_of_networks.input_size == db[keyVal].input_size
			test2 = list_of_networks.output_dim == db[keyVal].output_dim
			test3 = list_of_networks.net_depth == db[keyVal].net_depth
			test4 = list_of_networks.mlp_width == db[keyVal].mlp_width

			if test1 and test2 and test3 and test4:
				db[keyVal] = list_of_networks
				print('\t\tSucessful...')
				return True

	print('\t\tFailed...')
	return False


def export_pretrained_network(db, keyVal, stage_name):
	if 'running_batch_mode' in db: return

	ensure_path_exists('./pretrained')
	ensure_path_exists('./pretrained/' + db['data_name'])
	pth = './pretrained/' + db['data_name'] + '/' + db['data_name'] + '_' + stage_name + '.pk'
	path_list = [pth]

	if path_list_exists(path_list):
		list_of_networks = pickle.load( open( path_list[0], "rb" ) )

		for p, itm in enumerate(list_of_networks):
			test1 = itm.input_size == db[keyVal].input_size
			test2 = itm.output_dim == db[keyVal].output_dim
			test3 = itm.net_depth == db[keyVal].net_depth
			test4 = itm.mlp_width == db[keyVal].mlp_width

			if test1 and test2 and test3 and test4:
				list_of_networks[p] = db[keyVal]
				pickle.dump( list_of_networks, open(pth, "wb" ) )
				return 
	else:
		list_of_networks = []


	list_of_networks.append(db[keyVal])
	pickle.dump( list_of_networks, open(pth, "wb" ) )


def load_db(db):
	if len(sys.argv) == 1: return db

	fin = open(sys.argv[1],'r')
	cmds = fin.readlines()
	fin.close()
	
	for i in cmds: 
		try:
			exec(i)
		except:
			import pdb; pdb.set_trace()
	return db


