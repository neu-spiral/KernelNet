
import sys
sys.path.append('./src/validation')

from path_tools import *
from opt_Kernel import *
from sm_opt_Kernel import *
from AE_validate import *
from AE import *
from AE_RFF import *
from MLP import *
from MLP_RFF import *
import numpy as np
import pickle
import sys
import shutil
import time


def import_pretrained_network(db, keyVal, stage_name, ignore_in_batch=False):
	print('\tLoading %s from %s...'%(stage_name, keyVal))
	if ignore_in_batch:
		if 'running_batch_mode' in db: 
			print('\t\tFailed...')
			return False

	ensure_path_exists('./pretrained')
	ensure_path_exists('./pretrained/' + db['data_name'])
	path_list = ['./pretrained/' + db['data_name'] + '/' + db['data_name'] + '_' + stage_name + '.pk']

	if path_list_exists(path_list):
		saved_networks = pickle.load( open( path_list[0], "rb" ) )
		
		test1 = saved_networks.input_size == db[keyVal].input_size
		test2 = saved_networks.output_dim == db[keyVal].output_dim
		test3 = saved_networks.net_depth == db[keyVal].net_depth
		test4 = saved_networks.__class__.__name__ == db[keyVal].__class__.__name__

		if test1 and test2 and test3 and test4:
			db[keyVal] = saved_networks
			print('\t\tSucessful...')
			return True
		else:
			print('\t\tloaded input size : %d, current input size : %d'%(saved_networks.input_size, db[keyVal].input_size))
			print('\t\tloaded output size : %d, current output size : %d'%(saved_networks.output_dim, db[keyVal].output_dim))
			print('\t\tloaded depth : %d, current depth : %d'%(saved_networks.net_depth, db[keyVal].net_depth))
			print('\t\tloaded model : %s, current model: %s'%(saved_networks.__class__.__name__, db[keyVal].__class__.__name__))


	print('\t\tFailed...')
	return False


def export_pretrained_network(db, keyVal, stage_name, ignore_in_batch=False):
	if ignore_in_batch:
		if 'running_batch_mode' in db: return False

	ensure_path_exists('./pretrained')
	ensure_path_exists('./pretrained/' + db['data_name'])
	pth = './pretrained/' + db['data_name'] + '/' + db['data_name'] + '_' + stage_name + '.pk'
	pickle.dump( db[keyVal], open(pth, "wb" ) )


def load_db(db):
	if len(sys.argv) == 1: return db
	if sys.argv[1] == 'at_discovery':
		db['running_batch_mode'] = True
		return db

	fin = open(sys.argv[1],'r')
	cmds = fin.readlines()
	fin.close()
	db['running_batch_mode'] = True
	
	for i in cmds: 
		try: exec(i)
		except:
			print('Attempted to execuse command : %s'%i)
			import pdb; pdb.set_trace()
	return db

def save_to_lowest_end2end(db):
	ensure_path_exists('./pretrained')
	ensure_path_exists('./pretrained/' + db['data_name'])
	pth = './pretrained/' + db['data_name'] + '/' + db['data_name'] + '_best_' + db['knet'].__class__.__name__ + '.pk'
	lowest_error_list = './pretrained/' + db['data_name'] + '/' + 'lowest_error_list.txt'
	mutex = './pretrained/' + db['data_name'] + '/' + db['data_name'] + '_best_end2end.writing'
	tmp_writing = './pretrained/' + db['data_name'] + '/' + 'tmp.' + str(int(10000000*np.random.rand()))

	if path_list_exists([pth]):
		best_knet = pickle.load( open( pth, "rb" ) )
		if db['knet'].end2end_error < best_knet.end2end_error:
			pickle.dump( db['knet'] , open(tmp_writing, "wb" ) )
	else:
		pickle.dump( db['knet'] , open(tmp_writing, "wb" ) )


	while os.path.exists(mutex): 
		print('waiting .....')
		time.sleep(20*np.random.rand())

	create_file(mutex)
	fin = open(lowest_error_list,'a')
	fin.write('%.4f\n'%db['knet'].end2end_error)
	fin.close()
	if os.path.exists(tmp_writing): shutil.move(tmp_writing, pth)
	delete_file(mutex)


