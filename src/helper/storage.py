
from path_tools import *
import pickle

def import_pretrained_network(db, keyVal, stage_name):
	ensure_path_exists('./pretrained')
	path_list = ['./pretrained/' + db['data_name'] + '_' + stage_name + '.pk']

	print('\tLoading %s from %s...'%(stage_name, keyVal))
	if path_list_exists(path_list):
		list_of_networks = pickle.load( open( path_list[0], "rb" ) )

		for itm in list_of_networks:
			test1 = itm.input_size == db[keyVal].input_size
			test2 = itm.output_dim == db[keyVal].output_dim
			test3 = itm.net_depth == db[keyVal].net_depth

			if test1 and test2 and test3:
				db[keyVal] = itm
				print('\t\tSucessful...')
				return True


	print('\t\tFailed...')
	return False


def export_pretrained_network(db, keyVal, stage_name):
	ensure_path_exists('./pretrained')
	pth = './pretrained/' + db['data_name'] + '_' + stage_name + '.pk'
	path_list = [pth]

	if path_list_exists(path_list):
		list_of_networks = pickle.load( open( path_list[0], "rb" ) )

		for p, itm in enumerate(list_of_networks):
			test1 = itm.input_size == db[keyVal].input_size
			test2 = itm.output_dim == db[keyVal].output_dim
			test3 = itm.net_depth == db[keyVal].net_depth

			if test1 and test2 and test3:
				list_of_networks[p] = db[keyVal]
				pickle.dump( list_of_networks, open(pth, "wb" ) )
				return 
	else:
		list_of_networks = []


	list_of_networks.append(db[keyVal])
	pickle.dump( list_of_networks, open(pth, "wb" ) )





