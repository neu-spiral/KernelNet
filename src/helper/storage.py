
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
			test4 = itm.dataType == db[keyVal].dataType

			if test1 and test2 and test3 and test4:
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




#def write_rand_file(wpath):
#	print('Waiting for write access....')
#	while os.path.exists(wpath): 
#		print '.',
#		time.sleep(65*np.random.rand())
#
#	v = str(int(10000000*np.random.rand()))
#	fin = open(wpath,'w')
#	fin.write(v)
#	fin.close()
#
#
#def store_best_results(db, my_results):	
#	v = str(int(10000000*np.random.rand()))
#
#	ensure_path_exists(db['best_path'])
#	ensure_path_exists(db['best_path'] + 'saved_weights/')
#	fpath = db['best_path'] + db['data_name'] + '.pk'
#	fpath_tmp = db['best_path'] + db['data_name'] + '_' + v + '.pk'
#	wpath = db['best_path'] + db['data_name'] + '_writing.tmp'
#
#	write_rand_file(wpath)
#	
#	try:
#		all_results = safe_pickle_load(fpath)
#
#		if all_results['best_kernel']['NMI_avg'] < my_results['NMI_avg']:
#			#my_results['kernel_net'] = db['kernel_net']
#			all_results['best_kernel'] = my_results
#			copy_best_weights(db)
#
#		all_results['result_list'].append(my_results)
#		safe_pickle_dump(fpath_tmp, all_results)
#		
#		shutil.move(fpath_tmp, fpath)
#	except:
#		#my_results['kernel_net'] = db['kernel_net']
#		all_results = {}	
#		all_results['best_kernel'] = my_results
#		all_results['result_list'] = [my_results]
#		safe_pickle_dump(fpath, all_results)
#		copy_best_weights(db)
#		
#	delete_file(wpath)
#
#
