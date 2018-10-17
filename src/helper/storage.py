
from path_tools import *
import numpy as np
import pickle
import shutil

def import_pretrained_network(db, keyVal, stage_name):
	if 'running_batch_mode' in db: return False

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
	if 'running_batch_mode' in db: return

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

def save_results_to_text_file(db, most_recent_result_path, output_str):
	if 'running_batch_mode' not in db: 
		fin = open(most_recent_result_path, 'w')
		fin.write(output_str)
		fin.close()

def save_result_to_history(db, result, run_history_path):
	file_path = run_history_path + '.pk'
	mutex = run_history_path + '.writing'
	tmp_writing = run_history_path + '.' + str(int(10000000*np.random.rand()))


	if path_list_exists([file_path]):
		past_runs = pickle.load( open( file_path, "rb" ) )

		past_runs['list_of_runs'].append(result)

		if past_runs['best_train_nmi']['train_nmi'] < result['train_nmi']:
			past_runs['best_train_nmi'] = result

		if past_runs['best_valid_nmi']['valid_nmi'] < result['valid_nmi']:
			past_runs['best_valid_nmi'] = result

		if past_runs['best_train_loss']['train_loss'] > result['train_loss']:
			past_runs['best_train_loss'] = result

		if past_runs['best_valid_loss']['valid_loss'] > result['valid_loss']:
			past_runs['best_valid_loss'] = result

	else:
		result['knet'] = db["knet"]

		past_runs = {}
		past_runs['list_of_runs'] = [result]
		past_runs['best_train_nmi'] = result
		past_runs['best_valid_nmi'] = result
		past_runs['best_train_loss'] = result
		past_runs['best_valid_loss'] = result


	pickle.dump( past_runs, open(tmp_writing, "wb" ) )


	while os.path.exists(mutex): time.sleep(20*np.random.rand())
	create_file(mutex)
	shutil.move(tmp_writing, file_path)
	delete_file(mutex)

