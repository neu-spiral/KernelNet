

from dataset_manipulate import *
from terminal_print import *
from path_tools import *
from classifier import *
import time
import pickle

def save_results_to_text_file(db, result_path, fname, output_str):
	if 'running_batch_mode' not in db: 
		most_recent_result_path = result_path + fname
		mr_mutex = result_path + fname + '.writing_mutex'

		while os.path.exists(mr_mutex): time.sleep(20*np.random.rand())

		create_file(mr_mutex)
		fin = open(most_recent_result_path, 'w')
		fin.write(output_str)
		fin.close()
		delete_file(mr_mutex)

def save_result_to_history(db, result, result_path, fname, output_str):
	file_path = result_path + fname + '.pk'
	mutex = result_path + fname + '.writing'
	tmp_writing = result_path + fname + '.' + str(int(10000000*np.random.rand()))


	if path_list_exists([file_path]):
		past_runs = pickle.load( open( file_path, "rb" ) )

		past_runs['list_of_runs'].append(result)

		if past_runs['best_train_nmi']['train_nmi'] < result['train_nmi']:
			past_runs['best_train_nmi'] = result
			save_results_to_text_file(db, result_path, 'best_train_nmi.txt' , output_str)

		if past_runs['best_valid_nmi']['valid_nmi'] < result['valid_nmi']:
			past_runs['best_valid_nmi'] = result
			save_results_to_text_file(db, result_path, 'best_valid_nmi.txt' , output_str)

		if past_runs['best_train_loss']['train_loss'] > result['train_loss']:
			past_runs['best_train_loss'] = result
			save_results_to_text_file(db, result_path, 'lowest_train_loss.txt' , output_str)

		if past_runs['best_valid_loss']['valid_loss'] > result['valid_loss']:
			past_runs['best_valid_loss'] = result
			save_results_to_text_file(db, result_path, 'lowest_valid_loss.txt' , output_str)

	else:
		result['knet'] = db["knet"]

		past_runs = {}
		past_runs['list_of_runs'] = [result]
		past_runs['best_train_nmi'] = result
		past_runs['best_valid_nmi'] = result
		past_runs['best_train_loss'] = result
		past_runs['best_valid_loss'] = result

		save_results_to_text_file(db, result_path, 'best_train_nmi.txt' , output_str)
		save_results_to_text_file(db, result_path, 'best_valid_nmi.txt' , output_str)
		save_results_to_text_file(db, result_path, 'lowest_train_loss.txt' , output_str)
		save_results_to_text_file(db, result_path, 'lowest_valid_loss.txt' , output_str)

	pickle.dump( past_runs, open(tmp_writing, "wb" ) )


	while os.path.exists(mutex): time.sleep(20*np.random.rand())
	create_file(mutex)
	shutil.move(tmp_writing, file_path)
	delete_file(mutex)

def AE_validate(db):
	#	get loss objective
	[db['train_loss'], db['train_hsic'], db['train_AE_loss'], φ_x] = db['knet'].get_current_state(db, db['train_data'].X_Var)
	[db['valid_loss'], db['valid_hsic'], db['valid_AE_loss'], φ_x] = db['knet'].get_current_state(db, db['valid_data'].X_Var)

	#	get training nmi
	current_label = kmeans(db['num_of_clusters'], db['U'])
	db['train_nmi'] = normalized_mutual_info_score(current_label, db['train_data'].Y)

	#	get validation nmi
	[x_hat, U] = db['knet'](db['valid_data'].X_Var)		# <- update this to be used in opt_K
	current_label = kmeans(db['num_of_clusters'], U)
	db['valid_nmi'] = normalized_mutual_info_score(current_label, db['valid_data'].Y)


	#	setting up paths
	result_path = './results/' + db["data_name"] + '/'
	most_recent_result_path = result_path + 'most_recent.txt'
	ensure_path_exists('./results')
	ensure_path_exists(result_path)
	output_str = ''

	output_str += '\tBasic settings that does not change often\n'
	packet_1 = {}
	list_of_keys = ['data_name', 'center_and_scale', 'pretrain_repeats', 'batch_size', 'num_of_clusters', 'use_Degree_matrix', 'cuda']
	fill_dictionary(db, packet_1, list_of_keys)
	output_str += dictionary_to_str(packet_1)


	output_str += '\tKey Settings'
	packet_2 = {}
	list_of_keys = ['output_dim', "kernel_net_depth", "σ_ratio", "λ_ratio", 'λ_obj_ratio', 'λ']
	fill_dictionary(db, packet_2, list_of_keys)
	output_str += '\n' + dictionary_to_str(packet_2)

	output_str += '\tCode class components'
	packet_3 = {}
	list_of_keys = ['kernel_model','opt_K_class','opt_U_class','exit_cond','validate_function']
	fill_dictionary(db, packet_3, list_of_keys)
	output_str += '\n' + dictionary_to_str(packet_3)


	output_str += '\tTime Results'
	packet_4 = {}
	list_of_keys = ['train_time', 'end2end_time','pretrain_time', 'itr_til_converge']
	db['train_time'] = db['knet'].train_time
	db['end2end_time'] = db['knet'].end2end_time
	db['pretrain_time'] = db['knet'].pretrain_time
	db['itr_til_converge'] = db['knet'].itr_til_converge

	fill_dictionary(db, packet_4, list_of_keys)
	output_str += '\n' + dictionary_to_str(packet_4)


	output_str += '\tResults of Initial States '
	packet_5 = {}
	list_of_keys = ['init_spectral_nmi', 'init_AE+Kmeans_nmi', 'initial_loss', 'initial_hsic', 'initial_AE_loss']
	fill_dictionary(db, packet_5, list_of_keys)
	output_str += '\n' + dictionary_to_str(packet_5)


	#	objective results
	output_str += '\tObjective Results'
	packet_6 = {}
	list_of_keys = ['train_loss','train_hsic','train_AE_loss','valid_loss','valid_hsic','valid_AE_loss']
	fill_dictionary(db, packet_6, list_of_keys)
	output_str += '\n' + dictionary_to_str(packet_6)

	#	NMI results
	output_str += '\tNMI Results'
	packet_7 = {}
	list_of_keys = ['train_nmi','valid_nmi']
	fill_dictionary(db, packet_7, list_of_keys)
	output_str += '\n' + dictionary_to_str(packet_7)

	result = {**packet_1, **packet_2, **packet_3, **packet_4, **packet_5, **packet_6, **packet_7}
	print(output_str)


	save_results_to_text_file(db, result_path,  'most_recent.txt', output_str)
	save_result_to_history(db, result, result_path, 'run_history', output_str)

