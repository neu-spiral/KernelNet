

from dataset_manipulate import *
from terminal_print import *
from path_tools import *
from classifier import *
import pickle

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

