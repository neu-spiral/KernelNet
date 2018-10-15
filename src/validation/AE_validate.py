
from path_tools import *


def AE_validate(db):
	result_path = './results/' + db["data_name"] + '/'
	most_recent_result_path = result_path + 'most_recent.txt'
	ensure_path_exists('./results')
	ensure_path_exists(result_path)
	output_str = ''

	#	initial settings
	packet_1 = {}
	packet_1['data_name'] = db["data_name"]
	packet_1['center_and_scale'] = db["center_and_scale"]
	packet_1['output_dim'] = db["output_dim"]
	packet_1["kernel_net_depth"] = db["kernel_net_depth"]
	packet_1["σ_ratio"] = db["σ_ratio"]
	packet_1["λ_ratio"] = db["λ_ratio"]
	packet_1['pretrain_repeats'] = db['pretrain_repeats']
	packet_1['batch_size'] = db['batch_size']
	packet_1['num_of_clusters'] = db['num_of_clusters']
	packet_1['use_Degree_matrix'] = db['use_Degree_matrix']
	packet_1['λ_obj_ratio'] = db['λ_obj_ratio']
	packet_1['λ'] = db['λ']


	#	Components used
	packet_2 = {}
	packet_2['kernel_model'] = db['kernel_model'].__name__
	packet_2['opt_K_class'] = db['opt_K_class'].__name__
	packet_2['opt_U_class'] = db['opt_U_class'].__name__
	packet_2['exit_cond_class'] = db['exit_cond_class'].__name__
	packet_2['validate_class'] = db['validate_class'].__name__
	output_str += '\n' + dictionary_to_str(packet_2)


	#	results
	packet_3 = {}
	packet_3['knet'] = db["knet"]
	packet_3['train_time'] = db['knet'].train_time
	packet_3['end2end_time'] = db['knet'].end2end_time
	packet_3['pretrain_time'] = db['knet'].pretrain_time
	packet_3['test_nmi'] = db['test_nmi']
	packet_3['current_loss'] = db['current_loss']
	packet_3['current_hsic'] = db['current_hsic']
	packet_3['current_AE_loss'] = db['current_AE_loss']
	output_str += '\n' + dictionary_to_str(packet_3)

	result = {**packet_1, **packet_2, **packet_3}
	print(output_str)


	#	Save results to text file
	if running_batch_mode not in db: 
		fin = open(most_recent_result_path, 'w')
		fin.write(output_str)
		fin.close()
