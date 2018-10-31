
from AE import *
from MLP import *
from AE_validate import *
from storage import *
from DManager import *
from opt_Kernel import *



def moon_raw_data():
	db = {}
	# Data info
	dn = db["data_name"]="moon"
	db["center_and_scale"]=True
	db["data_path"]="./datasets/" + dn + "/"
	db["orig_data_file_name"]="./datasets/" + dn + "/" + dn + ".csv"
	db["orig_label_file_name"]="./datasets/" + dn + "/" + dn + "_label.csv"
	db['data_folder']  = db['data_path'] 
	db["train_data_file_name"]="./datasets/moon/train_test/train.csv"
	db["train_label_file_name"]="./datasets/moon/train_test/train_label.csv"
	db["test_data_file_name"]="./datasets/moon/train_test/test.csv"
	db["test_label_file_name"]="./datasets/moon/train_test/test_label.csv"

	db['10_fold_id'] = 0
	db['cuda'] = False #torch.cuda.is_available()
	
	# debug tracking
	db['objective_tracker'] = []
	
	# hyperparams
	db["output_dim"]=2
	db["kernel_net_depth"]=3
	db["mlp_width"]= 20
	db["σ_ratio"]=0.1
	db["λ_ratio"]=2
	db['pretrain_repeats'] = 4
	db['batch_size'] = 5
	db['num_of_clusters'] = 2
	db['use_Degree_matrix'] = True
	db['use_U_normalize'] = True
	
	# code
	db['kernel_model'] = MLP	#AE
	db['opt_K_class'] = opt_K
	db['opt_U_class'] = opt_U
	db['exit_cond'] = exit_cond
	db['validate_function'] = AE_validate

	return db
