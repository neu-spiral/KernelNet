
from AE_RFF import *
from MLP import *
from AE_validate import *
from storage import *
from DManager import *
from opt_Kernel import *



def face_raw_data_RFF():
	db = {}
	# Data info
	dn = db["data_name"]="face"
	db["center_and_scale"]=True
	db["data_path"]="./datasets/" + dn + "/"
	db["orig_data_file_name"]="./datasets/" + dn + "/" + dn + ".csv"
	db["orig_label_file_name"]="./datasets/" + dn + "/" + dn + "_label.csv"
	db['data_folder']  = db['data_path'] 
	db['train_data_file_name']  = db['data_folder'] + db['data_name'] + '.csv'
	db['train_label_file_name']  = db['data_folder'] + db['data_name'] + '_label.csv'
	db['test_data_file_name']  = ''
	db['test_label_file_name']  = ''
	db['10_fold_id'] = 0
	db['cuda'] = False #torch.cuda.is_available()
	
	# debug tracking
	db['objective_tracker'] = []
	
	# hyperparams
	db["output_dim"]=27
	db["kernel_net_depth"]=3
	db["mlp_width"]=1
	db["σ_ratio"]=1.0
	db["λ_ratio"]=2
	db['pretrain_repeats'] = 4
	db['batch_size'] = 5
	db['num_of_clusters'] = 20
	db['use_Degree_matrix'] = True
	db['use_U_normalize'] = True
	
	# code
	db['kernel_model'] = AE_RFF
	db['opt_K_class'] = opt_K
	db['opt_U_class'] = opt_U
	db['exit_cond'] = exit_cond
	db['validate_function'] = AE_validate

	return db
