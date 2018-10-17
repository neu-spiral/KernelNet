#!/usr/bin/env python

from subprocess import call
import dataset_manipulate 
import random
import string
from termcolor import colored
from path_tools import *
import numpy as np
import itertools
import types
import socket



class test_parent():
	def __init__(self,db):
		self.db = db
		db['data_path'] = './datasets/' + db['data_name'] + '/'
		db['orig_data_file_name']  = db['data_path'] +  db['data_name'] + '.csv'
		db['orig_label_file_name'] = db['data_path'] +  db['data_name'] + '_label.csv'

		tmp_path = './tmp/' + db['data_name'] + '/'
		db_output_path = tmp_path + 'db_files/'
		batch_output_path = tmp_path + 'batch_outputs/'

		ensure_path_exists('./tmp')
		ensure_path_exists(tmp_path)
		ensure_path_exists(db_output_path)
		ensure_path_exists(batch_output_path)

		remove_files(tmp_path)
		remove_files(db_output_path)
		remove_files(batch_output_path)

	def run(self):
		db = self.db
		dataset_manipulate.gen_10_fold_data(db)

		output_list = self.parameter_ranges()
		every_combination = list(itertools.product(*output_list))

		for count, single_instance in enumerate(every_combination):
			[output_dim, kernel_net_depth, σ_ratio, extra_repeat, λ_ratio, id_10_fold] = single_instance

			db['running_batch_mode'] = True
			db['10_fold_id'] = id_10_fold
			db['output_dim'] = output_dim
			db['kernel_net_depth'] = kernel_net_depth
			db['σ_ratio'] = float(σ_ratio)
			db['λ_ratio'] = float(λ_ratio)
			db['data_folder']  = db['data_path'] +  '10_fold/split_' + str(id_10_fold) + '/'
			db['train_data_file_name']  = db['data_path'] +  '10_fold/split_' + str(id_10_fold) + '/train.csv'
			db['train_label_file_name']  = db['data_path'] +  '10_fold/split_' + str(id_10_fold) + '/train_label.csv'
			db['test_data_file_name']  = db['data_path'] +  '10_fold/split_' + str(id_10_fold) + '/test.csv'
			db['test_label_file_name']  = db['data_path'] +  '10_fold/split_' + str(id_10_fold) + '/test_label.csv'


			export_db = self.output_db_to_text(id_10_fold, count)
			self.export_bash_file(id_10_fold, db['data_name'], export_db)

			if socket.gethostname().find('login') != -1:
				call(["sbatch", "execute_combined.bash"])
			else:
				call(["bash", "./execute_combined.bash"])


	def output_db_to_text(self, i, count):
		db = self.db
		db['db_file']  = './tmp/' + db['data_name'] + '/db_files/' + db['data_name'] + '_' +  str(i) + '_' + str(count) + '.txt'

		fin = open(db['db_file'], 'w')

		for i,j in db.items():
			if type(j) == str:
				fin.write('db["' + i + '"]="' + str(j) + '"\n')
			elif type(j) == bool:
				fin.write('db["' + i + '"]=' + str(j) + '\n')
			elif type(j) == type:
				fin.write('db["' + i + '"]=' + j.__name__ + '\n')
			elif type(j) == types.FunctionType:
				fin.write('db["' + i + '"]=' + j.__name__ + '\n')
			elif type(j) == float:
				fin.write('db["' + i + '"]=' + str(j) + '\n')
			elif type(j) == int:
				fin.write('db["' + i + '"]=' + str(j) + '\n')
			elif j is None:
				fin.write('db["' + i + '"]=None\n')
			else:
				print('unrecognized type : ' + str(type(j)) + ' found.')
				import pdb; pdb.set_trace()
				raise ValueError('unrecognized type : ' + str(type(j)) + ' found.')

		fin.close()
		return db['db_file']



	def export_bash_file(self, i, test_name, export_db):
		run_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(2))

		cmd = ''
		cmd += "#!/bin/bash\n"
		cmd += "\n#set a job name  "
		cmd += "\n#SBATCH --job-name=%d_%s_%s"%(i, test_name, run_name)
		cmd += "\n#################  "
		cmd += "\n#a file for job output, you can check job progress"
		cmd += "\n#SBATCH --output=./tmp/%s/batch_outputs/%d_%s_%s.out"%(test_name, i, test_name, run_name)
		cmd += "\n#################"
		cmd += "\n# a file for errors from the job"
		cmd += "\n#SBATCH --error=./tmp/%s/batch_outputs/%d_%s_%s.err"%(test_name, i, test_name, run_name)
		cmd += "\n#################"
		cmd += "\n#time you think you need; default is one day"
		cmd += "\n#in minutes in this case, hh:mm:ss"
		cmd += "\n#SBATCH --time=24:00:00"
		cmd += "\n#################"
		cmd += "\n#number of tasks you are requesting"
		cmd += "\n#SBATCH -N 1"
		cmd += "\n#SBATCH --exclusive"
		cmd += "\n#################"
		cmd += "\n#partition to use"
		cmd += "\n#SBATCH --partition=general"
		cmd += "\n#SBATCH --mem=120Gb"
		cmd += "\n#################"
		cmd += "\n#number of nodes to distribute n tasks across"
		cmd += "\n#################"
		cmd += "\n"
		cmd += "\npython knet.py " + export_db
		
		fin = open('execute_combined.bash','w')
		fin.write(cmd)
		fin.close()

