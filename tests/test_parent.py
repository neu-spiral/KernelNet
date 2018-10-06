#!/usr/bin/env python

import dataset_manipulate 
from termcolor import colored
import numpy as np


class test_parent():
	def __init__(self,db):
		self.db = db
		db['data_path'] = './datasets/' + db['data_name'] + '/'
		

	def run(self):
		dataset_manipulate.gen_10_fold_data(self.db)




