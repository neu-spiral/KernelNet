#!/usr/bin/env python

import sys
import matplotlib
import numpy as np
import random
import itertools
import socket

sys.path.append('./src')
sys.path.append('./tests')
sys.path.append('./src/helper')

if socket.gethostname().find('login') != -1:
	print('\nError : you cannot run program on login node.......\n\n')
	sys.exit()


np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)



#	Loading DB	
def load_db():
	db = {}
	fin = open(sys.argv[1],'r')
	cmds = fin.readlines()
	fin.close()
	
	for i in cmds: exec(i)
	return db



db = load_db()
import pdb; pdb.set_trace()
#split_data_into_train_validation(db)
#initialize_autoencoder(db)
#train_kernel_net(db)


