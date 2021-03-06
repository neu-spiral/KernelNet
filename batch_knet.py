#!/usr/bin/env python


import sys
import os
import matplotlib
import numpy as np
import random
import itertools
import socket

sys.path.append('./src')
sys.path.append('./src/helper')
sys.path.append('./src/models')
sys.path.append('./src/optimizer')
sys.path.append('./tests')


np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
os.system('tabs -4')

#	This controls which dataset to run
#from moon import *
#from spiral import *
#from wine import *
#from cancer import *
#from face import *
from rcv import *
#from car import *

#from RFF_moon import *
#from RFF_spiral import *
#from RFF_wine import *
#from RFF_cancer import *
#from RFF_face import *
#from RFF_rcv import *




#	Program Run
code = test_code()
#code.run_10_fold()
#code.run_batch()
#code.run_train_test_batch(0.2)
code.run_subset_and_rest_batch()
