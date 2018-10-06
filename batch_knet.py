#!/usr/bin/env python


import sys
import matplotlib
import numpy as np
import random
import itertools
import socket

sys.path.append('./src')
sys.path.append('./src/models')
sys.path.append('./src/optimizer')
sys.path.append('./tests')

if socket.gethostname().find('login') != -1:
	print('\nError : you cannot run program on login node.......\n\n')
	sys.exit()


np.set_printoptions(precision=4)
np.set_printoptions(threshold=np.nan)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)


#	This controls which dataset to run
from wine import *





#	Program Run
code = test_code()
code.run()
