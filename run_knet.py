#!/usr/bin/env python

import sys
sys.path.append('./src')
sys.path.append('./src/helper')

from subprocess import call
from path_tools import *
import random
import string
import socket


run_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(2))
test_name = 'wine'
#test_name = 'cancer'
#test_name = 'face'




pretrain_path = './pretrained/' + test_name + '/'
remove_files(pretrain_path)
tmp_path = './tmp/' + test_name + '/'
tmp_path_2 = './tmp/' + test_name + '/batch_outputs'
tmp_path_3 = './tmp/' + test_name + '/db_files'

ensure_path_exists(tmp_path)
ensure_path_exists(tmp_path_2)
ensure_path_exists(tmp_path_3)


cmd = ''
cmd += "#!/bin/bash\n"
cmd += "\n#set a job name  "
cmd += "\n#SBATCH --job-name=%s_%s"%(test_name, run_name)
cmd += "\n#################  "
cmd += "\n#a file for job output, you can check job progress"
cmd += "\n#SBATCH --output=./tmp/%s/batch_outputs/%s_%s.out"%(test_name, test_name, run_name)
cmd += "\n#################"
cmd += "\n# a file for errors from the job"
cmd += "\n#SBATCH --error=./tmp/%s/batch_outputs/%s_%s.err"%(test_name, test_name, run_name)
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
cmd += "\npython knet.py "

fin = open('execute_combined.bash','w')
fin.write(cmd)
fin.close()

call(["sbatch", "execute_combined.bash"])


