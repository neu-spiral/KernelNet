#!/usr/bin/env python

import os
import numpy as np


file_in_tmp = os.listdir('./')
l = []
item_list = []
for i in file_in_tmp:
	if i.find('.out') != -1:
		fin = open(i,'r')
		lines = fin.readlines()
		fin.close()

		last_line = lines[-1].strip()
		items = last_line.split('\t')
		item_list.append(items)
		l.append(items[-1])
		print(last_line)

l.sort(reverse=True)
least_val = (l[0:10][-1])

min_list = []
for item in item_list:
	if item[-1] >= least_val: 
		min_list.append(item)

zipList = zip(*min_list)
results = []
for item in list(zipList):
	try: 
		#M = np.mean(item)
		avg = np.mean([float(x) for x in item])
		std = np.std([float(x) for x in item])
		single_result = '%.2f±%.2f'%(avg,std)
	except: single_result = 'N/A'
	results.append(single_result)

out_str = ''
for i in results:
	out_str += '%s\t'%i

print(out_str)
import pdb; pdb.set_trace()
