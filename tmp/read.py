#!/usr/bin/env python

import os
import numpy as np


file_in_tmp = os.listdir('./')
l = []
item_list = []
spectral_values = []
kmeans_values = []
init_loss = []
init_hsic = []
init_AE = []
for i in file_in_tmp:
	if i.find('.out') != -1:
		fin = open(i,'r')
		lines = fin.readlines()
		fin.close()

		for j in lines:
			key = j.split(':')[0]
			if key.find('init_spectral_nmi') != -1:
				val = float(j.split(':')[1])
				spectral_values.append( val )
			if key.find('init_AE+Kmeans_nmi') != -1:
				val = float(j.split(':')[1])
				kmeans_values.append( val )
			if key.find('initial_loss') != -1:
				val = float(j.split(':')[1])
				init_loss.append( val )
			if key.find('initial_hsic') != -1:
				val = float(j.split(':')[1])
				init_hsic.append( val )
			if key.find('initial_AE_loss') != -1:
				val = float(j.split(':')[1])
				init_AE.append( val )


		last_line = lines[-1].strip()
		if last_line.find('opt_K') == -1 and last_line.find('sm_opt_K') == -1: continue

		items = last_line.split('\t')
		item_list.append(items)
		l.append(items[-1])
		print('%s\t%s'%(i,last_line))

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
	except: single_result = item[0]
	results.append(single_result)

out_str = ''
for i in results:
	out_str += '%s\t'%i

out_str = '%s%.3f\t'%(out_str, np.max(init_hsic))
out_str = '%s%.3f\t'%(out_str, np.max(init_AE))
out_str = '%s%.3f\t'%(out_str, np.max(init_loss))
out_str = '%s%.3f\t'%(out_str, np.min(spectral_values))
out_str = '%s%.3f\t'%(out_str, np.min(kmeans_values))

print(out_str)
print(init_loss)
