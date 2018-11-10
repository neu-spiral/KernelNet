#!/usr/bin/env python

import os
import shutil


reply = str(input('With [R]FF or [N]ot ?'+' (R/[N]): ')).lower().strip()

if reply == 'r':
	files = os.listdir('./')
	for i in files:
		if os.path.isdir(i):
			if os.path.exists('%s/%s_best_AE_RFF.pk'%(i,i)):
				fpth = '%s/%s_best_AE_RFF.pk'%(i,i)
			elif os.path.exists('%s/%s_best_MLP_RFF.pk'%(i,i)):
				fpth = '%s/%s_best_MLP_RFF.pk'%(i,i)
			else:
				fpth = '%s/%s_best_AE_RFF.pk'%(i,i)
				fpth2 = '%s/%s_best_MLP_RFF.pk'%(i,i)
				print('Paths %s and %s are not found...'%(fpth, fpth2))
				import pdb; pdb.set_trace()	

			print('Copying %s to %s...'%(fpth, '%s/%s_rbm.pk'%(i,i)))
			shutil.copy(fpth, '%s/%s_rbm.pk'%(i,i))
			print('Copying %s to %s...'%(fpth, '%s/%s_end2end.pk'%(i,i)))
			shutil.copy(fpth, '%s/%s_end2end.pk'%(i,i))
else:
	files = os.listdir('./')
	for i in files:
		if os.path.isdir(i):
			if os.path.exists('%s/%s_best_AE.pk'%(i,i)):
				fpth = '%s/%s_best_AE.pk'%(i,i)
			elif os.path.exists('%s/%s_best_MLP.pk'%(i,i)):
				fpth = '%s/%s_best_MLP.pk'%(i,i)
			else:
				fpth = '%s/%s_best_AE.pk'%(i,i)
				fpth2 = '%s/%s_best_MLP.pk'%(i,i)
				print('Paths %s and %s are not found...'%(fpth, fpth2))
				import pdb; pdb.set_trace()	
			
			print('Copying %s to %s...'%(fpth, '%s/%s_rbm.pk'%(i,i)))
			shutil.copy(fpth, '%s/%s_rbm.pk'%(i,i))
			print('Copying %s to %s...'%(fpth, '%s/%s_end2end.pk'%(i,i)))
			shutil.copy(fpth, '%s/%s_end2end.pk'%(i,i))

