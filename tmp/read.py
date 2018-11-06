#!/usr/bin/env python

import os


file_in_tmp = os.listdir('./')
for i in file_in_tmp:
	if i.find('.out') != -1:
		fin = open(i,'r')
		lines = fin.readlines()
		fin.close()

		print(lines[-1].strip())


