#!/usr/bin/env python

import shutil, errno
import sys
import os


mydir = "../kernel_net"



def copyFolder(src, dst):
	try:
		shutil.copytree(src, dst)
	except OSError as exc: # python >2.5
		if exc.errno == errno.ENOTDIR:
			shutil.copy(src, dst)
		else: raise




L = ['b','c','d','e','f']						#	1
L += ['g','h','i','j','k','l']					#	2
L += ['m','n','o','p','q','r']					#	3
L += ['s','t','u','v','w','x']					#	4
L += ['a1','a2','a3','a4','a5','a6']			#	5
L += ['b1','b2','b3','b4','b5','b6']			#	6
L += ['c1','c2','c3','c4','c5','c6']			#	7
L += ['d1','d2','d3','d4','d5','d6']			#	8
L += ['e1','e2','e3','e4','e5','e6']			#	9
L += ['f1','f2','f3','f4','f5','f6']			#	10
L += ['g1','g2','g3','g4','g5','g6']			#	11
L += ['h1','h2','h3','h4','h5','h6']			#	12
L += ['i1','i2','i3','i4','i5','i6']			#	13
L += ['j1','j2','j3','j4','j5','j6']			#	14
L += ['k1','k2','k3','k4','k5','k6']			#	15
L += ['l1','l2','l3','l4','l5','l6']			#	16
L += ['m1','m2','m3','m4','m5','m6']			#	17
L += ['n1','n2','n3','n4','n5','n6']			#	18
L += ['o1','o2','o3','o4','o5','o6']			#	19
L += ['p1','p2','p3','p4','p5','p6']			#	20
L += ['q1','q2','q3','q4','q5','q6']			#	21
L += ['r1','r2','r3','r4','r5','r6']			#	22
L += ['s1','s2','s3','s4','s5','s6']			#	23
L += ['t1','t2','t3','t4','t5','t6']			#	24
L += ['u1','u2','u3','u4','u5','u6']			#	25





if len(sys.argv) > 1:
	for i in range(int(sys.argv[1])):
		copyFolder(mydir, mydir + '_' + L[i])


