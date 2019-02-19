#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt
from numpy import genfromtxt



def gen_spiral(N,D,K, valid):
	X = np.zeros((N*K,D)) # data matrix (each row = single example)
	y = np.zeros(N*K, dtype='uint8') # class labels
	for j in range(K):
	  ix = range(N*j,N*(j+1))
	  r = np.linspace(0.3,2,N) # radius
	  t = np.linspace(j*4,(j+0.6)*4,N) + np.random.randn(N)*0.16 # theta
	  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
	  y[ix] = j
	
	
	labels = np.vstack((np.ones((N,1)), 2*np.ones((N,1)), 3*np.ones((N,1))))
	np.savetxt('spiral_' + valid + '.csv', X, delimiter=',', fmt='%.3f') 
	np.savetxt('spiral_label' + valid + '.csv', labels, delimiter=',', fmt='%d') 
	
	plt.scatter(X[0:N, 0], X[0:N, 1], c='blue')
	plt.scatter(X[N:2*N, 0], X[N:2*N, 1], c='green')
	plt.scatter(X[2*N:3*N, 0], X[2*N:3*N, 1], c='red')
	plt.show()

def load_spiral(valid):
	X = genfromtxt('spiral_arm' + valid + '.csv', delimiter=',')
	E = len(X)/3
	E2 = E*2
	E3 = E*3
	
	plt.scatter(X[0:E, 0], X[0:E, 1], c='blue')
	plt.scatter(X[E:E2, 0], X[E:E2, 1], c='green')
	plt.scatter(X[E2:E3, 0], X[E2:E3, 1], c='red')
	plt.show()

if __name__ == '__main__':
	N = 1000 # number of points per class
	D = 2 # dimensionality
	K = 3 # number of classes

	#gen_spiral(N,D,K, '_validation')
	gen_spiral(N,D,K, '')
	#load_spiral('_validation')
	#load_spiral('_6')
