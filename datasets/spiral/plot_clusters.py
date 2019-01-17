#!/usr/bin/env python


from matplotlib import pyplot as plt
from numpy import genfromtxt
import numpy as np

label = 'spiral'

X = genfromtxt(label + '.csv', delimiter=',')
Y = genfromtxt(label + '_label.csv', delimiter=',')

Uq_a = np.unique(Y)
color_list = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']


for m in range(len(Uq_a)):
	g = X[Y == Uq_a[m]]
	plt.plot(g[:,0], g[:,1], color_list[m] + 'o')

plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
plt.show()

