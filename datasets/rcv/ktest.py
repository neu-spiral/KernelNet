#!/usr/bin/env python

import sklearn
import numpy as np
import sklearn.metrics
import sklearn.cluster
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize			# version : 0.17
from sklearn.metrics.cluster import normalized_mutual_info_score



def eig_solver(L, k, mode='smallest'):
	#L = ensure_matrix_is_numpy(L)
	eigenValues,eigenVectors = np.linalg.eigh(L)

	if mode == 'smallest':
		U = eigenVectors[:, 0:k]
		U_λ = eigenValues[0:k]
	elif mode == 'largest':
		n2 = len(eigenValues)
		n1 = n2 - k
		U = eigenVectors[:, n1:n2]
		U_λ = eigenValues[n1:n2]
	else:
		raise ValueError('unrecognized mode : ' + str(mode) + ' found.')
	
	return [U, U_λ]

def compute_inverted_Degree_matrix(M):
	return np.diag(1.0/np.sqrt(M.sum(axis=1)))

def normalize_U(U):
	return normalize(U, norm='l2', axis=1)

X = np.loadtxt('rcv.csv', delimiter=',', dtype=np.float64)			
Y = np.loadtxt('rcv_label.csv', delimiter=',', dtype=np.int32)
#K = X.dot(X.T)
K = sklearn.metrics.pairwise.polynomial_kernel(X, Y=None, degree=3, gamma=None, coef0=1)
D = compute_inverted_Degree_matrix(K)
DKD = D.dot(K).dot(D)


[U, U_λ] = eig_solver(DKD, 4, mode='largest')
U_normed = normalize_U(U)
labels = KMeans(4).fit_predict(U_normed)
nmi = normalized_mutual_info_score(labels, Y)
print(nmi)
import pdb; pdb.set_trace()
