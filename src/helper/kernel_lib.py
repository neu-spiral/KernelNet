#!/usr/bin/env python3

import sklearn.metrics
import numpy as np
from sklearn.preprocessing import normalize			# version : 0.17
from format_conversion import *

def Y_2_allocation(Y):
	i = 0
	allocation = np.array([])
	for m in range(Y.shape[0]):
		allocation = np.hstack((allocation, np.where(Y[m] == 1)[0][0]))
		i += 1

	return allocation


def Allocation_2_Y(allocation):
	
	N = np.size(allocation)
	unique_elements = np.unique(allocation)
	num_of_classes = len(unique_elements)
	class_ids = np.arange(num_of_classes)

	i = 0
	Y = np.zeros(num_of_classes)
	for m in allocation:
		class_label = np.where(unique_elements == m)[0]
		a_row = np.zeros(num_of_classes)
		a_row[class_label] = 1
		Y = np.hstack((Y, a_row))

	Y = np.reshape(Y, (N+1,num_of_classes))
	Y = np.delete(Y, 0, 0)

	return Y

def getLaplacian(db, data, σ, H=None):
	[L, Dinv] = normalized_rbk_sklearn(data, σ)

	if H is not None:
		L = center_matrix(db, L)

	return [L, Dinv]


def Kx_D_given_W(db, setX=None, setW=None):
	if setX is None: outX = db['Dloader'].X.dot(db['W'])
	else: outX = setX.dot(db['W'])
	
	if setW is None: outX = db['Dloader'].X.dot(db['W'])
	else: outX = db['Dloader'].X.dot(setW)

	if db['kernel_type'] == 'rbf':
		Kx = rbk_sklearn(outX, db['Dloader'].σ)
	elif db['kernel_type'] == 'linear':
		Kx = outX.dot(outX.T)
	elif db['kernel_type'] == 'polynomial':
		poly_sklearn(outX, db['poly_power'], db['poly_constant'])


	np.fill_diagonal(Kx, 0)			#	Set diagonal of adjacency matrix to 0
	D = compute_inverted_Degree_matrix(Kx)
	return [Kx, D]


def poly_sklearn(data, p, c):
	poly = sklearn.metrics.pairwise.polynomial_kernel(data, degree=p, coef0=c)
	return poly

def normalized_rbk_sklearn(X, σ):
	X = ensure_matrix_is_numpy(X)
	Kx = rbk_sklearn(X, σ)       	
	np.fill_diagonal(Kx, 0)			#	Set diagonal of adjacency matrix to 0
	Dinv = compute_inverted_Degree_matrix(Kx)

	KD = Kx - Dinv
	return [KD, Dinv]

	#DKD = Dinv.dot(Kx).dot(Dinv)
	#return [DKD, Dinv]


def rbk_sklearn(data, σ):
	γ = 1.0/(2*σ*σ)
	rbk = sklearn.metrics.pairwise.rbf_kernel(data, gamma=γ)
	return rbk

def Ku_kernel(labels):
	Y = Allocation_2_Y(labels)
	Ky = Y.dot(Y.T)
	
	return Ky

def center_matrix(db,M):
	N = M.shape[0]
	if 'H' in db:
		if db['H'].shape[0] != M.shape[0]:
			db['H'] = H = np.eye(N) - np.ones((N,N))/float(N)
		else:
			H = db['H']
	else:
		db['H'] = H = np.eye(N) - np.ones((N,N))/float(N)

	return H.dot(M).dot(H)	


def nomalized_by_Degree_matrix(M, D):
	D2 = np.diag(D)
	DMD = M*(np.outer(D2, D2))
	return DMD

def compute_inverted_Degree_matrix(M):
	return np.diag(1.0/np.sqrt(M.sum(axis=1)))

def compute_Degree_matrix(M):
	return np.diag(np.sum(M, axis=0))


def normalize_U(U):
	return normalize(U, norm='l2', axis=1)


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

def L_to_U(db, L, return_eig_val=False):
	L = ensure_matrix_is_numpy(L)
	eigenValues,eigenVectors = np.linalg.eigh(L)

	n2 = len(eigenValues)
	n1 = n2 - db['num_of_clusters']
	U = eigenVectors[:, n1:n2]
	U_lambda = eigenValues[n1:n2]
	U_normalized = normalize(U, norm='l2', axis=1)
	
	if return_eig_val: return [U, U_normalized, U_lambda]
	else: return [U, U_normalized]

