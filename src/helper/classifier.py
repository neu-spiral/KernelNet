		
from sklearn import svm
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from kernel_lib import *
from sklearn import linear_model

		
def use_svm(X,Y,W=None, k='rbf'):
	if W is not None:
		X = X.dot(W)

	clf = svm.SVC(kernel=k)
	clf.fit(X, Y)
	out_allocation = clf.predict(X)
	nmi = normalized_mutual_info_score(out_allocation, Y)

	return [out_allocation, nmi]

def kmeans(k, U, Y=None):

	allocation = KMeans(k).fit_predict(U)

	if Y is None:
		return allocation
	else:
		nmi = normalized_mutual_info_score(allocation, Y)
		return [allocation, nmi]

def centered_spectral_clustering(data, k, σ, Y=None):
	[L,D] = normalized_rbk_sklearn(data, σ)
	[U, U_λ] = eig_solver(L, k, mode='largest')
	U_normed = normalize_U(U)

	return kmeans(k, U_normed, Y)

def my_spectral_clustering(data, k, σ, H=None, Y=None):
	[L,D] = normalized_rbk_sklearn(data, σ)
	if H is not None:
		L = H.dot(L).dot(H)

	[U, U_λ] = eig_solver(L, k, mode='largest')
	U_normed = normalize_U(U)

	return kmeans(k, U_normed, Y)


def get_slope(y_axis):
	y_axis = np.array(list(y_axis))

	n = len(y_axis)
	LR = linear_model.LinearRegression()
	X = np.array(range(n))
	X = X.reshape((n,1))
	y_axis = y_axis.reshape((n,1))
	LR.fit(X, y_axis)
	#print LR.intercept_
	
	return LR.coef_



