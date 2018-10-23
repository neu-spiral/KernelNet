
import torch
from matplotlib import pyplot as plt

def plot_alloc(self, db, plotID, data, title, linetype=None, fsize=20, xyLabels=[]):
	color_list = ['b', 'r', 'g', 'c', 'm', 'y', 'k', 'w']

	plt.subplot(plotID)
	Uq_a = np.unique(db['allocation'])
	
	for m in range(len(Uq_a)):
		g = data[db['allocation'] == Uq_a[m]]
		if linetype is None:
			plt.plot(g[:,0], g[:,1], color_list[m] + 'o')
		else:
			plt.plot(g[:,0], g[:,1], color_list[m] + linetype[m])

	#plt.tick_params(labelsize=9)
	plt.title(title, fontsize=fsize, fontweight='bold')
	if len(xyLabels) > 1:
		plt.xlabel(xyLabels[0], fontsize=fsize, fontweight='bold')
		plt.ylabel(xyLabels[1], fontsize=fsize, fontweight='bold')
	plt.tick_params(labelsize=10)

def plot_output(db):
	[current_loss, current_hsic, current_AE_loss, φ_x, U, U_normalized] = db['knet'].get_current_state(db, db['train_data'].X_Var)
	plt.plot(φ_x[:,0], φ_x[:,1], 'go')
	plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
	plt.show()

	#print(φ_x)

def end2end(db):
	X = db['train_data'].X_Var

	[X_hat, fx] = db['knet'](X)
	print('X_hat : %d x %d'%(X_hat.shape[0], X_hat.shape[1]))
	print(X_hat[0:10,0:4])
	print('\n')
	print('X : %d x %d'%(X.shape[0], X.shape[1]))
	print(X[0:10,0:4])

	element_error = torch.abs(X - X_hat).mean()
	print('\nAE error per element %.3f'%element_error)

	import pdb; pdb.set_trace()

def layer_wise(db, rbm):
	print('debug code inside : pretrain')
	X = db['train_data'].X_Var
	X_hat = rbm(X)

	print('X_hat : %d x %d'%(X_hat.shape[0], X_hat.shape[1]))
	print(X_hat[0:10,0:4])
	print('\n')
	print('X : %d x %d'%(X.shape[0], X.shape[1]))
	print(X[0:10,0:4])
	import pdb; pdb.set_trace()


