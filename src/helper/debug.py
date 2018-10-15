

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


