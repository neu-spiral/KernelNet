
import torch
import time
from classifier import *
from path_tools import *
from matplotlib import pyplot as plt

def plot_alloc(db, plotID, data, title, linetype=None, fsize=20, xyLabels=[]):
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
	if 'running_batch_mode' in db: return

	[current_loss, current_hsic, current_AE_loss, φ_x, U, U_normalized] = db['knet'].get_current_state(db, db['train_data'].X_Var)
	db['allocation'] = kmeans(db['num_of_clusters'], U_normalized)
	#plt.plot(φ_x[:,0], φ_x[:,1], 'go')
	plot_alloc(db, 111, db['train_data'].X, '', linetype=None, fsize=20, xyLabels=[])

	plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)
	plt.show()

	import pdb; pdb.set_trace()

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


def print_opt_K_status(db, start_time):
	if 'objective_tracker' in db:
		if 'running_batch_mode' in db: return
		if 'constraint_tracker' not in db: db['constraint_tracker'] = []
		if 'time_tracker' not in db: db['time_tracker'] = []

		φ_x = ensure_matrix_is_numpy(db['ϕ_x'])
		[DKxD, Dinv] = normalized_rbk_sklearn(φ_x, db['knet'].σ)
		HDKxDH = center_matrix(db, DKxD)

		current_hsic = -float(np.sum(HDKxDH*db['Ku']))
		current_AE_loss = float(ensure_matrix_is_numpy(db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)))
		current_loss = float(current_hsic + db['λ']*current_AE_loss)

		db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)
		db['time_tracker'].append(time.time() - start_time)
		#[allocation, train_nmi] = kmeans(db['num_of_clusters'], db['U_normalized'], Y=db['train_data'].Y)
		[allocation, train_nmi] = kmeans(db['num_of_clusters'], db['U'], Y=db['train_data'].Y)

		print('\t\tCurrent obj loss : %.5f from %.5f +  (%.3f)(%.3f)[%.5f]'%(current_loss, current_hsic, db["λ_ratio"], db['λ_obj_ratio'], current_AE_loss))
		print('\t\tTrain NMI after optimizing θ : %.3f'%(train_nmi))


def print_opt_U_status(db, HDKxDH, train_nmi, U, start_time):
	if 'objective_tracker' in db:
		if 'running_batch_mode' in db: return

		constraint_loss = np.linalg.norm(U.T.dot(U) - np.eye(U.shape[1]))
		db['constraint_tracker'] = np.append(db['constraint_tracker'], constraint_loss)


		current_hsic = -float(np.sum(HDKxDH*db['Ku']))
		current_AE_loss = float(ensure_matrix_is_numpy(db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)))
		current_loss = float(current_hsic + db['λ']*current_AE_loss)

		db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)
		db['time_tracker'].append(time.time() - start_time)

		#[allocation, train_nmi] = kmeans(db['num_of_clusters'], db['U_normalized'] , Y=db['train_data'].Y)
		print('\t\tCurrent obj loss : %.5f from %.5f +  (%.3f)(%.3f)[%.5f]'%(current_loss, current_hsic, db["λ_ratio"], db['λ_obj_ratio'], current_AE_loss))
		print('\t\tTrain NMI after optimizing U : %.3f'%(train_nmi))

def plot_Objective_trajectories(db):
	if 'objective_tracker' in db:
		if 'running_batch_mode' in db: return

		Y = db['knet'].objective_tracker
		X = range(len(Y))
		Y2 = Y[::2]
		X2 = X[::2]

		plt.figure(figsize=(8,12))

		plt.subplot(311)
		plt.plot(X, Y, 'g')
		#plt.plot(X2, Y2, 'rx')
		#plt.xlabel('Iteration', fontsize=13)
		plt.ylabel('objective', fontsize=10)
		plt.title('%s : Cost vs K, U iterations'%db['data_name'], fontsize=12)
		plt.tick_params(labelsize=8)

		msg = 'U optimizer : %s\n'%db['opt_U_class'].__name__
		msg += 'Kernel Model : %s\n'%db['kernel_model'].__name__

		plt.text(0, np.max(Y),msg, ha='left', va='top',fontsize=10)


		#---------------------
		Y = db['knet'].objective_tracker
		X = db['time_tracker']
		Y2 = Y[::2]
		X2 = X[::2]
		Y3 = Y[1::2]
		X3 = X[1::2]

		plt.subplot(312)
		plt.plot(X, Y, 'g')
		plt.plot(X2, Y2, 'x')
		plt.plot(X3, Y3, '.')
		#plt.xlabel('Iteration', fontsize=10)
		plt.ylabel('objective', fontsize=10)
		plt.title('Objective vs time', fontsize=12)
		plt.tick_params(labelsize=8)

		#msg = 'using Full Data:%s\nMax Error:%e\nTraining Time:%.3f s'%(db['train_on_full'], np.max(Y), db['kernel_net_training_time'])
		#plt.text(0, np.max(Y),msg, ha='left', va='top',fontsize=10)



		#---------------------
		Y = db['knet'].constraint_tracker
		X = range(len(Y))

		plt.subplot(313)
		#plt.xlabel('Iteration', fontsize=10)
		plt.ylabel('Feasibility objective', fontsize=10)
		plt.title('Feasibility vs iterations', fontsize=10)
		plt.tick_params(labelsize=8)
		plt.plot(X, Y, 'g')

		#msg = 'using Full Data:%s\nMax Error:%e\nTraining Time:%.3f s'%(db['train_on_full'], np.max(Y), db['kernel_net_training_time'])
		#plt.text(0, np.max(Y),msg, ha='left', va='top',fontsize=10)

		##---------------------
		#Y = db['error_list']
		#X = range(len(Y))

		#plt.subplot(414)
		#plt.plot(X, Y, 'g')
		#plt.gca().invert_yaxis()
		#plt.xlabel('Iteration', fontsize=10)
		#plt.ylabel('Convergence objective', fontsize=10)
		#plt.title('Convergence vs iterations', fontsize=10)
		#plt.tick_params(labelsize=8)


		plt.tight_layout(rect=[0, 0.03, 1, 0.95])
		
		ensure_path_exists('./img_results/')
		pth = './img_results/' + db['data_name'] + '.png' 
		plt.savefig(pth)

		if 'running_batchmode' not in db: 
			try: plt.show()
			except: pass

		plt.clf()

