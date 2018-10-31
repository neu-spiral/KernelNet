
import torch
from format_conversion import *
from kernel_lib import *
from basic_optimizer import *
from RFF import *

class opt_K():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		print('\t' + str(count) + ' : Computing K...')

		db = self.db
		if 'Ku' not in db: db['Ku'] = db['U'].dot(db['U'].T)
	
		Y = center_matrix(db, db['Ku'])
		if db['use_Degree_matrix']: Y = db['D_inv'].dot(Y).dot(db['D_inv'])	
		np.fill_diagonal(Y, 0)
		Y = numpy2Variable(Y, db['dataType'])

		db['knet'].set_Y(Y)
		[avgLoss, avgGrad, progression_slope] = basic_optimizer(db['knet'], db, loss_callback='compute_loss', data_loader_name='train_loader', epoc_loop=1000)
		[db['x_hat'], db['ϕ_x']] = db['knet'](db['train_data'].X_Var)		# <- update this to be used in opt_K

		if 'objective_tracker' in db:
			if 'running_batch_mode' in db: return

			φ_x = ensure_matrix_is_numpy(db['ϕ_x'])
			[DKxD, Dinv] = normalized_rbk_sklearn(φ_x, db['knet'].σ)
			HDKxDH = center_matrix(db, DKxD)

			current_hsic = -float(np.sum(HDKxDH*db['Ku']))
			current_AE_loss = float(ensure_matrix_is_numpy(db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)))
			current_loss = float(current_hsic + db['λ']*current_AE_loss)

			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)
			#[allocation, train_nmi] = kmeans(db['num_of_clusters'], db['U_normalized'], Y=db['train_data'].Y)
			[allocation, train_nmi] = kmeans(db['num_of_clusters'], db['U'], Y=db['train_data'].Y)

			print('\t\tCurrent obj loss : %.5f from %.5f +  (%.3f)(%.3f)[%.5f]'%(current_loss, current_hsic, db["λ_ratio"], db['λ_obj_ratio'], current_AE_loss))
			print('\t\tTrain NMI after optimizing θ : %.3f'%(train_nmi))


class opt_U():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		print('\t' + str(count) + ' : Computing U with Eigen Decomposition ...')
		db = self.db

		φ_x = ensure_matrix_is_numpy(db['ϕ_x'])
		[DKxD, Dinv] = normalized_rbk_sklearn(φ_x, db['knet'].σ)
		HDKxDH = center_matrix(db, DKxD)

		#[db['U'], db['U_normalized']] = L_to_U(db, HDKxDH)
		[U, U_normalized] = L_to_U(db, HDKxDH)
		[allocation, train_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=db['train_data'].Y)
		db['U'] = Allocation_2_Y(allocation)


		db['prev_Ku'] = db['Ku']
		db['Ku'] = db['U'].dot(db['U'].T)

		if 'objective_tracker' in db:
			if 'running_batch_mode' in db: return

			current_hsic = -float(np.sum(HDKxDH*db['Ku']))
			current_AE_loss = float(ensure_matrix_is_numpy(db['knet'].autoencoder_loss(db['train_data'].X_Var, None, None)))
			current_loss = float(current_hsic + db['λ']*current_AE_loss)

			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)

			#[allocation, train_nmi] = kmeans(db['num_of_clusters'], db['U_normalized'] , Y=db['train_data'].Y)
			print('\t\tCurrent obj loss : %.5f from %.5f +  (%.3f)(%.3f)[%.5f]'%(current_loss, current_hsic, db["λ_ratio"], db['λ_obj_ratio'], current_AE_loss))
			print('\t\tTrain NMI after optimizing U : %.3f'%(train_nmi))

	

def exit_cond(db, count):
	if 'prev_Ku' not in db: return count

	N = float(db['Ku'].shape[0])
<<<<<<< HEAD
	error_per_element = 10*np.absolute(db['prev_Ku'] - db['Ku']).sum()/(N*N)
=======
	error_per_element = np.absolute(db['prev_Ku'] - db['Ku']).sum()/(N*N)
>>>>>>> dcf3e6f57a88c0983b20818245fc24dc1cefcf47
	error_per_element = '%.2f' % error_per_element
	db['converge_list'].append(error_per_element)

	#clear_previous_line()
	print('\t\tBetween U, Kx error Per element : ' + str(db['converge_list']))
	if float(error_per_element) <= 0.001:
		db['knet'].itr_til_converge = count
		exit_count = 100
	else:
		exit_count = count

	return exit_count

