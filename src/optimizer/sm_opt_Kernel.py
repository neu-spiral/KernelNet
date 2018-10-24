
import torch
from format_conversion import *
from kernel_lib import *
from basic_optimizer import *
from U_optimize_cost import *
from orthogonal_optimization import *
from RFF import *

class sm_opt_K():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		print('\t' + str(count) + ' : Computing K...')

		db = self.db
		U = ensure_matrix_is_numpy(db['U'])
		if 'Ku' in db: db['prev_Ku'] = db['Ku']
		db['Ku'] = U.dot(U.T)
	
		Y = center_matrix(db, db['Ku'])
		if db['use_Degree_matrix']: Y = db['D_inv'].dot(Y).dot(db['D_inv'])	
		np.fill_diagonal(Y, 0)
		Y = numpy2Variable(Y, db['dataType'])

		db['knet'].set_Y(Y)
		[avgLoss, avgGrad, progression_slope] = basic_optimizer(db['knet'], db, loss_callback='compute_loss', data_loader_name='train_loader', epoc_loop=10)
		[db['x_hat'], db['ϕ_x']] = db['knet'](db['train_data'].X_Var)		# <- update this to be used in opt_K

		if 'objective_tracker' in db:
			if 'running_batch_mode' in db: return
			[current_loss, current_hsic, current_AE_loss, φ_x, U, U_normalized] = db['knet'].get_current_state(db, db['train_data'].X_Var)
			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)
			[allocation, train_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=db['train_data'].Y)

			print('\t\tCurrent obj loss : %.5f from %.5f +  (%.3f)(%.3f)[%.5f]'%(current_loss, current_hsic, db["λ_ratio"], db['λ_obj_ratio'], current_AE_loss))
			print('\t\tTrain NMI after optimizing θ : %.3f'%(train_nmi))


class sm_opt_U():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		print('\t' + str(count) + ' : Computing U with Orthogonal Optimization...')

		db = self.db
		[x_hat, φ_x] = db['knet'](db['train_data'].X_Var)
		φ_x = ensure_matrix_is_numpy(φ_x)

		[DKxD, Dinv] = normalized_rbk_sklearn(φ_x, db['knet'].σ)
		HDKxDH = center_matrix(db, DKxD)

		u_cost = U_optimize_cost(db['train_data'].X, HDKxDH, db['knet'].σ)
		OO = orthogonal_optimization(u_cost.compute_cost, u_cost.compute_gradient)
		U = OO.run(db['U'], max_rep=5)
		U_normalized = normalize(U, norm='l2', axis=1)

		db['U_prev'] = db['U']
		if db['use_U_normalize']: db['U'] = U_normalized 	# <- update this to be used in opt_K
		else: db['U'] = U

		if 'objective_tracker' in db:
			if 'running_batch_mode' in db: return
			[current_loss, current_hsic, current_AE_loss, φ_x, U, U_normalized] = db['knet'].get_current_state(db, db['train_data'].X_Var)
			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)	

			#[allocation, train_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=db['train_data'].Y)
			[allocation, train_nmi] = kmeans(db['num_of_clusters'], db['U'] , Y=db['train_data'].Y)
			print('\t\tCurrent obj loss : %.5f from %.5f +  (%.3f)(%.3f)[%.5f]'%(current_loss, current_hsic, db["λ_ratio"], db['λ_obj_ratio'], current_AE_loss))
			print('\t\tTrain NMI after optimizing U : %.3f'%(train_nmi))

