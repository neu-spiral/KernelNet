
import torch
import debug
from format_conversion import *
from kernel_lib import *
from basic_optimizer import *
from U_optimize_cost import *
from orthogonal_optimization import *
from RFF import *

class sm_opt_K():
	def __init__(self, db):
		self.db = db

	def run(self, count, start_time):
		print('\t' + str(count) + ' : Computing K...')

		db = self.db
		if 'Ku' not in db: db['Ku'] = db['U'].dot(db['U'].T)
	
		Y = center_matrix(db, db['Ku'])
		if db['use_Degree_matrix']: Y = db['D_inv'].dot(Y).dot(db['D_inv'])	
		np.fill_diagonal(Y, 0)
		Y = numpy2Variable(Y, db['dataType'])

		db['knet'].set_Y(Y)
		[avgLoss, avgGrad, progression_slope] = basic_optimizer(db['knet'], db, loss_callback='compute_loss', data_loader_name='train_loader', epoc_loop=10)
		[db['x_hat'], db['ϕ_x']] = db['knet'](db['train_data'].X_Var)		# <- update this to be used in opt_K

		debug.print_opt_K_status(db, start_time)


class sm_opt_U():
	def __init__(self, db):
		self.db = db

	def run(self, count, start_time):
		print('\t' + str(count) + ' : Computing U with Orthogonal Optimization...')
		db = self.db

		φ_x = ensure_matrix_is_numpy(db['ϕ_x'])
		[DKxD, db['D_inv']] = normalized_rbk_sklearn(φ_x, db['knet'].σ)
		HDKxDH = center_matrix(db, DKxD)

		u_cost = U_optimize_cost(db['train_data'].X, HDKxDH, db['knet'].σ)
		OO = orthogonal_optimization(u_cost.compute_cost, u_cost.compute_gradient)
		U = OO.run(db['U'], max_rep=5)
		U_normalized = normalize(U, norm='l2', axis=1)


		if db['use_delta_kernel_for_U']: 
			[allocation, train_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=db['train_data'].Y)
			db['U'] = Allocation_2_Y(allocation)
		else:
			db['U'] = U
			[allocation, train_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=db['train_data'].Y)

		db['prev_Ku'] = db['Ku']
		db['Ku'] = db['U'].dot(db['U'].T)


		debug.print_opt_U_status(db, HDKxDH, train_nmi, U, start_time)

