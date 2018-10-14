
import torch
from format_conversion import *
from kernel_lib import *

class opt_K():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		print('\t' + str(count) + ' : Computing K...')

		db = self.db
		U = ensure_matrix_is_numpy(db['U'])
		Ku = U.dot(U.T)
	
		Y = center_matrix(db, Ku)
		if db['use_Degree_matrix']: Y = db['D_inv'].dot(Y).dot(db['D_inv'])	
		np.fill_diagonal(Y, 0)

		db['knet'].set_Y(Y)
		avgLoss = loss_optimizer(db['knet'], db, epoc_loop=600, grad_exit=0.0005)	#600
		db['ϕ_x'] = db['knet'](X)		# <- update this to be used in opt_K


		if 'objective_tracker' in db:
			[current_loss, current_hsic, current_AE_loss] = db['knet'].get_current_state()
			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)


class opt_U():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		print('\t' + str(count) + ' : Computing U with Eigen Decomposition ...')
		[DKxD, D] = normalized_rbk_sklearn(db['ϕ_x'], db['φ_x_mpd'])
		HDKxKH = center_matrix(db,DKxD)
		[U, U_normalized] = L_to_U(db, L)
		db['U'] = U_normalized 	# <- update this to be used in opt_K
	
		if 'objective_tracker' in db:
			#feasibility_obj = np.linalg.norm(np.eye(U.shape[1]) - U.T.dot(U))
			#db['feasibility_tracker'] = np.append(db['feasibility_tracker'], feasibility_obj)
			#db['feasibility_tracker'] = np.append(db['feasibility_tracker'], feasibility_obj)
			[current_loss, current_hsic, current_AE_loss] = db['knet'].get_current_state()
			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)
	

def exit_cond():
	return True
