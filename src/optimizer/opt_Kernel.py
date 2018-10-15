
import torch
from format_conversion import *
from kernel_lib import *
from basic_optimizer import *

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
		Y = numpy2Variable(Y, db['dataType'])

		db['knet'].set_Y(Y)
		[avgLoss, avgGrad, progression_slope] = basic_optimizer(db['knet'], db, loss_callback='compute_loss', data_loader_name='train_loader')
		[db['x_hat'], db['ϕ_x']] = db['knet'](db['train_data'].X_Var)		# <- update this to be used in opt_K

		if 'objective_tracker' in db:
			[current_loss, current_hsic, current_AE_loss] = db['knet'].get_current_state(db)
			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)


class opt_U():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		print('\t' + str(count) + ' : Computing U with Eigen Decomposition ...')
		db = self.db

		[DKxD, D] = normalized_rbk_sklearn(db['ϕ_x'], db['φ_x_mpd'])
		HDKxDH = center_matrix(db,DKxD)
		[U, U_normalized] = L_to_U(db, HDKxDH)
		db['U_prev'] = U
		db['U'] = U_normalized 	# <- update this to be used in opt_K
	
		if 'objective_tracker' in db:
			#feasibility_obj = np.linalg.norm(np.eye(U.shape[1]) - U.T.dot(U))
			#db['feasibility_tracker'] = np.append(db['feasibility_tracker'], feasibility_obj)
			#db['feasibility_tracker'] = np.append(db['feasibility_tracker'], feasibility_obj)
			[current_loss, current_hsic, current_AE_loss] = db['knet'].get_current_state(db)
			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)
	

def exit_cond(db, count):

	dims = float(db['U'].shape[0]*db['U'].shape[1])
	error_per_element = np.absolute(db['U']- db['U_prev']).sum()/dims
	error_per_element = '%.4f' % error_per_element
	db['converge_list'].append(error_per_element)

	clear_previous_line()
	print('\tBetween U, Kx error Per element : ' + str(db['converge_list']))
	if float(error_per_element) <= 0.01:
		db['itr_til_converge'] = count
		exit_count = 100
	else:
		exit_count = count


	if exit_count > 99 or 'objective_tracker' in db:
		[db['current_loss'], db['current_hsic'], db['current_AE_loss']] = db['knet'].get_current_state(db)

		current_label = kmeans(db['num_of_clusters'], db['U'])
		db['test_nmi'] = normalized_mutual_info_score(current_label, db['train_data'].Y)
		print('\t\t%d , current nmmi : %.3f'%(count, db['test_nmi']))
	


	return exit_count

