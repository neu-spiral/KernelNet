
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
		if 'Ku' in db: db['prev_Ku'] = db['Ku']
		db['Ku'] = U.dot(U.T)
	
		Y = center_matrix(db, db['Ku'])
		if db['use_Degree_matrix']: Y = db['D_inv'].dot(Y).dot(db['D_inv'])	
		np.fill_diagonal(Y, 0)
		Y = numpy2Variable(Y, db['dataType'])

		db['knet'].set_Y(Y)
		[avgLoss, avgGrad, progression_slope] = basic_optimizer(db['knet'], db, loss_callback='compute_loss', data_loader_name='train_loader')
		[db['x_hat'], db['ϕ_x']] = db['knet'](db['train_data'].X_Var)		# <- update this to be used in opt_K

		if 'objective_tracker' in db:
			if 'running_batch_mode' in db: return
			[current_loss, current_hsic, current_AE_loss, φ_x] = db['knet'].get_current_state(db, db['train_data'].X_Var)
			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)


class opt_U():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		print('\t' + str(count) + ' : Computing U with Eigen Decomposition ...')
		db = self.db

		[DKxD, D] = normalized_rbk_sklearn(db['ϕ_x'], db['knet'].φ_x_mpd)
		HDKxDH = center_matrix(db,DKxD)
		[U, U_normalized] = L_to_U(db, HDKxDH)
		db['U_prev'] = U

		if db['use_U_normalize']: db['U'] = U_normalized 	# <- update this to be used in opt_K
		else: db['U'] = U
	
		if 'objective_tracker' in db:
			if 'running_batch_mode' in db: return
			[current_loss, current_hsic, current_AE_loss, φ_x] = db['knet'].get_current_state(db, db['train_data'].X_Var)
			db['objective_tracker'] = np.append(db['objective_tracker'], current_loss)
	

def exit_cond(db, count):
	if 'prev_Ku' not in db: return count

	N = float(db['Ku'].shape[0])
	error_per_element = np.absolute(db['prev_Ku'] - db['Ku']).sum()/(N*N)
	error_per_element = '%.2f' % error_per_element
	db['converge_list'].append(error_per_element)

	clear_previous_line()
	print('\t\tBetween U, Kx error Per element : ' + str(db['converge_list']))
	if float(error_per_element) <= 0.05:
		db['knet'].itr_til_converge = count
		exit_count = 100
	else:
		exit_count = count

	if exit_count > 99 or 'running_batch_mode' not in db:
		#	get training nmi
		current_label = kmeans(db['num_of_clusters'], db['U'])
		train_nmi = normalized_mutual_info_score(current_label, db['train_data'].Y)
	
		#	get validation nmi
		if 'valid_data' in db:
			[x_hat, U] = db['knet'](db['valid_data'].X_Var)		# <- update this to be used in opt_K
			current_label = kmeans(db['num_of_clusters'], U)
			valid_nmi = normalized_mutual_info_score(current_label, db['valid_data'].Y)
			print('\t\tTrain NMI : %.3f, Valid NMI : %.3f'%(train_nmi, valid_nmi))
		else:
			print('\t\tTrain NMI : %.3f'%(train_nmi))


	return exit_count

