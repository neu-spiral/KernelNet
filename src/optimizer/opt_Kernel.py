
import torch
import debug
from format_conversion import *
from kernel_lib import *
from basic_optimizer import *
from RFF import *

class opt_K():
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
		[avgLoss, avgGrad, progression_slope] = basic_optimizer(db['knet'], db, loss_callback='compute_loss', data_loader_name='train_loader', epoc_loop=20)
		[db['x_hat'], db['ϕ_x']] = db['knet'](db['train_data'].X_Var)		# <- update this to be used in opt_K

		debug.print_opt_K_status(db, start_time)

class opt_U():
	def __init__(self, db):
		self.db = db

	def run(self, count, start_time):
		print('\t' + str(count) + ' : Computing U with Eigen Decomposition ...')
		db = self.db

		φ_x = ensure_matrix_is_numpy(db['ϕ_x'])
		[DKxD, Dinv] = normalized_rbk_sklearn(φ_x, db['knet'].σ)
		HDKxDH = center_matrix(db, DKxD)

		if db['use_delta_kernel_for_U']: 
			[U, U_normalized] = L_to_U(db, HDKxDH)
			[allocation, train_nmi] = kmeans(db['num_of_clusters'], U_normalized, Y=db['train_data'].Y)
			db['U'] = Allocation_2_Y(allocation)
		else:
			[U, db['U_normalized']] = L_to_U(db, HDKxDH)
			[allocation, train_nmi] = kmeans(db['num_of_clusters'], db['U_normalized'], Y=db['train_data'].Y)
			db['U'] = U
		
		db['prev_Ku'] = db['Ku']
		db['Ku'] = db['U'].dot(db['U'].T)

		debug.print_opt_U_status(db, HDKxDH, train_nmi, U, start_time)
	

def exit_cond(db, count):
	if 'prev_Ku' not in db: return count

	N = float(db['Ku'].shape[0])
	error_per_element = np.absolute(db['prev_Ku'] - db['Ku']).sum()/(N*N)
	error_per_element = '%.2f' % error_per_element
	db['converge_list'].append(error_per_element)

	#clear_previous_line()
	print('\t\tBetween U, Kx error Per element : ' + str(db['converge_list']))
	if float(error_per_element) <= 0.01:
		return True

	return False




