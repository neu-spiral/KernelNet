

class opt_K():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		db = self.db
		U = ensure_matrix_is_numpy(U)
		Ku = U.dot(U.T)
	
		Y = center_matrix(db, Ku)
		if db['use_Degree_matrix']: Y = db['D_inv'].dot(Y).dot(db['D_inv'])	
		np.fill_diagonal(Y, 0)
		Knet.set_Y(Y)
	
		avgLoss = loss_optimizer(Knet, db, epoc_loop=600, grad_exit=0.0005)	#600
		Xout = Knet(X)


	if 'objective_tracker' in db:
		[Kx, Dinv, L] = compute_output_Laplacian(db, Xout, Knet.sigma)
		L = center_matrix(db,L)

		obj = np.sum(Ku*L)
		db['objective_tracker'] = np.append(db['objective_tracker'], obj)
		print('\nlen : %d,  objective after K : %.4f\n'%(len(db['objective_tracker']), obj))

	clear_previous_line()
	return [Ku, Xout, Knet]

class opt_U():
	def __init__(self, db):
		self.db = db

	def run(self, count):
		pass

