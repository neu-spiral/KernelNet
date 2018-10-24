#!/usr/bin/python
#	Note : This is designed for Python 3


import numpy as np

class orthogonal_optimization:
	def __init__(self, cost_function, gradient_function):
		self.cost_function = cost_function
		self.gradient_function = gradient_function
		self.x_opt = None
		self.cost_opt = None
		self.db = {}
		#self.db['run_debug_2'] = True
		#self.db['run_debug_1'] = True

	def calc_A(self, x):
		G = self.gradient_function(x)
		A = G.dot(x.T) - x.dot(G.T)
		return [A,G]

	#	Applying Sherman-Morrison-Woodbury Theorem ( A faster way to update instead of recalculating inverse )
	def constant_update_inv(self, x, G, M_inv, alpha_D):
		if alpha_D == 0: return M_inv
		d = x.shape[1]
		I = np.eye(d)

		#	1st update
		U = alpha_D*G
		V = x
		E = np.linalg.inv(I + V.T.dot(M_inv).dot(U))
		M_inv = M_inv - M_inv.dot(U).dot(E).dot(V.T).dot(M_inv)

		#	2nd update
		U = -alpha_D*x
		V = G
		E = np.linalg.inv(I + V.T.dot(M_inv).dot(U))
		M_inv = M_inv - M_inv.dot(U).dot(E).dot(V.T).dot(M_inv)
	
		return M_inv

	def compute_gradient(self, x):
		[A,G] = self.calc_A(x)
		return A.dot(x)

	def run(self, x_init, max_rep=400):
		d = x_init.shape[0]
		self.x_opt = x_init
		I = np.eye(d)
		converged = False
		x_change = np.linalg.norm(x_init)
		m = 0

		in_cost = self.cost_function(self.x_opt)

		while( (converged == False) and (m < max_rep)):
			old_alpha = 2
			new_alpha = 2
			alpha_D = 0
			cost_1 = self.cost_function(self.x_opt)
			[A,g] = self.calc_A(self.x_opt)
			M_inv = np.linalg.inv(I + new_alpha*A)

			while(new_alpha > 0.000000001):	
				if True: M_inv = self.constant_update_inv(self.x_opt, g, M_inv, alpha_D)		#	using woodbury inverse update
				else:	M_inv = np.linalg.inv(I + new_alpha*A) 									#	using slow inverse

				#next_x_o = M_inv.dot(I - new_alpha*A).dot(self.x_opt)
				#M_inv = np.linalg.inv(I + new_alpha*A) 									#	using slow inverse
				#next_x = M_inv.dot(I - new_alpha*A).dot(self.x_opt)
				#print '\n'
				#print '------------------------------------', np.linalg.norm(next_x - next_x_old)
				#import pdb; pdb.set_trace()


				next_x = M_inv.dot(I - new_alpha*A).dot(self.x_opt)
				cost_2 = self.cost_function(next_x)

				if 'run_debug_1' in self.db: print(new_alpha, cost_1, cost_2)
				#if((cost_2 < cost_1) or (abs(cost_1 - cost_2)/abs(cost_1) < 0.0000001)):
				if(cost_2 < cost_1):
					x_change = np.linalg.norm(next_x - self.x_opt)
					[self.x_opt,R] = np.linalg.qr(next_x)		# QR ensures orthogonality
					self.cost_opt = cost_2
					break
				else:
					old_alpha = new_alpha
					new_alpha = new_alpha*0.2
					alpha_D = new_alpha - old_alpha

			m += 1

			if 'run_debug_2' in self.db: print('Cost Norm : %.3f'%cost_2)
			if 'run_debug_3' in self.db: print('Gradient Norm : %.3f'%np.linalg.norm(self.compute_gradient(self.x_opt)))

			#print(x_change)
			if(x_change < 0.001*np.linalg.norm(self.x_opt)): converged = True

		out_cost = self.cost_function(self.x_opt)
		print('\t\tin cost %.3f , out cost %.3f'%(in_cost,out_cost))

		#if out_cost > in_cost:
		#	import pdb; pdb.set_trace()

		return self.x_opt	
