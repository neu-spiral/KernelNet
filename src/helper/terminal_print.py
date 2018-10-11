
import sys

def clear_current_line():
	sys.stdout.write("\r")
	sys.stdout.write("\033[K")
	sys.stdout.flush()

def write_to_current_line(txt):
	clear_current_line()
	sys.stdout.write(txt)
	#print(txt)

def clear_previous_line():
	clear_current_line()
	sys.stdout.write("\r")
	sys.stdout.write("\033[F")
	sys.stdout.flush()
	clear_current_line()

def loss_optimization_printout(db, epoch, avgLoss, avgGrad, epoc_loop, slope):
	if db['print_optimizer_loss']: 
		sys.stdout.write("\r\t\t%d/%d, MaxLoss : %f, AvgGra : %f, progress slope : %f" % (epoch, epoc_loop, avgLoss, avgGrad, slope))
		sys.stdout.flush()

