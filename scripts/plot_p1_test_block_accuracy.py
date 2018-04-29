# read the eval_details.txt file and plot the test accuracy 


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys 
from pprint import pprint 
import matplotlib.pyplot as pl
import matplotlib.cm as cm

kept_string = 'kept_percentage'
scope_string = 'indexed_prune_scopes'
accuracy_string = 'test_accuracy'

def is_float(s):
	try:
		float(s)
	except:
		return False
	return True

def get_float(s):
	try:
		return float(s.strip())
	except:
		raise ValueError('cannot convert to float: %s' %(s))
	return s 

def get_int(s):
	try:
		return int(s.strip())
	except:
		raise ValueError('cannot convert to int: %s' %(s))
	return s 

def extract_config(line):
	items = line.split(',')

	# extract kept_percentage
	kept_percentage = items[0].strip().split(':')[-1]
	if not is_float(kept_percentage):
		raise ValueError('kept_percentage is not float: %s' %(kept_percentage))
	# kept_percentage = float(kept_percentage)

	# extract block and unit index , e.g., 11, 12 
	prune_scopes = items[-1].strip()
	block_unit = prune_scopes[1:3]
	return kept_percentage, block_unit 



filename = sys.argv[1]
filepath = '/'.join(filename.split('/')[:-1])
print('filename:'+filename)
print('filepath:'+filepath)
data = {}
with open(filename, 'r') as f:
	lines = f.readlines()
	kept_pecentange_changed = False
	
	for line in lines:
		if kept_string in line:
			# this line contains the kept_percentage and the prune scopes
			kept_percentage, block_unit = extract_config(line)
			if kept_percentage not in data:
				data[kept_percentage] = {}
			if block_unit not in data[kept_percentage]:
				data[kept_percentage][block_unit] = {'steps':[], 'accuracies':[]}

			steps = data[kept_percentage][block_unit]['steps']
			accuracies = data[kept_percentage][block_unit]['accuracies']

		elif accuracy_string in line:
			step  = get_int(line.split(',')[0].strip().split()[-1])
			accuracy = get_float(line.strip().split()[-1])
			steps.append(step)
			accuracies.append(accuracy)
# pprint(data)

# plot data

for kept_percentage in sorted(data.keys()):
	block_unit_results = data[kept_percentage]
	xs = []
	ys = []
	legends = []
	fontsize = 18

	for block_unit in sorted(block_unit_results.keys()):
		results = block_unit_results[block_unit]
		xs.append(results['steps'])
		ys.append(results['accuracies'])
		legends.append(block_unit)
	# pl.plot(xs, ys)
	#pl.figure()
	for i in range(len(xs)):
		pl.plot(xs[i], ys[i], color=cm.spectral(1.0*i/len(xs)))
	pl.legend(legends, bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
	pl.grid()
	pl.xlabel('iteration', fontsize=fontsize)
	pl.ylabel('accuracy', fontsize=fontsize)
	pl.xticks(size=fontsize)
	pl.yticks(size=fontsize)
	figname = os.path.join(figname, 'test_acc.png')
	pl.savefig(figname, bbox_inches='tight')
	#pl.show()



# show the initial accuracy and the final accuracy
	print('kept_percentage', kept_percentage)
	acc = [(y[0],y[-1]) for y in ys]
	print('init-final acc:',zip(legends, acc))
	
	


				
