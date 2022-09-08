import os
import random
from re import L
import numpy as np
import pandas as pd
from ALL import ALL
from MRHC import MRHC
from MChen import MChen
from MRSP3 import MRSP3
from MRSP2 import MRSP2
from MRSP1 import MRSP1
from scipy.sparse import issparse
from skmultilearn.dataset import load_dataset
from skmultilearn.dataset import available_data_sets
from timeit import default_timer as timer


results_path_root = 'Results'
reduction_path = 'Reduction'
scores_path = 'Scores'

""" Corpus loading procedure """
def loadCorpus(corpus_name):
	# Train partition:
	X_train, y_train, _, _ = load_dataset(corpus_name, 'train')

	# Test partition:
	X_test, y_test, _, _ = load_dataset(corpus_name, 'test')

	return X_train, y_train, X_test, y_test


""" Noise induction procedure """ 
def create_label_noise(y_train, noise_perc):
	# Number of elements (even number):
	number_of_elements = 2*round(noise_perc*y_train.shape[0]/(200))

	# Selecting elements:
	list_elements = random.sample(list(range(y_train.shape[0])), k = number_of_elements)

	# Creating output vector:
	y_out = y_train.todense().copy() if issparse(y_train) else y_train.copy()

	# Iterating through the selected pairs:
	for it in range(int(len(list_elements)/2)):
		temp = y_out[list_elements[it]].copy()
		y_out[list_elements[it]] = y_out[list_elements[len(list_elements)-1-it]].copy()
		y_out[list_elements[len(list_elements)-1-it]] = temp.copy()
	

	return y_out




""" Experimentation """
def experiments():

	# Reduction path:
	if not os.path.exists(os.path.join(results_path_root, reduction_path)):
		os.makedirs(os.path.join(results_path_root, reduction_path))

	# Selected corpora:
	corpora = ['bibtex', 'birds', 'Corel5k', 'emotions', 'genbase', 'medical', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'scene', 'yeast']

	# Label noise:
	label_noise_percentage = [0]#, 20, 40]

	# Reduction algorithms:
	red_algs = ['ALL', 'MRSP1', 'MRSP2', 'MRSP3', 'MRHC', 'MChen']

	# Params dict:
	red_algos_param = {
		'ALL' : [1],
		'MRHC' : [1],
		'MRSP3' : [1],
		'MRSP1' : [10, 30, 50, 70, 90],
		'MRSP2' : [10, 30, 50, 70, 90],
		'MChen' : [10, 30, 50, 70, 90],
	}


	# Creating pandas dataframe for the results:
	out_file = pd.DataFrame(columns=['red_alg', 'red_alg_params', 'corpus', 'noise', 'time'])

	# Auxiliar dictionary for the results:
	res_line = dict()
	
	# Iterating through the different corpora:
	for single_corpus in corpora:
		print("-"*80)
		print("CORPUS {}".format(single_corpus))

		# Output dictionary:
		res_line['corpus'] = [single_corpus]

		# Dst folder:
		corpus_dst_path = os.path.join(results_path_root, reduction_path,single_corpus)
		if not os.path.exists(corpus_dst_path):
			os.makedirs(corpus_dst_path)
		
		# Loading corpus:
		X_train, y_train, _, _ = loadCorpus(single_corpus)

		# Iterating through noise percentage:
		for noise_percentage in label_noise_percentage:
			print("- Noise {}%".format(noise_percentage))

			# Output dictionary:
			res_line['noise'] = [noise_percentage]

			# Induce label noise in the train data:
			y_train = create_label_noise(y_train, noise_percentage)

			# Iterating through PG algorithms:
			for single_red in red_algs:
				print("-- Reduction {}".format(single_red))

				# Output dictionary:
				res_line['red_alg'] = [single_red]
				
				# Reduction method:
				red = eval(single_red + '()')
				
				# Iterating through the different reduction parameters of the current PG method:
				for red_parameter in red_algos_param[single_red]:
					print("--- Reduction parameter {}".format(red_parameter))

					# Output dictionary:
					res_line['red_alg_params'] = [str(red_parameter)]
					
					# Start timer:
					start = timer()

					# Performing the reduction:
					X_red, y_red = red.reduceSet(X = X_train.toarray(), y = y_train, params = red_parameter)

					# End timer:
					end = timer()

					# Output dictionary:
					res_line['time'] = end - start

					# Saving results in the output file:
					out_file = out_file.append(pd.DataFrame(res_line), ignore_index = False)
					out_file.to_csv(os.path.join(results_path_root, 'Results_time_plain_1.csv'), index=False)

	return




if __name__ == '__main__':
	random.seed(123)

	"""Running experiments"""
	experiments()


	


