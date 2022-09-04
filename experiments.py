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
from sklearn.metrics import hamming_loss, accuracy_score, f1_score, jaccard_score, label_ranking_loss
from skmultilearn.dataset import load_dataset
from sklearn.neighbors import KNeighborsClassifier
from skmultilearn.dataset import available_data_sets
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import BRkNNaClassifier, BRkNNbClassifier, MLkNN


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


""" Accuracy for the multilabel case """
def accuracy_ml(y_true, y_pred):
	n_elements = y_true.toarray().shape[0]

	y_true_array = y_true.toarray()
	y_pred_array = y_pred.toarray()

	acc = 0
	for it_element in range(n_elements):
		true_labels = set(np.where(y_true_array[it_element]==1)[0])
		estimated_labels = set(np.where(y_pred_array[it_element]==1)[0])

		try:
			acc += len(true_labels.intersection(estimated_labels)) / len(true_labels.union(estimated_labels))
		except:
			acc += 0

	return acc/n_elements


""" F1 multilabel case (in-house) """
def F1_inhouse(y_true, y_pred):
	n_elements = y_true.toarray().shape[0]

	y_true_array = y_true.toarray()
	y_pred_array = y_pred.toarray()

	out = 0
	for it_element in range(n_elements):
		true_labels = set(np.where(y_true_array[it_element]==1)[0])
		estimated_labels = set(np.where(y_pred_array[it_element]==1)[0])

		try:
			out += 2*len(true_labels.intersection(estimated_labels)) / (len(true_labels) + len(estimated_labels))
		except:
			out += 0

	return out/n_elements






""" Experimentation """
def experiments():

	# Reduction path:
	if not os.path.exists(os.path.join(results_path_root, reduction_path)):
		os.makedirs(os.path.join(results_path_root, reduction_path))

	# Selected corpora:
	corpora = ['bibtex', 'birds', 'Corel5k', 'emotions', 'genbase', 'medical', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'scene', 'yeast']

	# Label noise:
	label_noise_percentage = [0, 20, 40]

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

	# Classifiers:
	classifiers = ['LabelPowerset', 'BRkNNaClassifier', 'MLkNN']#, 'BRkNNbClassifier']

	# Classifier params:
	classifiers_param = {
		'LabelPowerset' : [1, 3, 5, 7],
		'MLkNN' : [1, 3, 5, 7],
		'BRkNNaClassifier' : [1, 3, 5, 7],
		'BRkNNbClassifier' : [1, 3, 5, 7],
	}

	# Creating pandas dataframe for the results:
	out_file = pd.DataFrame(columns=['cls', 'cls_params', 'red_alg', 'red_alg_params', 'corpus', 'noise', 'HL', 'EMR', 'acc', 'jaccard-m', 'F1-m', 'jaccard-M', 'F1-M', 'jaccard-s', 'F1-s', 'F1_inhouse', 'RL', 'Size'])

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
		X_train, y_train, X_test, y_test = loadCorpus(single_corpus)

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
					
					# Reduced files paths:
					X_dst_file = os.path.join(corpus_dst_path, 'X_' + red.getFileName(red_parameter) + '_N' + str(noise_percentage) + '.csv.gz')
					y_dst_file = os.path.join(corpus_dst_path, 'y_' + red.getFileName(red_parameter) + '_N' + str(noise_percentage) + '.csv.gz')
					
					if os.path.isfile(X_dst_file) and os.path.isfile(y_dst_file):
						X_red = np.array(pd.read_csv(X_dst_file, sep=',', header=None, compression='gzip'))
						y_red = np.array(pd.read_csv(y_dst_file, sep=',', header=None, compression='gzip'))
					else:
						# Performing the reduction:
						X_red, y_red = red.reduceSet(X = X_train.toarray(), y = y_train, params = red_parameter)

						# Saving reduction to files:
						pd.DataFrame(X_red).to_csv(X_dst_file, header=None, index=None, compression='gzip')
						pd.DataFrame(y_red).to_csv(y_dst_file, header=None, index=None, compression='gzip')
					
					# Iterating through the contemplated classifiers:
					for single_classifier in classifiers:
						print("---- Classifier {}".format(single_classifier))

						# Output dictionary:
						res_line['cls'] = [single_classifier]

						# Iterating through the different parameters of the current classification strategy:
						for classifier_parameters in classifiers_param[single_classifier]:
							print("----- Classifier param {}".format(classifier_parameters))

							# Output dictionary:
							res_line['cls_params'] = [str(classifier_parameters)]

							# Instantiating classifier:
							if single_classifier == 'LabelPowerset': # LP-kNN
								kNN = KNeighborsClassifier(n_neighbors = classifier_parameters)
								cls = LabelPowerset(classifier = kNN, require_dense=[False, False])
							else: # BRkNN, ML-kNN
								cls = eval(single_classifier + '(k=' + str(classifier_parameters) + ')')
							
							# Fitting classifier:
							cls.fit(X_red, y_red)
							
							# Inference stage
							y_pred = cls.predict(X_test)
							
							# Saving results in the auxiliar dictionary:
							res_line['HL'] = [hamming_loss(y_true = y_test, y_pred = y_pred)]
							res_line['EMR'] = [accuracy_score(y_true = y_test, y_pred = y_pred)]
							res_line['acc'] = [accuracy_ml(y_true = y_test, y_pred = y_pred)]
							res_line['jaccard-m'] = [jaccard_score(y_true = y_test, y_pred = y_pred, average = 'micro')]
							res_line['F1-m'] = [f1_score(y_true = y_test, y_pred = y_pred, average = 'micro')]
							res_line['jaccard-M'] = [jaccard_score(y_true = y_test, y_pred = y_pred, average = 'macro')]
							res_line['F1-M'] = [f1_score(y_true = y_test, y_pred = y_pred, average = 'macro')]
							res_line['jaccard-s'] = [jaccard_score(y_true = y_test, y_pred = y_pred, average = 'samples')]
							res_line['F1-s'] = [f1_score(y_true = y_test, y_pred = y_pred, average = 'samples')]
							res_line['F1_inhouse'] = [F1_inhouse(y_true = y_test, y_pred = y_pred)]
							res_line['RL'] = [label_ranking_loss(y_true = y_test.toarray(), y_score = y_pred.toarray())]


							res_line['Size'] = [100*X_red.shape[0]/X_train.shape[0]]

							print("------ DONE!")

							# Saving results in the output file:
							out_file = out_file.append(pd.DataFrame(res_line), ignore_index = False)
							out_file.to_csv(os.path.join(results_path_root, 'Results_plain.csv'), index=False)


	# Creating the summary of the results:
	out_file.groupby(['cls', 'cls_params', 'red_alg', 'red_alg_params', 'noise']).mean().reset_index().to_csv(os.path.join(results_path_root, "Results_summary.csv"), index=False)

	return


""" Getting data stats from the corpora considered for the experiments """
def getDataStats():
	print("-"*80)
	print("Name, Train size, Test size, Features, Possible tags, Cardinality, Density")

	for single_key in sorted(list(set([u[0] for u in available_data_sets().keys()]))):

		# Loading train/test partitions:
		X_train, y_train, _, _ = load_dataset(single_key,'train')
		X_test, y_test,  _, _ = load_dataset(single_key,'test')

		# Computing cardinality:
		cardinality = (np.count_nonzero(y_train.toarray()) + np.count_nonzero(y_test.toarray()))/(y_train.shape[0] + y_test.shape[0])

		# Printing data:
		print("{},{},{},{},{},{},{}".format(single_key, X_train.shape[0], X_test.shape[0], X_train.shape[1], y_train.shape[1], cardinality, cardinality/y_train.shape[1] ))

	return



""" Imbalance metrics """
def imbalance_metrics():

	# Selected corpora:
	corpora = ['bibtex', 'birds', 'Corel5k', 'emotions', 'genbase', 'medical', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3', 'rcv1subset4', 'scene', 'yeast']

	for single_corpus in corpora:
		# Loading corpus:
		_, y_train, _, _ = loadCorpus(single_corpus)

		temp = [np.count_nonzero(y_train[:,it].toarray().reshape(-1)) for it in range(y_train.shape[1])]
		temp = [1 if (u == 0) else u for u in temp]
		
		IRLbl = [max(temp)/temp[u] for u in range(len(temp))]

		MeanIR = np.average(IRLbl)

		IRLbl_rho = np.std(IRLbl)

		CVIR = IRLbl_rho/MeanIR


		print("{},{:.2f},{:.2f}".format(single_corpus, MeanIR, CVIR))

	return


""" Imbalance analysis """
def getImbalanceStats():
	imbalance_metrics()
	return



if __name__ == '__main__':
	random.seed(123)

	""" Corpora stats """
	# getDataStats()


	""" Imbalance stats """
	getImbalanceStats()


	""" Running experiments """
	# experiments()


	


