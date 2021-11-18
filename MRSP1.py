import numpy as np
from itertools import combinations
from scipy.spatial import distance
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier



class MRSP1():

	@staticmethod
	def getFileName(*params):
		return 'MRSP1_' + str(params[0])


	def getMostDistantPrototypes(self, in_list):
		duples = list(combinations(in_list, 2))

		max_dist = float('-inf')
		most_distant_duple = duples[0]
		for single_duple in duples:
			curr_dist = self.distances_dict[single_duple[0]][single_duple[1]]
			if curr_dist > max_dist:
				most_distant_duple = single_duple
				max_dist = curr_dist


		return most_distant_duple[0], most_distant_duple[1]



	def divideBIntoSubsets(self, B, p1, p2):
		B1_indexes = [B[idx] for idx in np.where(np.array([self.distances_dict[min(u, p1)][max(u, p1)] <= self.distances_dict[min(u, p2)][max(u, p2)] for u in B]) == True)[0]]
		B2_indexes = np.array(sorted(list(set(B) - set(B1_indexes))))

		return B1_indexes, B2_indexes


	def setContainSeveralClasses(self, in_set):
		return True if np.unique(self.y[in_set], axis = 0).shape[0] > 1 else False



	def generatePrototypes(self, indexes):
		X = self.X[indexes]
		y = self.y[indexes]

		X_out = list()
		y_out = list()

		# Retrieving different classes:
		unique_classes = np.unique(y, axis = 0)

		for single_unique_class in unique_classes:
			# Retrieving indexes of the elements with the same multilabel class:
			common_indexes = list()
			for it_prot in range(len(indexes)):
				if np.all(y[it_prot] == single_unique_class):
					common_indexes.append(it_prot)

			r = np.median(X[common_indexes], axis = 0)
			r_labelset = single_unique_class
			
			X_out.append(r)
			y_out.append(r_labelset)

		return (X_out, y_out)



	def computePairwiseDistances(self):

		self.distances_dict = {}
		for it_row in range(self.X.shape[0]):
			current_row_dict = {}
			for it_col in range(it_row, self.X.shape[0]):
				current_row_dict[it_col] = distance.euclidean(self.X[it_row], self.X[it_col])
			self.distances_dict[it_row] = current_row_dict
		return



	def reduceSet(self, X, y, params):
		self.X = X
		self.y = y

		# Number of out elements:
		n_out = int(params * self.X.shape[0]/100)

		self.computePairwiseDistances()

		# Out elements:
		C = {1 : list(range(self.X.shape[0]))}


		# Step 2:
		bc = 1
		Qchosen = 1
		prototypeIndexesQchosen_dict = {1: self.getMostDistantPrototypes(C[1])}
		

		for _ in range(n_out - 1):

			# Step 3:
			B = C[Qchosen]
				
			# Step 4:
			p1, p2 = prototypeIndexesQchosen_dict[Qchosen]

			# Step 5:
			B1_indexes, B2_indexes = self.divideBIntoSubsets(B, p1, p2)
			B1 = B1_indexes
			B2 = B2_indexes

			# Step 6:
			i = Qchosen
			bc += 1
			C.pop(i)
			C[i] = B1
			C[bc] = B2
			prototypeIndexesQchosen_dict[i] = -1
			prototypeIndexesQchosen_dict[bc] = -1

			# Step 7:
			I1 = dict()
			I2 = dict()

			for index, currentSubset in C.items():
				if len(currentSubset) > 0:
					if self.setContainSeveralClasses(currentSubset):
						I1[index] = currentSubset
					else:
						I2[index] = currentSubset
			

			I = I1 if len(I1) > 0 else I2 


			maxDist = float('-inf')
			for index, currentSubset in I.items():
				if len(currentSubset) > 1:
					if prototypeIndexesQchosen_dict[index] == -1:
						prototypeIndexesQchosen_dict[index] = self.getMostDistantPrototypes(currentSubset)
					
					p1, p2 = prototypeIndexesQchosen_dict[index]

					curDist = self.distances_dict[min(p1, p2)][max(p1, p2)]

					if maxDist < curDist:
						maxDist = curDist
						Qchosen = index
		

		self.X_out = list()
		self.y_out = list()
		for single_cluster in C.values():
			if len(single_cluster) > 0:
				prot, labels = self.generatePrototypes(single_cluster)
				self.X_out.extend(prot)
				self.y_out.extend(labels)

		return np.array(self.X_out), np.array(self.y_out)



if __name__ == '__main__':
	X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
	X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')

	X_red, y_red = MRSP1().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), 90)


	cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
	cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)

	y_pred_ori = cls_ori.predict(X_test)
	y_pred_red = cls_red.predict(X_test)

	print("Results:")
	print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
	print("\t - Hamming Loss: {:.3f} - Size {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
	print("Done!")




