import numpy as np
from itertools import combinations
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier


class MRPS3():

	def getMostDistantPrototypes(self, in_set):
		duples = list(combinations(list(range(in_set.shape[0])), 2))

		max_dist = float('-inf')
		most_distant_duple = duples[0]
		for single_duple in duples:
			curr_dist = distance.euclidean(in_set[single_duple[0]], in_set[single_duple[1]])
			if curr_dist > max_dist:
				most_distant_duple = single_duple
				max_dist = curr_dist


		return most_distant_duple[0], most_distant_duple[1]


	def divideBIntoSubsets(self, B, p1, p2):
		B1_indexes = np.where(np.array([distance.euclidean(B[u], B[p1]) <= distance.euclidean(B[u], B[p2]) for u in range(B.shape[0])]) == True)[0]
		B2_indexes = np.array(sorted(list(set(range(B.shape[0])) - set(B1_indexes))))

		return B1_indexes, B2_indexes


	def checkClusterCommonLabel(self, in_elements):
    	# Checking whether there is a common label in ALL elements in the set:
		common_label_vec = [len(np.nonzero(in_elements[:,it]==1)[0]) == len(in_elements) for it in range(in_elements.shape[1])]
		# common_label_vec = [len(np.nonzero(in_elements[:,it]==1)[0]) >= len(in_elements)-5 for it in range(in_elements.shape[1])] # Relaxing the homogeneity

		return True if True in common_label_vec else False



	def generatePrototype(self, C):
		r = np.median(C[0], axis = 0)

		r_labelset = list()

		for it_label in range(C[1].shape[1]):
			n = len(np.where(C[1][:, it_label] == 1)[0])
			r_labelset.append(1) if n > C[1].shape[0]//2 else r_labelset.append(0)

		return (r, r_labelset)


	def reduceSet(self, X, y):
		self.X_init = X
		self.y_init = y

		Q = list()
		Q.append((self.X_init, self.y_init))
		CS = list()

		while len(Q) > 0:
			C = Q.pop() # Dequeing Q
			p1, p2 = self.getMostDistantPrototypes(C[0])

			B1_indexes, B2_indexes = self.divideBIntoSubsets(C[0], p1, p2)

			B1 = (C[0][B1_indexes], C[1][B1_indexes])
			B2 = (C[0][B2_indexes], C[1][B2_indexes])

			for single_partition in [B1, B2]:

				if self.checkClusterCommonLabel(single_partition[1]):
					CS.append(self.generatePrototype(single_partition))
				else:
					# print("B1 Non-homogeneous")
					Q.append(single_partition)
			# print("hello")


		self.X_out = np.array([CS[u][0] for u in range(len(CS))])
		self.y_out = np.array([CS[u][1] for u in range(len(CS))])

		return self.X_out, self.y_out



if __name__ == '__main__':
	X_train, y_train, feature_names, label_names = load_dataset('scene', 'train')
	X_test, y_test, feature_names, label_names = load_dataset('scene', 'test')

	X_red, y_red = MRPS3().reduceSet(X_train.toarray(), y_train.toarray())

	cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
	cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)

	y_pred_ori = cls_ori.predict(X_test)
	y_pred_red = cls_red.predict(X_test)

	print("Results:")
	print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
	print("\t - Hamming Loss: {:.3f} - Size {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
	print("Done!")