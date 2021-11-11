import numpy as np
from itertools import combinations
from scipy.spatial import distance
from skmultilearn.dataset import load_dataset



class MChen():



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


	def setContainSeveralClasses(self, in_set):
		return True if np.unique(in_set, axis = 1).shape[1] > 1 else False



	def generatePrototype(self, C):
		r = np.median(C[0], axis = 0)

		r_labelset = list()

		for it_label in range(C[1].shape[1]):
			n = len(np.where(C[1][:, it_label] == 1)[0])
			r_labelset.append(1) if n > C[1].shape[0]//2 else r_labelset.append(0)

		return (r, r_labelset)



	def reduceSet(self, X, y, n_out):
		self.X = X
		self.y = y

		C = {1 : (self.X, self.y)}


		# Step 2:
		bc = 1
		Qchosen = 1
		prototypeIndexesQchosen = self.getMostDistantPrototypes(C[1][0])

		for _ in range(n_out - 1):

			# Step 3:
			B = C[Qchosen]
				
			# Step 4:
			p1, p2 = prototypeIndexesQchosen

			# Step 5:
			B1_indexes, B2_indexes = self.divideBIntoSubsets(B[0], p1, p2)
			B1 = (B[0][B1_indexes], B[1][B1_indexes])
			B2 = (B[0][B2_indexes], B[1][B2_indexes])

			# Step 6:
			i = Qchosen
			bc += 1
			C[i] = B1
			C[bc] = B2

			# Step 7:
			I1 = dict()
			I2 = dict()

			for index, currentSubset in C.items():
				if self.setContainSeveralClasses(currentSubset[0]):
					I1[index] = currentSubset
				else:
					I2[index] = currentSubset
			
			I = I1 if len(I1) > 0 else I2 

			maxDist = float('-inf')
			for index, currentSubset in I.items():
				if currentSubset[0].shape[0] > 1:
					p1, p2 = self.getMostDistantPrototypes(currentSubset[0])
					curDist = distance.euclidean(currentSubset[0][p1], currentSubset[0][p2])

					if maxDist < curDist:
						maxDist = curDist
						Qchosen = index
						prototypeIndexesQchosen = (p1, p2)
		

		self.X_out = list()
		self.y_out = list()
		for single_cluster in C.values():
			prot, labels = self.generatePrototype(single_cluster)
			self.X_out.append(prot)
			self.y_out.append(labels)

		return np.array(self.X_out), np.array(self.y_out)



if __name__ == '__main__':
	X_train, y_train, feature_names, label_names = load_dataset('scene', 'train')


	X_red, y_red = MChen().reduceSet(X_train.toarray(), y_train.toarray(), 754)