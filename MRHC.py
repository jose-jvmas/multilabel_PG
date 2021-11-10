import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier





class MRHC():

	def checkClusterCommonLabel(self, in_elements):
		# Checking whether there is a common label in ALL elements in the set:
		common_label_vec = [len(np.nonzero(in_elements[:,it]==1)[0]) == len(in_elements) for it in range(in_elements.shape[1])]

		return True if True in common_label_vec else False



	def reduceSet(self, X, y):
		self.X_init = X
		self.y_init = y

		Q = list()
		Q.append((self.X_init, self.y_init))
		CS = list()

		while len(Q) > 0:
			C = Q.pop() # Dequeing Q
			if self.checkClusterCommonLabel(C[1]):
				r = np.median(C[0], axis = 0)

				r_labelset = list()

				for it_label in range(C[1].shape[1]):
					n = len(np.where(C[1][:, it_label] == 1)[0])
					r_labelset.append(1) if n > C[1].shape[0]//2 else r_labelset.append(0)

				CS.append((r, r_labelset))
			else:
				M = list() # Initializing set of label-centroids

				# Obtaining set of label-centroids:
				for it_label in range(C[1].shape[1]):
					label_indexes = np.where(C[1][:,it_label] == 1)[0]
					if len(label_indexes) > 0:
						M.append(np.median(C[0][label_indexes,:], axis = 0))
				M = np.array(M) # label X n_features

				resulting_labels = list(range(C[0].shape[0]))
				if C[0].shape[0] >= M.shape[0]:
					# Kmeans with M as initial centroids:
					kmeans = KMeans(n_clusters = M.shape[0], init = M)
					kmeans.fit(np.array(C[0] + 0.001, dtype = 'double'))
					resulting_labels = kmeans.labels_
    				
				
				# Create new groups and enqueue them:
				for cluster_index in np.unique(resulting_labels):
					indexes = np.where(resulting_labels == cluster_index)[0]
					Q.append((C[0][indexes], C[1][indexes]))


		self.X_out = np.array([CS[u][0] for u in range(len(CS))])
		self.y_out = np.array([CS[u][1] for u in range(len(CS))])

		return self.X_out, self.y_out



if __name__ == '__main__':
	# from skmultilearn.dataset import available_data_sets

	# set([x[0] for x in available_data_sets().keys()])
	# set([x[1] for x in available_data_sets().keys()])


	X_train, y_train, feature_names, label_names = load_dataset('scene', 'train')
	X_test, y_test, feature_names, label_names = load_dataset('scene', 'test')

	mrhc = MRHC()
	X_red, y_red = mrhc.reduceSet(X_train.toarray(), y_train.toarray())

	cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
	cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)


	y_pred_ori = cls_ori.predict(X_test)
	y_pred_red = cls_red.predict(X_test)

	print("Results:")
	print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
	print("\t - Hamming Loss: {:.3f} - Size {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
