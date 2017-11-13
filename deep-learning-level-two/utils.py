

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
import cPickle
   


def getNormalizedData(fileDirectory, fileName):

	file = fileDirectory + fileName

	df = pd.read_csv(file)

	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)

	X = data[:, 1:]
	mu = X.mean(axis = 0)
	std = X.std(axis = 0)

	np.place(std, std == 0, 1)  # Change elements of an array based on conditional (std = 0) and input values
	X = (X - mu)/std  # normalization helps to overcome the limitation of both sigmoid and tanh functions where only small interval of X 
					  # where both functions being non-linear.
	Y = data[:, 0]

	return X, Y 

def getPCAData(fileDirectory, fileName, pcaVarianceThreshold):
	file = fileDirectory + fileName

	df = pd.read_csv(file)

	data = df.as_matrix().astype(np.float32)
	np.random.shuffle(data)

	X = data[:, 1:]
	mu = X.mean(axis = 0)
	
	X = (X - mu)  # It has been suggested that it is good practice to centralized the data before PCA transformation.
	pca = PCA()
	Z = pca.fit_transform(X)

	reductionIndex = 0
	accumulativePCAVariance = 0
	for i in range(len(Z)):
		if accumulativePCAVariance <= pcaVarianceThreshold:
			reductionIndex += 1
			accumulativePCAVariance += pca.explained_variance_ratio_[i]
	
	# print "accumulativePCAVariance:", accumulativePCAVariance
	# print reductionIndex

	Y = data[:, 0]
	# save the classifier
	with open('my_dumped_classifier.pkl', 'wb') as fid:
		cPickle.dump(pca, fid) 

	return Z, Y, pca, reductionIndex, mu, X


# fileDirectory = "../large_files/"
# fileName = "train.csv"
# pcaVarianceThreshold = 0.98
# X, Y = getNormalizedData(fileDirectory, fileName)
# X, Y, pca, reductionIndex, mu, test = getPCAData(fileDirectory, fileName, pcaVarianceThreshold)

# # load it again
# with open('my_dumped_classifier.pkl', 'rb') as fid:
#     pca_loaded = cPickle.load(fid)

# h = pca_loaded.fit_transform(test)  # We should see h is identifica to X. 


class ForwardPredictLogits(object):

	def __init__(self, input):
		self.input = input 

	def forward(self, W, b):
    # softmax
		a = self.input.dot(W) + b
		expa = np.exp(a)
		prediction = expa / expa.sum(axis=1, keepdims=True)
		return prediction

	def predict(self, p_y):
		return np.argmax(p_y, axis=1)

	def error_rate(self, p_y, t):
		prediction = self.predict(p_y)
		return np.mean(prediction != t)

class ModelTrainLogits(object):

	def __init__(self, t):
		self.t = t

	def cost(self, p_y):
		tot = self.t * np.log(p_y)
		return -tot.sum()

	def gradW(self, y, X):
		return X.T.dot(self.t - y)

	def gradb(self, y):
		return (self.t - y).sum(axis=0)


def yIndicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, len(set(y))))

    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def pcaReductionIndex(X, pca, pcaVarianceThreshold):

	reductionIndex = 0
	accumulativePCAVariance = 0
	for i in range(X.shape[1]):
		if accumulativePCAVariance <= pcaVarianceThreshold:
			reductionIndex += 1
			accumulativePCAVariance += pca.explained_variance_ratio_[i]
		
	return reductionIndex, accumulativePCAVariance





	





