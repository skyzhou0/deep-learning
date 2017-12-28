from __future__ import print_function, division

import pandas as pd 
import numpy as np 
from sklearn.utils import shuffle
import _pickle as cPickle

from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier



def main():

	data = pd.read_csv("spambase.data").as_matrix()
	np.random.shuffle(data)

	X = data[:, :48]
	Y = data[:, -1]

	print("traget to base ratio is: {}".format(Y.sum()/len(Y)))

	Xtrain = X[:-1000, :]
	Ytrain = Y[:-1000]
	Xtest = X[-1000:, :]
	Ytest = Y[-1000:]

	# a. -- build a neural network model.
	t0_MLP = datetime.now()
	model = MLPClassifier(hidden_layer_sizes = (20, 20), max_iter = 2000)

	# train the model.
	model.fit(Xtrain, Ytrain)

	trainAccuracyMLP = model.score(Xtrain, Ytrain)
	testAccuracyMLP = model.score(Xtest, Ytest)

	print("train accuracy: ", trainAccuracyMLP, "test accuracy: ", testAccuracyMLP)

	# save the classifier
	with open('spam-detector-pickles/MLP_classifier.pkl', 'wb') as fid:
		cPickle.dump(model, fid)    

	# load it again
	with open('spam-detector-pickles/MLP_classifier.pkl', 'rb') as fid:
		modelLoadedMLP = cPickle.load(fid)

	testAccuracyLoadMLP = modelLoadedMLP.score(Xtest, Ytest)
	print("test accuracy: ", testAccuracyLoadMLP)

	print("Elapsted time for MLP:", datetime.now() - t0_MLP)


	# b. -- build a Naive Bayes model.
	t0_NB = datetime.now()
	model = MultinomialNB()

	# train the model.
	model.fit(Xtrain, Ytrain)

	trainAccuracyNB = model.score(Xtrain, Ytrain)
	testAccuracyNB = model.score(Xtest, Ytest)

	print("train accuracy: ", trainAccuracyNB, "test accuracy: ", testAccuracyNB)

	# save the classifier
	with open('spam-detector-pickles/NB_classifier.pkl', 'wb') as fid:
		cPickle.dump(model, fid)    

	# load it again
	with open('spam-detector-pickles/NB_classifier.pkl', 'rb') as fid:
		modelLoadedNB = cPickle.load(fid)

	testAccuracyLoadNB = modelLoadedNB.score(Xtest, Ytest)
	print("test accuracy: ", testAccuracyLoadNB)

	print("Elapsted time for NB:", datetime.now() - t0_NB)

	# c. -- build a Adaboost model.
	t0_Ada = datetime.now()
	model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
        					   n_estimators=100,
        					   learning_rate=0.03,
        					   random_state=101
        					  )

	# train the model.
	model.fit(Xtrain, Ytrain)

	trainAccuracyAda = model.score(Xtrain, Ytrain)
	testAccuracyAda = model.score(Xtest, Ytest)

	print("train accuracy: ", trainAccuracyAda, "test accuracy: ", testAccuracyAda)

	# save the classifier
	with open('spam-detector-pickles/Ada_classifier.pkl', 'wb') as fid:
		cPickle.dump(model, fid)    

	# load it again
	with open('spam-detector-pickles/Ada_classifier.pkl', 'rb') as fid:
		modelLoadedAda = cPickle.load(fid)

	testAccuracyLoadAda = modelLoadedAda.score(Xtest, Ytest)
	print("test accuracy: ", testAccuracyLoadAda)

	print("Elapsted time for Adaboost:", datetime.now() - t0_Ada)


	# d. -- build a Gradient Boost model.
	t0_GB = datetime.now()
	model = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=100, subsample=1.0, 
									   #criterion='friedman_mse',
									   min_samples_split=2, min_samples_leaf=1,max_depth=20, 
									   max_features="auto", random_state = 101
									  )
	# train the model.
	model.fit(Xtrain, Ytrain)

	trainAccuracyGB = model.score(Xtrain, Ytrain)
	testAccuracyGB = model.score(Xtest, Ytest)

	print("train accuracy: ", trainAccuracyGB, "test accuracy: ", testAccuracyGB)

	# save the classifier
	with open('spam-detector-pickles/GB_classifier.pkl', 'wb') as fid:
		cPickle.dump(model, fid)    

	# load it again
	with open('spam-detector-pickles/GB_classifier.pkl', 'rb') as fid:
		modelLoadedGB = cPickle.load(fid)

	testAccuracyLoadGB = modelLoadedGB.score(Xtest, Ytest)
	print("test accuracy: ", testAccuracyLoadGB)

	print("Elapsted time for Adaboost:", datetime.now() - t0_GB)
	



if __name__ == '__main__':
	main()

# The End.