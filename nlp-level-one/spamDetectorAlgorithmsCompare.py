# date: 25th Dec 2017.

# thank you lazyprogrammer for the teaching.

# notes: In this example, we compare the accuracy and the computational time of 4 existing sklearn classifiers.
# The number of iterations has been set as 5 (as sys.args whilst run the python script) and we found that 
# the average accuracy: MLP vs. Naive Bayes vs. Ada Boost vs. Gradient Boost are: 0.9286 vs. 0.852 vs. 0.902 vs. 0.9126
# the computational time (in secs): MLP vs. Naive Bayes vs. Ada Boost vs. Gradient Boost are: 6.2 vs. 0.012 vs. 3.5 vs. 28.5.

# One important note is that we can see all 5 test accuracies are the same for Naive Bayes and Ada boost, this is due to the iteration loops has been set 
# at the model level rather than at the data level. One can make a slight change on the existing code to achieve that. However I have not choosen to do that is
# it already allow to see that MLP and Gradient boost classifiers seems to be the most accurate ones given this data. However they require the most computational
# time, with Gradient Boost being the worst among all 4 classifiers (one should seek to use xgboost or light GBM whereby the parallelisation took place at the iterations(tree) level).

from __future__ import print_function, division

import pandas as pd 
import numpy as np 
import sys
from sklearn.utils import shuffle
import _pickle as cPickle

from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier


def main(iterations):

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
	trainAccuracyMLPArray = []
	testAccuracyMLPArray = []

	for i in range(iterations):
		model = MLPClassifier(hidden_layer_sizes = (20, 20), max_iter = 2000)

		# train the model.
		model.fit(Xtrain, Ytrain)

		trainAccuracyMLP = model.score(Xtrain, Ytrain)
		testAccuracyMLP = model.score(Xtest, Ytest)

		trainAccuracyMLPArray.append(trainAccuracyMLP)
		testAccuracyMLPArray.append(testAccuracyMLP)

	print(testAccuracyMLPArray)
	print("Averge train accuracy: ", sum(trainAccuracyMLPArray)/iterations, "Averge test accuracy: ", sum(testAccuracyMLPArray)/iterations)

	# save the las model classifier
	with open('spam-detector-pickles/MLP_classifier_last.pkl', 'wb') as fid:
		cPickle.dump(model, fid)    

	# load it again
	with open('spam-detector-pickles/MLP_classifier_last.pkl', 'rb') as fid:
		modelLoadedMLP = cPickle.load(fid)

	testAccuracyLoadMLP = modelLoadedMLP.score(Xtest, Ytest)
	print("test accuracy: ", testAccuracyLoadMLP)

	print("Elapsted time for MLP:", datetime.now() - t0_MLP)


	# b. -- build a Naive Bayes model.
	t0_NB = datetime.now()
	trainAccuracyNBArray = []
	testAccuracyNBArray = []

	for i in range(iterations):
		model = MultinomialNB()

		# train the model.
		model.fit(Xtrain, Ytrain)

		trainAccuracyNB = model.score(Xtrain, Ytrain)
		testAccuracyNB = model.score(Xtest, Ytest)

		trainAccuracyNBArray.append(trainAccuracyNB)
		testAccuracyNBArray.append(testAccuracyNB)

	print(testAccuracyNBArray)
	print("Averge train accuracy: ", sum(trainAccuracyNBArray)/iterations, "Averge test accuracy: ", sum(testAccuracyNBArray)/iterations)

	# save the las model classifier
	with open('spam-detector-pickles/NB_classifier_last.pkl', 'wb') as fid:
		cPickle.dump(model, fid)    

	# load it again
	with open('spam-detector-pickles/NB_classifier_last.pkl', 'rb') as fid:
		modelLoadedNB = cPickle.load(fid)

	testAccuracyLoadNB = modelLoadedNB.score(Xtest, Ytest)
	print("test accuracy: ", testAccuracyLoadNB)

	print("Elapsted time for NB:", datetime.now() - t0_NB)

	# c. -- build a Adaboost model.
	t0_Ada = datetime.now()
	trainAccuracyAdaArray = []
	testAccuracyAdaArray = []

	for i in range(iterations):
		model = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=2),
	        					   n_estimators=100,
	        					   learning_rate=0.03
	        					   #random_state=101
	        					  )

		# train the model.
		model.fit(Xtrain, Ytrain)

		trainAccuracyAda = model.score(Xtrain, Ytrain)
		testAccuracyAda = model.score(Xtest, Ytest)

		trainAccuracyAdaArray.append(trainAccuracyAda)
		testAccuracyAdaArray.append(testAccuracyAda)

	print(testAccuracyAdaArray)
	print("Averge train accuracy: ", sum(trainAccuracyAdaArray)/iterations, "Averge test accuracy: ", sum(testAccuracyAdaArray)/iterations)

	# save the las model classifier
	with open('spam-detector-pickles/Ada_classifier_last.pkl', 'wb') as fid:
		cPickle.dump(model, fid)    

	# load it again
	with open('spam-detector-pickles/Ada_classifier_last.pkl', 'rb') as fid:
		modelLoadedAda = cPickle.load(fid)

	testAccuracyLoadAda = modelLoadedAda.score(Xtest, Ytest)
	print("test accuracy: ", testAccuracyLoadAda)

	print("Elapsted time for Adaboost:", datetime.now() - t0_Ada)


	# d. -- build a Gradient Boost model.
	t0_GB = datetime.now()
	trainAccuracyGBArray = []
	testAccuracyGBArray = []

	for i in range(iterations):
		model = GradientBoostingClassifier(loss='deviance', learning_rate=0.01, n_estimators=100, subsample=1.0, 
										   #criterion='friedman_mse',
										   min_samples_split=2, min_samples_leaf=1,max_depth=20, 
										   max_features="auto"#, random_state = 101
										  )
		# train the model.
		model.fit(Xtrain, Ytrain)

		trainAccuracyGB = model.score(Xtrain, Ytrain)
		testAccuracyGB = model.score(Xtest, Ytest)

		trainAccuracyGBArray.append(trainAccuracyGB)
		testAccuracyGBArray.append(testAccuracyGB)

	print(testAccuracyGBArray)
	print("Averge train accuracy: ", sum(trainAccuracyGBArray)/iterations, "Averge test accuracy: ", sum(testAccuracyGBArray)/iterations)

	# save the las model classifier
	with open('spam-detector-pickles/GB_classifier_last.pkl', 'wb') as fid:
		cPickle.dump(model, fid)    

	# load it again
	with open('spam-detector-pickles/GB_classifier_last.pkl', 'rb') as fid:
		modelLoadedGB = cPickle.load(fid)

	testAccuracyLoadGB = modelLoadedGB.score(Xtest, Ytest)
	print("test accuracy: ", testAccuracyLoadGB)

	print("Elapsted time for Adaboost:", datetime.now() - t0_GB)
	


if __name__ == '__main__':
	main(iterations = int(sys.argv[1]))

# The End.