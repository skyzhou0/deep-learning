import sys 
sys.path.append("../machine_learning_examples/ann_logistic_extra")
import numpy as np 
import cPickle


from process import get_data
from dataProcess import dataProcess
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier


def main():
	X, Y = dataProcess()
	# X, Y = get_data()

	#tempY = np.reshape(Y, (len(X), 1))
	X, Y = shuffle(X, Y)
	trainLen = int(len(X) * 0.7)
	Xtrain = X[:trainLen]
	Ytrain = Y[:trainLen]
	Xtest = X[trainLen:]
	Ytest = Y[trainLen:]

	# build a network model.
	model = MLPClassifier(hidden_layer_sizes = (20, 20), max_iter = 2000)

	# train the model.
	model.fit(Xtrain, Ytrain)

	trainAccuracy = model.score(Xtrain, Ytrain)
	testAccuracy = model.score(Xtest, Ytest)

	print("train accuracy: ", trainAccuracy, "test accuracy: ", testAccuracy)

	# save the classifier
	with open('my_dumped_classifier.pkl', 'wb') as fid:
		cPickle.dump(model, fid)    

	# load it again
	with open('my_dumped_classifier.pkl', 'rb') as fid:
		model_loaded = cPickle.load(fid)

	trainAccuracy1 = model_loaded.score(Xtrain, Ytrain)
	testAccuracy1 = model_loaded.score(Xtest, Ytest)
	print("train accuracy: ", trainAccuracy1, "test accuracy: ", testAccuracy1)


if __name__ == '__main__':
	main()

