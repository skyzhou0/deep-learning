from __future__ import print_function, division

#from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future
import numpy as np 
import matplotlib.pyplot as plt 

from errorBackProp import BackPropagation, ForwardDL
from dataProcess import getBinaryData, yIndicator
from sklearn.utils import shuffle


def main():	

	X, Y = getBinaryData()
	X, Y = shuffle(X, Y)
	Y = Y.astype(np.int32)

	M = 5 # number of hidden nodes in the hidden layer.
	D = X.shape[1] 
	K = len(set(Y))

	Xtrain = X[:-100]
	Ytrain = Y[:-100]
	Ttrain = yIndicator(Ytrain, K)
	Xtest = X[-100:]
	Ytest = Y[-100:]
	Ttest = yIndicator(Ytest, K)

	W1 = np.random.randn(D, M)
	b1 = np.random.randn(M)
	W2 = np.random.randn(M, K)
	b2 = np.random.randn(K)

	forwardObj = ForwardDL(input = Xtrain)
	backPropObj = BackPropagation(Ttrain)
	trainCost = []
	learningRate = 0.001

	print("target to base ratio: ", float(Ytrain.sum()) / len(Ytrain) )

	for i in xrange(10000):
		predictionTrain, Ztrain = forwardObj.forward("sigmoid",1, W1, b1, W2, b2)
		cost = backPropObj.cost(predictionTrain)
		trainCost.append(cost)

		W2 += learningRate * backPropObj.derivative_w2(Ztrain, predictionTrain)
        b2 += learningRate * backPropObj.derivative_b2( predictionTrain)
        W1 += learningRate * backPropObj.derivative_w1(Xtrain, Ztrain,  predictionTrain, W2)
        b1 += learningRate * backPropObj.derivative_b1( predictionTrain, W2, Ztrain)

        if i % 1000 == 0:
        	print(i, cost )

	print("final classification rate for the train data: ", forwardObj.classificationRate(predictionTrain, Ytrain))

    # for prediction on the test data.
	forwardObj_test = ForwardDL(input = Xtest)
	predictionTtest, Ztest = forwardObj_test.forward("sigmoid",1, W1, b1, W2, b2)
	print("final classification rate for the train data: ", forwardObj_test.classificationRate(predictionTtest, Ytest))

    # plot the training cost.
	legendTrain = plt.plot(trainCost, label = 'train cost')
	plt.legend(legendTrain)
	plt.show()
	

if __name__ == '__main__':

	main()
