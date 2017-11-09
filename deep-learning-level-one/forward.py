# 

import numpy as np 

import matplotlib.pyplot as plt 

class ForwardDL(object):
	def __init__(self, input):
		self.input = input 

	def tanh(self, W, b1):
		return ( np.exp(self.input.dot(W) + b1) - np.exp(-(self.input.dot(W) + b1)) ) / ( np.exp(self.input.dot(W) + b1) + np.exp(-(self.input.dot(W) + b1)) )

	def relu(self, W, b1):
		return (self.input.dot(W) + b1) * ( (self.input.dot(W) + b1) > 0 )

	def forward(self, activations, layers, M, K):

		if layers == 1:

			# let assume input dimension being n x D, W should be D, M.
			n = self.input.shape[0]  	# number of samples.
			D = self.input.shape[1]  	# features' dimension.
			W = np.random.randn(D, M)   # weights at the first hidden layer.
			b1 = np.random.randn(M)     # bias at the first hidden layer.
			V = np.random.randn(M, K)
			c = np.random.randn(K)

			if activations == "sigmoid":
				Z = 1 / (1 + np.exp(-(self.input.dot(W) + b1)))
			elif activations == "tanh":
				Z = self.tanh(W, b1)
			elif activations == "relu":
				X = self.relu(W, b1)
		else:
			print "the class does not have capability for deep network yet."

		A = Z.dot(V) + c 

		expA = np.exp(A)

		return expA / expA.sum(axis = 1, keepdims = True), Z 

	def classificationRate(self, prediction, label):

		predictionClass = np.argmax(prediction, axis = 1)

		return np.mean(predictionClass == label)

def main():	

	# simulate example - Gaussian Clouds.
	Nclass = 500 
	X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
	X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
	X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])

	X = np.vstack([X1, X2, X3])
	Y = np.array([0] * Nclass + [1] * Nclass + [2] * Nclass)

	plt.scatter(X[:, 0], X[:, 1], c = Y, s = 100, alpha = 0.5)
	plt.show()

	forwardObj = ForwardDL(input = X)

	prediction_Y_given_X_sigmoidActivate = forwardObj.forward("sigmoid", 1, M = 3, K = len(set(Y))) # note that  activation function can be set as the default.
	prediction_Y_given_X_tanhActivate = forwardObj.forward("tanh", 1, M = 3, K = len(set(Y))) # note that  activation function can be set as the default.
	prediction_Y_given_X_reluActivate = forwardObj.forward("relu", 1, M = 3, K = len(set(Y))) 


	# print prediction_Y_given_X_sigmoidActivate
	print "sigmoid activation classification rate is: ", forwardObj.classificationRate(prediction_Y_given_X_sigmoidActivate, Y)
	print "tanh activation classification rate is: ", forwardObj.classificationRate(prediction_Y_given_X_tanhActivate, Y)
	print "relu activation classification rate is: ", forwardObj.classificationRate(prediction_Y_given_X_reluActivate, Y)


if __name__ == '__main__':

	main()








