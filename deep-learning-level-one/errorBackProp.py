# backpropagation example for deep learning in python class.
# with sigmoid activation
#



import numpy as np
import matplotlib.pyplot as plt

# np.random.seed(1)

class BackPropagation(object):

	def __init__(self, T):
		self.T = T
	
	# note that we want to obtain the derivatives on those weights from last hidden layers to the output layer. The dimension of the the weight matrix will be
	# M x K, where M is number of hidden nodes, and K is number categories in the label target matrix.
	
	def derivative_w2(self, Z, Y):
		# note that Y is prediction, rather than the target label.

		N, K = self.T.shape     # T is the label target matrix, its diemsion is N x K, where K is number of categories.
		M = Z.shape[1]     # Z is the activation value, its dimension is N x M.

		res = Z.T.dot(self.T - Y)  # Z transpose is dimension of M X N, and (T - Y) is of N x K, which implies that res being M x K.

		return res 

	# Note that derivative of the weight(s) from input to hidden layer should be D x M.
	def derivative_w1(self, X, Z, Y, W2):
		N, D = X.shape 
		M, K = W2.shape 

		dZ = (self.T - Y).dot(W2.T) * Z * (1 - Z)  # dimension trace:  (T - Y) is (N x K); (W2.T) is (K x M), Z * (1 - Z) should be (N x M) in python. then (N x M) * (N x M) is elment by element.
		return X.T.dot(dZ)                    # X is (N x D), and dZ should be (N x M).

	# Note that the bias from last hidden layer to output layer shoudl be K x 1.
	def derivative_b2(self, Y):
		return (self.T - Y).sum(axis=0)

    # Note that the bias from input to hidden layer shoudl be M x 1.
	def derivative_b1(self, Y, W2, Z):
		return ((self.T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

	def cost(self, Y):
		tot = self.T * np.log(Y)
		return tot.sum()


class ForwardDL(object):
	def __init__(self, input):
		self.input = input 

	def tanh(self, W, b1):
		return ( np.exp(self.input.dot(W) + b1) - np.exp(-(self.input.dot(W) + b1)) ) / ( np.exp(self.input.dot(W) + b1) + np.exp(-(self.input.dot(W) + b1)) )

	def relu(self, W, b1):
		return (self.input.dot(W) + b1) * ( (self.input.dot(W) + b1) > 0 )

	def forward(self, activations, layers, W, b1, V, c):

		if layers == 1:

			# let assume input dimension being n x D, W should be D, M.
			# n = self.input.shape[0]  	# number of samples.
			# D = self.input.shape[1]  	# features' dimension.
			# W = np.random.randn(D, M)   # weights at the first hidden layer.
			# b1 = np.random.randn(M)     # bias at the first hidden layer.
			# V = np.random.randn(M, K)
			# c = np.random.randn(K)

			if activations == "sigmoid":
				Z = 1 / (1 + np.exp(-(self.input.dot(W) + b1)))
			elif activations == "tanh":
				Z = self.tanh(W, b1)
			elif activations == "relu":
				Z = self.relu(W, b1)
		else:
			print "the class does not have capability for deep network yet."

		A = Z.dot(V) + c 

		expA = np.exp(A)

		return expA / expA.sum(axis = 1, keepdims = True), Z

	def classificationRate(self, prediction, label):

		predictionClass = np.argmax(prediction, axis = 1)

		return np.mean(predictionClass == label)


