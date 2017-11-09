
from __future__ import print_function, division 

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 

# 
def init_weights(shape):
	return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def forward(X, W1, b1, W2, b2):
	Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
	return tf.matmul(Z, W2) + b2 


def main():

	N = 500
	X1 = np.random.randn(N, 2) + np.array([0, -2])
	X2 = np.random.randn(N, 2) + np.array([2, 2])
	X3 = np.random.randn(N, 2) + np.array([-2, 2])

	X = np.vstack([X1, X2, X3]).astype(np.float32)
	Y = np.array([0] * N + [1] * N + [2] * N  )

	# plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
	# plt.show()

	N = len(Y)
	K = len(set(Y))
	D = X.shape[1]
	M = 3

	T = np.zeros((N, K))

	for i in range(len(Y)):
		T[i, Y[i]] = 1

	tfX = tf.placeholder(tf.float32, [None, D])
	tfY = tf.placeholder(tf.float32, [None, K])

	W1 = init_weights([D, M])
	b1 = init_weights([M])
	W2 = init_weights([M, K])
	b2 = init_weights([K])

	prediction_Y_given_x = forward(X, W1, b1, W2, b2)

	cost = tf.reduce_mean(
		tf.nn.softmax_cross_entropy_with_logits(
			logits = prediction_Y_given_x, 
			labels = tfY
		)
	)

	trainOp = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
	predictionOp = tf.argmax(prediction_Y_given_x, 1)

	sess = tf.Session()
	init = tf.initialize_all_variables()
	sess.run(init)

	for i in xrange(1000):
		sess.run(trainOp, feed_dict = {tfX: X,
									   tfY: T})
		predictionClass = sess.run(predictionOp, feed_dict = {tfX: X, tfY: T})
		if i % 100 == 0:
			print(np.mean(Y == predictionClass))

if __name__ == '__main__':
	main()


