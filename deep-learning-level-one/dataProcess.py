
# 
import numpy as np 
import pandas as pd 

# check number of categories in the data frame by using for example: set(df['n_products_viewed'])

def dataProcess():
	df = pd.read_csv("ecommerce_data.csv")

	data = df.as_matrix() 

	n = data.shape[0]

	X = data[:, :-1]
	Y = data[:, -1]
	# Y = np.reshape(Y, (len(X), 1))

	D = X.shape[1]

	# one-hot-encode the time of date column.
	numCategory = len(set(X[:, -1]))
	Z = np.zeros((n, (D + numCategory - 1)))

	# normalizes feild 2 and 3.
	X[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
	X[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()

	Z[:, 0:(D - 1)] =  X[:, 0:(D -1)]

	for i in xrange(n):
		t = int(X[i, -1])

		Z[i, D + t - 1] = 1

	return Z, Y 

def getBinaryData():

	X, Y = dataProcess()

	X = X[Y <= 1]
	Y = Y[Y <= 1]

	return X, Y 


def yIndicator(y, k): # where by the the label vector, and k is number of categories in y.
	n = len(y)
	ind = np.zeros((n, k))
	for i in xrange(n):
		t = int(y[i])
		ind[i, t] = 1
	return ind 


