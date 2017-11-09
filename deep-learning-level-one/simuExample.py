

from __future__ import print_function, division
#from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import numpy as np 
import matplotlib.pyplot as plt 

from errorBackProp import BackPropagation, ForwardDL

def main():
    # create the data
    Nclass = 500
    D = 2 # dimensionality of input
    M = 3 # hidden layer size
    K = 3 # number of classes

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)
    # turn Y into an indicator matrix for training
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1

    # let's see what it looks like
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()

    # randomly initialize weights
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    backPropObj = BackPropagation(T)
    forwardObj = ForwardDL(X)

    learning_rate = 1e-3
    costs = []
    for epoch in range(1000):

        output, hidden = forwardObj.forward("sigmoid", 1, W1, b1, W2, b2)
        if epoch % 100 == 0:
            c = backPropObj.cost(output)
            # P = np.argmax(output, axis=1)
            r = forwardObj.classificationRate(output, Y)
            print("cost:", c, "classification_rate:", r)
            costs.append(c)

        # this is gradient ASCENT, not DESCENT
        # be comfortable with both!
        # oldW2 = W2.copy()
        W2 += learning_rate * backPropObj.derivative_w2(hidden, output)
        b2 += learning_rate * backPropObj.derivative_b2( output)
        W1 += learning_rate * backPropObj.derivative_w1(X, hidden,  output, W2)
        b1 += learning_rate * backPropObj.derivative_b1( output, W2, hidden)

    plt.plot(costs)
    plt.show()


if __name__ == '__main__':
    main()

