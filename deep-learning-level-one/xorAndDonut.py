# revisiting the XOR and donut problems to show how features
# can be learned automatically using neural networks.


import numpy as np
import matplotlib.pyplot as plt

# for binary classification! no softmax here
class ForwardDL(object):

    def __init__(self, input):
        self.input = input 

    def tanh(self, W, b1):
        # Z = np.tanh(X.dot(W1) + b1)
        # return ( np.exp(self.input.dot(W) + b1) - np.exp(-(self.input.dot(W) + b1)) ) / ( np.exp(self.input.dot(W) + b1) + np.exp(-(self.input.dot(W) + b1)) )
        return np.tanh(self.input.dot(W) + b1)

    def relu(self, W, b1):
        return (self.input.dot(W) + b1) * ( (self.input.dot(W) + b1) > 0 )

    def forward(self, activations, layers, W, b1, V, c):

        if layers == 1:

            # let assume input dimension being n x D, W should be D, M.
            # n = self.input.shape[0]   # number of samples.
            # D = self.input.shape[1]   # features' dimension.
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


class BackPropagation(object):

    def __init__(self, T, activations):
        self.T = T
        self.activations = activations
    
    # note that we want to obtain the derivatives on those weights from last hidden layers to the output layer. The dimension of the the weight matrix will be
    # M x K, where M is number of hidden nodes, and K is number categories in the label target matrix.
    
    def derivative_w2(self, Z, Y):
        # note that Y is prediction, rather than the target label.

        # N, K = self.T.shape     # T is the label target matrix, its diemsion is N x K, where K is number of categories.
        # M = Z.shape[1]     # Z is the activation value, its dimension is N x M.

        res = Z.T.dot(self.T - Y)  # Z transpose is dimension of M X N, and (T - Y) is of N x K, which implies that res being M x K.
        # res = (self.T - Y).dot(Z)

        return res 

    # Note that derivative of the weight(s) from input to hidden layer should be D x M.
    def derivative_w1(self, X, Z, Y, W2):
        N, D = X.shape 
        M, K = W2.shape 
        if self.activations == "sigmoid":
            # dZ = np.outer(self.T-Y, W2) * Z * (1 - Z)
            dZ = (self.T - Y).dot(W2.T) * Z * (1 - Z)  # dimension trace:  (T - Y) is (N x K); (W2.T) is (K x M), Z * (1 - Z) should be (N x M) in python. then (N x M) * (N x M) is elment by element.
        elif self.activations == "tanh":
            # dZ = np.outer(self.T-Y, W2) * (1 - Z * Z) # this is for tanh activation
            dZ = (self.T - Y).dot(W2.T) * (1 - Z * Z)
        elif self.activations == "relu":
            dZ = (self.T - Y).dot(W2.T) * (Z > 0) # this is for relu activation

        return X.T.dot(dZ)                   # X is (N x D), and dZ should be (N x M).


    # Note that the bias from last hidden layer to output layer shoudl be K x 1.
    def derivative_b2(self, Y):
        return (self.T - Y).sum(axis=0)

    # Note that the bias from input to hidden layer shoudl be M x 1.
    def derivative_b1(self, Y, W2, Z):
        if self.activations == "sigmoid":
            dZ = (self.T - Y).dot(W2.T) * Z * (1 - Z)
            # dZ = np.outer(self.T-Y, W2) * Z * (1 - Z) # this is for sigmoid activation
        elif self.activations == "tanh":
            dZ = (self.T - Y).dot(W2.T) * (1 - Z * Z)
            # dZ = np.outer(self.T-Y, W2) * (1 - Z * Z) # this is for tanh activation
        elif self.activations == "relu":
            dZ = (self.T - Y).dot(W2.T) * (Z > 0)
            # dZ = np.outer(self.T-Y, W2) * (Z > 0) # this is for relu activation

        # return ((self.T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)
        return dZ.sum(axis = 0)

    def cost(self, Y):
        tot = self.T * np.log(Y)
        return tot.sum()

    def costBinary(T, Y):
    # tot = 0
    # for n in xrange(len(T)):
    #     if T[n] == 1:
    #         tot += np.log(Y[n])
    #     else:
    #         tot += np.log(1 - Y[n])
    # return tot
        return np.sum(self.T*np.log(Y) + (1-self.T)*np.log(1-Y))



# def forward(X, W1, b1, W2, b2):
#     # sigmoid
#     # Z = 1 / (1 + np.exp( -(X.dot(W1) + b1) ))

#     # tanh
#     # Z = np.tanh(X.dot(W1) + b1)

#     # relu
#     Z = X.dot(W1) + b1
#     Z = Z * (Z > 0)

#     activation = Z.dot(W2) + b2
#     Y = 1 / (1 + np.exp(-activation))
#     return Y, Z


# def predict(X, W1, b1, W2, b2):
#     Y, _ = forward(X, W1, b1, W2, b2)
#     return np.round(Y)


def yIndicator(Y):

    T = np.zeros((len(Y), len(set(Y))))
    for i in range(len(Y)):
        T[i, Y[i]] = 1

    return T


def test_xor():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y = np.array([0, 1, 1, 0])
    W1 = np.random.randn(2, 5)
    b1 = np.zeros(5)
    W2 = np.random.randn(5, len(set(Y)))
    b2 = 0

    T = yIndicator(Y)
    LL = [] # keep track of likelihoods
    learning_rate = 1e-2
    regularization = 0.
    last_error_rate = None

    forwareObj = ForwardDL(X)
    backpropObj = BackPropagation(T, "sigmoid")

    for i in xrange(30000):
        
        # pY, Z = forward(X, W1, b1, W2, b2)
        pY, Z = forwareObj.forward("sigmoid", 1, W1, b1, W2, b2)
        ll = backpropObj.cost(pY)
        er = forwareObj.classificationRate(pY, Y)

        if er != last_error_rate:
            last_error_rate = er
            print "error rate:", er
            print "true:", Y
            print "pred:", pY
        # if LL and ll < LL[-1]:
        #     print "early exit"
        #     break
        LL.append(ll)   
        W2 += learning_rate * (backpropObj.derivative_w2(Z, pY) - regularization * W2)
        b2 += learning_rate * (backpropObj.derivative_b2(pY) - regularization * b2)
        W1 += learning_rate * (backpropObj.derivative_w1(X, Z, pY, W2) - regularization * W1)
        b1 += learning_rate * (backpropObj.derivative_b1(pY, W2, Z) - regularization * b1)
        if i % 1000 == 0:
            print ll

    print "final classification rate:", forwareObj.classificationRate(pY, Y) 
    plt.plot(LL)
    plt.show()


def test_donut():
    # donut example
    N = 1000
    R_inner = 5
    R_outer = 10

    # distance from origin is radius + random normal
    # angle theta is uniformly distributed between (0, 2pi)
    R1 = np.random.randn(N/2) + R_inner
    theta = 2*np.pi*np.random.random(N/2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

    R2 = np.random.randn(N/2) + R_outer
    theta = 2*np.pi*np.random.random(N/2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([ X_inner, X_outer ])
    Y = np.array([0]*(N/2) + [1]*(N/2))
    T = yIndicator(Y)

    n_hidden = 8
    W1 = np.random.randn(2, n_hidden)
    b1 = np.random.randn(n_hidden)
    W2 = np.random.randn(n_hidden, len(set(Y)))
    b2 = np.random.randn(len(set(Y)))
    LL = [] # keep track of likelihoods
    learning_rate = 0.00005
    regularization = 0.2
    last_error_rate = None

    forwareObj = ForwardDL(X)
    backpropObj = BackPropagation(T, "sigmoid")

    for i in xrange(100000):
        pY, Z = forwareObj.forward("sigmoid", 1, W1, b1, W2, b2)
        ll = backpropObj.cost(pY)
        # prediction = predict(X, W1, b1, W2, b2)        

        er = forwareObj.classificationRate(pY, Y)
        LL.append(ll)
        W2 += learning_rate * (backpropObj.derivative_w2(Z, pY) - regularization * W2)
        b2 += learning_rate * (backpropObj.derivative_b2(pY) - regularization * b2)
        W1 += learning_rate * (backpropObj.derivative_w1(X, Z, pY, W2) - regularization * W1)
        b1 += learning_rate * (backpropObj.derivative_b1(pY, W2, Z) - regularization * b1)
        if i % 100 == 0:
            print "i:", i, "ll:", ll, "classification rate:", forwareObj.classificationRate(pY, Y)
    plt.plot(LL)
    plt.show()


if __name__ == '__main__':
    # test_xor()
    test_donut()

    


