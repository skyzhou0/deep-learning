# Appreciation for Lazyprogrammer for providing the util and mlp codes.

# Compare momentum + RMSprop with regular gradient descent

# In here, I have patched LazyProgrammers' code on Momentum by adding in RMSprop to compare the following:
# 1. ANN with Batch SGD,
# 2. ANN with Batch SGD + Momentum + RMSprop,
# 3. ANN with Batch SGD + Nesterov Momentum + RMSprop,
# across two different activation function: sigmoid and relu. Here is what we have:

# a. with sigmoid activation function for hidden layer.
# ANN error rate without momentum and RMSprop:  0.102 ANN error rate with momentum and RMSprop:  0.063 ANN error rate with Nesterov momentum and RMSprop:  0.055

# b. with relu activation function for hidden layer.
# ANN error rate without momentum and RMSprop:  0.06 ANN error rate with momentum and RMSprop:  0.032 ANN error rate with Nesterov momentum and RMSprop:  0.036

# Conclusion: Both  ANN with Batch SGD + Momentum + RMSprop and ANN with Batch SGD + Nesterov Momentum + RMSprop performs much better than without regardless the 
# activation functions. Moreover, we can see that on this data, according to above results that relu works better than simoid as far as activation function is 
# concern. Finally, the difference between ANN with Batch SGD + Momentum + RMSprop and ANN with Batch SGD + Nesterov Momentum + RMSprop aren't significant.


from __future__ import print_function, division

import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

from util import get_normalized_data, error_rate, cost, y2indicator
from mlp import forward, derivative_w2, derivative_w1, derivative_b2, derivative_b1


def main():
    # compare 3 scenarios:
    # 1. batch SGD
    # 2. batch SGD with momentum
    # 3. batch SGD with Nesterov momentum

    max_iter = 20 # make it 30 for sigmoid
    print_period = 10

    X, Y = get_normalized_data()
    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000,]
    Ytrain = Y[:-1000]
    Xtest  = X[-1000:,]
    Ytest  = Y[-1000:]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300
    K = 10
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)

    # 1. batch
    # cost = -16
    LL_batch = []
    CR_batch = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            # print "first batch cost:", cost(pYbatch, Ybatch)

            # updates
            W2 -= lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)
            b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
            W1 -= lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
            b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)

            if j % print_period == 0:
                # calculate just for LL
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                ll = cost(pY, Ytest_ind)
                LL_batch.append(ll)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll))

                err = error_rate(pY, Ytest)
                CR_batch.append(err)
                print("Error rate:", err)

    pY, _ = forward(Xtest, W1, b1, W2, b2)
    batchER = error_rate(pY, Ytest)
    print("Final error rate:", error_rate(pY, Ytest))

    # 2. batch with momentum
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)
    LL_momentum = []
    CR_momentum = []
    mu = 0.9
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0
    cache_W2 = 1
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1
    decay_rate = 0.999
    eps = 1e-10
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

            # updates
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
            cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
            # W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + eps)

            gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
            cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
            # b2 -= lr0 * gb2 / (np.sqrt(cache_b2) + eps)

            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
            cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
            # W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + eps)

            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1
            cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1
            # b1 -= lr0 * gb1 / (np.sqrt(cache_b1) + eps)

            # updates
            
            
            dW2 = mu*dW2 - lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)/(np.sqrt(cache_W2) + eps)
            W2 += dW2
            db2 = mu*db2 - lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)/(np.sqrt(cache_b2) + eps)
            b2 += db2
            dW1 = mu*dW1 - lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)/(np.sqrt(cache_W1) + eps)
            W1 += dW1
            db1 = mu*db1 - lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)/(np.sqrt(cache_b1) + eps)
            b1 += db1

            if j % print_period == 0:
                # calculate just for LL
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                # print "pY:", pY
                ll = cost(pY, Ytest_ind)
                LL_momentum.append(ll)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll))

                err = error_rate(pY, Ytest)
                CR_momentum.append(err)
                print("Error rate:", err)
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    momentumRMSER = error_rate(pY, Ytest)
    print("Final error rate:", error_rate(pY, Ytest))


    # 3. batch with Nesterov momentum
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.zeros(M)
    W2 = np.random.randn(M, K) / np.sqrt(M)
    b2 = np.zeros(K)
    LL_nest = []
    CR_nest = []
    mu = 0.9
    # alternate version uses dW
    dW2 = 0
    db2 = 0
    dW1 = 0
    db1 = 0
    # vW2 = 0
    # vb2 = 0
    # vW1 = 0
    # vb1 = 0
    cache_W2 = 1
    cache_b2 = 1
    cache_W1 = 1
    cache_b1 = 1
    decay_rate = 0.999
    eps = 1e-10
    for i in range(max_iter):
        for j in range(n_batches):
            # because we want g(t) = grad(f(W(t-1) - lr*mu*dW(t-1)))
            # dW(t) = mu*dW(t-1) + g(t)
            # W(t) = W(t-1) - mu*dW(t)
            # W1_tmp = W1 - lr*mu*vW1
            # b1_tmp = b1 - lr*mu*vb1
            # W2_tmp = W2 - lr*mu*vW2
            # b2_tmp = b2 - lr*mu*vb2

            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
            pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)
            # pYbatch, Z = forward(Xbatch, W1_tmp, b1_tmp, W2_tmp, b2_tmp)

            # updates
            gW2 = derivative_w2(Z, Ybatch, pYbatch) + reg*W2
            cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
            # W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + eps)

            gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
            cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
            # b2 -= lr0 * gb2 / (np.sqrt(cache_b2) + eps)

            gW1 = derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
            cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
            # W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + eps)

            gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1
            cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1
            # b1 -= lr0 * gb1 / (np.sqrt(cache_b1) + eps)

            # updates
            dW2 = mu*mu*dW2 - (1 + mu)*lr*(derivative_w2(Z, Ybatch, pYbatch) + reg*W2)/(np.sqrt(cache_W2) + eps)
            W2 += dW2
            db2 = mu*mu*db2 - (1 + mu)*lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)/(np.sqrt(cache_b2) + eps)
            b2 += db2
            dW1 = mu*mu*dW1 - (1 + mu)*lr*(derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)/(np.sqrt(cache_W1) + eps)
            W1 += dW1
            db1 = mu*mu*db1 - (1 + mu)*lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)/(np.sqrt(cache_b1) + eps)
            b1 += db1
            # vW2 = mu*vW2 + derivative_w2(Z, Ybatch, pYbatch) + reg*W2_tmp
            # W2 -= lr*vW2
            # vb2 = mu*vb2 + derivative_b2(Ybatch, pYbatch) + reg*b2_tmp
            # b2 -= lr*vb2
            # vW1 = mu*vW1 + derivative_w1(Xbatch, Z, Ybatch, pYbatch, W2_tmp) + reg*W1_tmp
            # W1 -= lr*vW1
            # vb1 = mu*vb1 + derivative_b1(Z, Ybatch, pYbatch, W2_tmp) + reg*b1_tmp
            # b1 -= lr*vb1

            if j % print_period == 0:
                # calculate just for LL
                pY, _ = forward(Xtest, W1, b1, W2, b2)
                # print "pY:", pY
                ll = cost(pY, Ytest_ind)
                LL_nest.append(ll)
                print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, ll))

                err = error_rate(pY, Ytest)
                CR_nest.append(err)
                print("Error rate:", err)
    pY, _ = forward(Xtest, W1, b1, W2, b2)
    nestrovRMSER = error_rate(pY, Ytest)
    print("Final error rate:", error_rate(pY, Ytest))

    print("ANN error rate without momentum and RMSprop: ", batchER,
          "ANN error rate with momentum and RMSprop: ", momentumRMSER,
          "ANN error rate with Nesterov momentum and RMSprop: ", nestrovRMSER)


    plt.plot(CR_batch, label="batch")
    plt.plot(CR_momentum, label="momentum")
    plt.plot(CR_nest, label="nesterov")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()
