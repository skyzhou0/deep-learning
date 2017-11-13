# Credit to LazyProgrammer.

# It is difficult to construct perfect running time comparison across all three methods:
# 1) full gradient descent
# 2) batch gradient descent
# 3) stochastic gradient descent.
# Therefore, the key is to understand how they differ from each other in terms of their data feed into the update. So a better question will be what are their 
# running time to achieve the same level of accuracy as they all should converge as long as the objective function is concave. 

# If the goal is to do on-line training and scoring and the data is massive, the choice of order should be SGD --> BGD --> GD (in my opinion). However, if the 
# usage of training models are for real-time scoring or batch predicting, then the choice should be depended on the training data size and users' preference.

# note that the learning rate of  0.0001 for Full Gradient descent is too large with the data we supplied here that trajectory is bouncing instead of diminishing 
# gradually. I therefore set the learning rate at this case to be  0.00001. Moreover, given the current hyperparameters setting, SGD is yet to converge; BGD has 
# converged, however it appears to have higher running time. Which maybe understandable given it is going through epoches of 50 * 58(updates with batch sample size = 500) as 
# compare to full GD which is 200 epoches (updates with batch sample size = 30,000). 

from __future__ import print_function, division

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.decomposition import PCA 
import cPickle
import timeit

from utils import getNormalizedData, pcaReductionIndex, ForwardPredictLogits, ModelTrainLogits, yIndicator

def main():

    fileDirectory = "../large_files/"
    fileName = "train.csv"
    pcaVarianceThreshold = 0.98
    X, Y = getNormalizedData(fileDirectory, fileName)

    trainLen = int(0.7 * len(Y))

    Xtrain = X[:trainLen]
    Ytrain = Y[:trainLen]

    Xtest = X[trainLen:]
    Ytest = Y[trainLen:]

    Ytrain_ind = yIndicator(Ytrain)
    Ytest_ind = yIndicator(Ytest)

    # PCA train data.
    pca = PCA()
    XtrainPCA = pca.fit_transform(Xtrain)

    pcaReductionInd, accumulativePCAVariance = pcaReductionIndex(X=Xtrain, pca=pca, pcaVarianceThreshold=0.98)
    print("accumulativePCAVariance: ", accumulativePCAVariance, "pcaReductionInd: ", pcaReductionInd)

    XtrainPCAReduction = XtrainPCA[:, :pcaReductionInd]
    # save the classifier
    with open('my_dumped_pca_gd.pkl', 'wb') as fid:
        cPickle.dump(pca, fid)

    # PCA test data.
    with open('my_dumped_pca_gd.pkl', 'rb') as fid:
        pca_loaded = cPickle.load(fid)

    XtestPCA = pca_loaded.transform(Xtest)
    XtestPCAReduction = XtestPCA[:, :pcaReductionInd]
    N, D = XtrainPCAReduction.shape

    # 1. full
    predictGDObj = ForwardPredictLogits(XtrainPCAReduction)
    modelGDObj = ModelTrainLogits(Ytrain_ind)
    predictGDTestObj = ForwardPredictLogits(XtestPCAReduction)
    modelGDTestObj = ModelTrainLogits(Ytest_ind)


    W = np.random.randn(D, len(set(Ytrain))) / np.sqrt(D)
    b = np.zeros(len(set(Ytrain)))
    LL = []
    lr = 0.00001  # 0.0001
    reg = 0.01
    t0 = datetime.now()
    for i in range(200):
        p_y = predictGDObj.forward(W, b)

        W += lr*(modelGDObj.gradW( p_y, XtrainPCAReduction) - reg*W)
        b += lr*(modelGDObj.gradb( p_y) - reg*b)
        

        p_y_test = predictGDTestObj.forward( W, b)
        ll = modelGDTestObj.cost(p_y_test)
        LL.append(ll)
        if i % 10 == 0:
            err = predictGDTestObj.error_rate(p_y_test, Ytest)
            print("Cost at iteration %d: %.6f" % (i, ll))
            print("Error rate:", err)
    p_y = predictGDTestObj.forward( W, b)
    print("Final error rate:", predictGDTestObj.error_rate(p_y, Ytest))
    print("Elapsted time for full GD:", datetime.now() - t0)


    # 2. stochastic

    predictSGDTestObj = ForwardPredictLogits(XtestPCAReduction)
    modelSGDTestObj = ModelTrainLogits(Ytest_ind)

    W = np.random.randn(D, len(set(Ytrain))) / np.sqrt(D)
    b = np.zeros(len(set(Ytrain)))
    LL_stochastic = []
    lr = 0.0001
    reg = 0.01

    t0 = datetime.now()
    for i in range(1): # takes very long since we're computing cost for 41k samples
        tmpX, tmpY = shuffle(XtrainPCAReduction, Ytrain_ind)
        for n in range(min(N, 500)): # shortcut so it won't take so long...
            x = tmpX[n,:].reshape(1,D)
            y = tmpY[n,:].reshape(1,10)
            predictSGDObj = ForwardPredictLogits(x)
            modelSGDObj = ModelTrainLogits(y)
                
            p_y = predictSGDObj.forward( W, b)

            W += lr*(modelSGDObj.gradW( p_y, x) - reg*W)
            b += lr*(modelSGDObj.gradb( p_y) - reg*b)

            p_y_test = predictSGDTestObj.forward( W, b)
            ll = modelSGDTestObj.cost(p_y_test)
            LL_stochastic.append(ll)

            if n % (N//2) == 0:
                err = predictSGDTestObj.error_rate(p_y_test, Ytest)
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = predictSGDTestObj.forward( W, b)
    print("Final error rate:", predictSGDTestObj.error_rate(p_y, Ytest))
    print("Elapsted time for SGD:", datetime.now() - t0)


    # 3. batch
    predictBGDTestObj = ForwardPredictLogits(XtestPCAReduction)
    modelBGDTestObj = ModelTrainLogits(Ytest_ind)

    W = np.random.randn(D, len(set(Ytrain))) / np.sqrt(D)
    b = np.zeros(len(set(Ytrain)))
    LL_batch = []
    lr = 0.0001
    reg = 0.01
    batch_sz = 500
    n_batches = N // batch_sz

    t0 = datetime.now()
    for i in range(50):
        tmpX, tmpY = shuffle(XtrainPCAReduction, Ytrain_ind)
        for j in range(n_batches):
            x = tmpX[j*batch_sz:(j*batch_sz + batch_sz),:]
            y = tmpY[j*batch_sz:(j*batch_sz + batch_sz),:]

            predictBGDObj = ForwardPredictLogits(x)
            modelBGDObj = ModelTrainLogits(y)

            p_y = predictBGDObj.forward( W, b)

            W += lr*(modelBGDObj.gradW( p_y, x) - reg*W)
            b += lr*(modelBGDObj.gradb( p_y) - reg*b)

            p_y_test = predictBGDTestObj.forward(W, b)
            ll = modelBGDTestObj.cost(p_y_test)
            LL_batch.append(ll)
            if j % (n_batches//2) == 0:
                err = predictBGDTestObj.error_rate(p_y_test, Ytest)
                print("Cost at iteration %d: %.6f" % (i, ll))
                print("Error rate:", err)
    p_y = predictBGDTestObj.forward( W, b)
    print("Final error rate:", predictBGDTestObj.error_rate(p_y, Ytest))
    print("Elapsted time for batch GD:", datetime.now() - t0)



    x1 = np.linspace(0, 1, len(LL))
    plt.plot(x1, LL, label="full")
    x2 = np.linspace(0, 1, len(LL_stochastic))
    plt.plot(x2, LL_stochastic, label="stochastic")
    x3 = np.linspace(0, 1, len(LL_batch))
    plt.plot(x3, LL_batch, label="batch")
    plt.legend()
    plt.show()



if __name__ == '__main__':
    main()