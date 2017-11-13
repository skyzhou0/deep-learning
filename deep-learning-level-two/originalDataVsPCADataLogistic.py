# In this example, we want to check the accuracy and time (training specifically) between the normalized full data-set with the PCA reduced data-set (where the variance threshold being set at 0.98).
# Note that we will apply these two senarios on the same train data (full vs. reduction) to compare training time and test data to compare accuracy.

# Findings: with Same epoches 500, Learning Rate = 0.00004, Regularisation = 0.01, we can see full data vs. PCA reduction data: 
# 1. full data model vs. PCA Reduction run time:  158.904171944  vs.  42.9709498882
# 2. full data model vs. PCA Reduction Final error rate:, 0.086421712562495034 vs.  0.08396159034997222

import numpy as np 
# import pandas as pd 
import matplotlib.pyplot as plt 
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

    # -- A. Full Data Model.
	start_time = timeit.default_timer()
	N, D = Xtrain.shape
	K = len(set(Ytrain))

	W = np.random.randn(D, K) / np.sqrt(D)
	b = np.zeros(K)
	# code you want to evaluate
	W_full = W 
	b_full = b 
	LL_full = []
	LLtest_full = []
	CRtest_full = []

    # reg = 1
    # learning rate 0.0001 is too high, 0.00005 is also too high
    # 0.00003 / 2000 iterations => 0.363 error, -7630 cost
    # 0.00004 / 1000 iterations => 0.295 error, -7902 cost
    # 0.00004 / 2000 iterations => 0.321 error, -7528 cost

    # reg = 0.1, still around 0.31 error
    # reg = 0.01, still around 0.31 error
	lr = 0.00004
	reg = 0.01

    # train objects.
	fullForwardObj = ForwardPredictLogits(Xtrain)
	fullTrainObj = ModelTrainLogits(Ytrain_ind)

    # test objects.
	fullForwardPredictObj = ForwardPredictLogits(Xtest)
	fullTestObj = ModelTrainLogits(Ytest_ind)


	for i in range(500):
		p_y = fullForwardObj.forward(W_full, b_full)
		# print "p_y:", p_y
		ll_full = fullTrainObj.cost(p_y)
		LL_full.append(ll_full)

		p_y_test = fullForwardPredictObj.forward(W_full, b_full)
		lltest_full = fullTestObj.cost(p_y_test)
		LLtest_full.append(lltest_full)
        
		err_full = fullForwardPredictObj.error_rate(p_y_test, Ytest)
		CRtest_full.append(err_full)

		W_full += lr*(fullTrainObj.gradW(p_y, Xtrain) - reg*W_full)
		b_full += lr*(fullTrainObj.gradb(p_y) - reg*b_full)
		if i % 10 == 0:
			print("Cost at iteration %d: %.6f" % (i, ll_full))
			print("Error rate:", err_full)

	p_y_full = fullForwardPredictObj.forward(W_full, b_full)
	print("Final error rate:", fullForwardPredictObj.error_rate(p_y_full, Ytest))

	elapsedFull = timeit.default_timer() - start_time
	# print "full data model run time: ", elapsedFull

	# iters = range(len(LL_full))
	# plt.plot(iters, LL_full, iters, LLtest_full)
	# plt.show()
	# plt.plot(CRtest_full)
	# plt.show()

    # -- B. PCA Reduction Data Model.
	start_time = timeit.default_timer()
	# PCA train data.
	pca = PCA()
	XtrainPCA = pca.fit_transform(Xtrain)

	pcaReductionInd, accumulativePCAVariance = pcaReductionIndex(X=Xtrain, pca=pca, pcaVarianceThreshold=0.98)
	print "accumulativePCAVariance: ", accumulativePCAVariance

	XtrainPCAReduction = XtrainPCA[:, :pcaReductionInd]
	# save the classifier
	with open('my_dumped_pca_model.pkl', 'wb') as fid:
		cPickle.dump(pca, fid)

	# PCA test data.
	with open('my_dumped_pca_model.pkl', 'rb') as fid:
		pca_loaded = cPickle.load(fid)

	XtestPCA = pca_loaded.transform(Xtest)
	XtestPCAReduction = XtestPCA[:, :pcaReductionInd]
	
	N, D = XtrainPCAReduction.shape
	K = len(set(Ytrain))

	W = np.random.randn(D, K) / np.sqrt(D)
	b = np.zeros(K)
	# code you want to evaluate
	W_pca = W 
	b_pca = b 
	LL_pca = []
	LLtest_pca = []
	CRtest_pca = []

    # reg = 1
    # learning rate 0.0001 is too high, 0.00005 is also too high
    # 0.00003 / 2000 iterations => 0.363 error, -7630 cost
    # 0.00004 / 1000 iterations => 0.295 error, -7902 cost
    # 0.00004 / 2000 iterations => 0.321 error, -7528 cost

    # reg = 0.1, still around 0.31 error
    # reg = 0.01, still around 0.31 error
	lr = 0.00004 #  0.0001
	reg = 0.01

    # train objects.
	fullForwardObj_PCA = ForwardPredictLogits(XtrainPCAReduction)
	fullTrainObj_PCA = ModelTrainLogits(Ytrain_ind)

    # test objects.
	fullForwardPredictObj_PCA = ForwardPredictLogits(XtestPCAReduction)
	fullTestObj_PCA = ModelTrainLogits(Ytest_ind)


	for i in range(500):
		p_y_pca = fullForwardObj_PCA.forward(W_pca, b_pca)
		# print "p_y:", p_y
		ll_pca = fullTrainObj_PCA.cost(p_y_pca)
		LL_pca.append(ll_pca)

		p_y_pca_test = fullForwardPredictObj_PCA.forward(W_pca, b_pca)
		lltest_pca = fullTestObj_PCA.cost(p_y_pca_test)
		LLtest_pca.append(lltest_pca)
        
		err_pca = fullForwardPredictObj_PCA.error_rate(p_y_pca_test, Ytest)
		CRtest_pca.append(err_pca)

		W_pca += lr*(fullTrainObj_PCA.gradW(p_y_pca, XtrainPCAReduction) - reg*W_pca)
		b_pca += lr*(fullTrainObj_PCA.gradb(p_y_pca) - reg*b_pca)
		if i % 10 == 0:
			print("Cost at iteration %d: %.6f" % (i, lltest_pca))
			print("Error rate:", err_pca)

	p_y_pca = fullForwardPredictObj_PCA.forward(W_pca, b_pca)
	print("Final error rate:", fullForwardPredictObj_PCA.error_rate(p_y_pca, Ytest))

	elapsedPCA = timeit.default_timer() - start_time
	print "full data model vs. PCA Reduction run time: ", elapsedFull, " vs. ", elapsedPCA

	print("full data model vs. PCA Reduction Final error rate:", fullForwardPredictObj.error_rate(p_y_full, Ytest), " vs. ", 
		fullForwardPredictObj_PCA.error_rate(p_y_pca, Ytest))
	iters = range(len(LL_full))
	plt.plot(iters, CRtest_full, iters, CRtest_pca)
	plt.show()



if __name__ == '__main__':
	main()





