# Creation date: 26th Dec 2017.

# Result analysis:
# the accuracy of logistic model is: 0.74 run time:  0:00:00.208646
# the accuracy of Naive Bayes model is: 0.785 run time:  0:00:00.164805
# the accuracy of Ada Boost model is: 0.675 run time:  0:01:07.988982  (note that setting for n_estimators = 100. if we sent it equal to total number of words/features, it will take much longer time to run)
# the accuracy of MLP model is: 0.82 run time:  0:00:25.220620

# The benefit of the Logistic model is that we can see the importance of each individual word.

from __future__ import print_function, division

import pandas as pd 
import numpy as np 
import sys
import nltk
from sklearn.utils import shuffle
import _pickle as cPickle
import nltk
from nltk.stem import WordNetLemmatizer

from bs4 import BeautifulSoup

from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from utilsSentiment import tokenizeText, wordDictionary, wordToVector, dataMatrix


# first let's just try to tokenize the text using nltk's tokenizer
# let's take the first review for example:
# t = positive_reviews[0]
# nltk.tokenize.word_tokenize(t.text)
#
# notice how it doesn't downcase, so It != it
# not only that, but do we really want to include the word "it" anyway?
# you can imagine it wouldn't be any more common in a positive review than a negative review
# so it might only add noise to our model.


def main():

	# stopwords dictionary will be used for us to exclude those words appear in the review data.
	# from http://www.lextek.com/manuals/onix/stopwords1.html
	stopwords = set(w.rstrip() for w in open('stopwords.txt'))

	positive_reviews = BeautifulSoup(open('amazon-electronics/positive.review').read())
	positive_reviews = positive_reviews.findAll('review_text')

	negative_reviews = BeautifulSoup(open('amazon-electronics/negative.review').read())
	negative_reviews = negative_reviews.findAll('review_text')

	np.random.shuffle(positive_reviews)
	positive_reviews = positive_reviews[:len(negative_reviews)]

	# note that work_index_map will be a dictionary so we can refer back to it.  current_index is = 11088, this indicates that there are 11088 distinct words in total. 
	# in other words, we will have 11088 featurs whilst of total reviews is only 2,000, which means that we have a "fat" input data matrix.
	word_index_map, current_index, positive_tokenized, negative_tokenized = wordDictionary(positive_reviews, negative_reviews, stopwords)

	# Create data matrix.
	data = dataMatrix(positive_tokenized, negative_tokenized, word_index_map)
	np.random.shuffle(data)
	X = data[:, :-1]
	Y = data[:, -1]

	Xtrain = X[:-200, :]
	Ytrain = Y[:-200]
	Xtest = X[-200:, :]
	Ytest = Y[-200:]

	# Build the model.
	# a. Logistic model.
	t0_logistic = datetime.now()
	modelLogistic = LogisticRegression()

	modelLogistic.fit(Xtrain, Ytrain)

	print("the accuracy of logistic model is:", modelLogistic.score(Xtest, Ytest), "run time: ", datetime.now() - t0_logistic)

	# b. Naive Bayes model.
	t0_NB = datetime.now()
	modelNB = MultinomialNB()

	modelNB.fit(Xtrain, Ytrain)

	print("the accuracy of Naive Bayes model is:", modelNB.score(Xtest, Ytest), "run time: ", datetime.now() - t0_NB)

	# c. Ada Boost model.
	t0_Ada = datetime.now()
	modelAda = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10),
	        					   n_estimators=current_index, #20000,
	        					   learning_rate=0.03
	        					   #random_state=101
	        					  )

	modelAda.fit(Xtrain, Ytrain)

	print("the accuracy of Ada Boost model is:", modelAda.score(Xtest, Ytest), "run time: ", datetime.now() - t0_Ada)	

	# d. MLP model.
	t0_MLP = datetime.now()
	modelMLP = MLPClassifier(hidden_layer_sizes = (20, 20), max_iter = 2000)

	modelMLP.fit(Xtrain, Ytrain)

	print("the accuracy of MLP model is:", modelMLP.score(Xtest, Ytest), "run time: ", datetime.now() - t0_MLP)

	# let's look at the weights for each word
	# try it with different threshold values!
	threshold = 0.5
	for word, index in word_index_map.items():
		weight = modelLogistic.coef_[0][index]
		if weight > threshold or weight < -threshold:
			print(word, weight)


if __name__ == '__main__':
	main()

# The End.






