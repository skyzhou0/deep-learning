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

# first let's just try to tokenize the text using nltk's tokenizer
# let's take the first review for example:
# t = positive_reviews[0]
# nltk.tokenize.word_tokenize(t.text)
#
# notice how it doesn't downcase, so It != it
# not only that, but do we really want to include the word "it" anyway?
# you can imagine it wouldn't be any more common in a positive review than a negative review
# so it might only add noise to our model.
def tokenizeText(inputText, stopwords):
	wordnet_lemmatizer = WordNetLemmatizer()
	# a. lower all text.
	tokenizer = inputText.text.lower()
	# b. nltk word_tokenize perform similar operation like the split() method. 
	tokenizer = nltk.tokenize.word_tokenize(tokenizer)
	# c. omit words length great than 2.
	tokenizer = [ word for word in tokenizer if len(word) > 2 ]
	# d put words into base form
	tokenizer = [wordnet_lemmatizer.lemmatize(t) for t in tokenizer] 
	# e. remove stopwords
	tokenizer = [word for word in tokenizer if word not in stopwords]

	return tokenizer

# create a word-to-index map so that we can create our word-frequency vectors later
# let's also save the tokenized versions so we don't have to tokenize again later

def wordDictionary(inputFile1, inputFile2, stopwords):
	word_index_map = {}
	current_index = 0
	positive_tokenized = []
	negative_tokenized = []

	for review in inputFile1:
		tokenizer = tokenizeText(review, stopwords)
		positive_tokenized.append(tokenizer)
		for token in tokenizer:
			if token not in word_index_map:
				word_index_map[token] = current_index
				current_index += 1

	for review in inputFile2:
		tokenizer = tokenizeText(review, stopwords)
		negative_tokenized.append(tokenizer)
		for token in tokenizer:
			if token not in word_index_map:
				word_index_map[token] = current_index
				current_index += 1
	return word_index_map, current_index, positive_tokenized, negative_tokenized

# note that work_index_map will be a dictionary so we can refer back to it.  current_index is = 11088, this indicates that there are 11088 distinct words in total. 
# in other words, we will have 11088 featurs whilst of total reviews is only 2,000, which means that we have a "fat" input data matrix.
# word_index_map, current_index, positive_tokenized, negative_tokenized = wordDictionary(positive_reviews, negative_reviews)


def wordToVector(review, label, wordDict):

	X = np.zeros(len(wordDict) + 1)

	for word in review:
		i = (wordDict[word])
		X[i] += 1

	X = X/X.sum()
	X[-1] = label
	return X 

# Create data matrix.

def dataMatrix(positive, negative, wordMap):

	positiveLen = len(positive) 
	negativeLen = len(negative) 
	N = positiveLen + negativeLen

	data = np.zeros((N, (len(wordMap) + 1)))

	for i in range(N):

		if i < positiveLen:
			X = wordToVector(review = positive[i], label = 1,  wordDict=wordMap)
			data[i, :] = X 
		else:
			X = wordToVector(review = negative[i-positiveLen], label = 0,  wordDict=wordMap)
			data[i, :] = X 

	# X = data[:, :-1]
	# Y = data[:, -1]

	return data