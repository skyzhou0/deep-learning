# Creation Date: 27th Dec 2017

# This is example of TF-IDF manual implementation.

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)

# Input
counts = [[3, 0, 1], [2, 0, 0], [3, 0, 0], [4, 0, 0], [3, 2, 0], [3, 0, 2]]

# a. using sklearn transformer.

tfidf = transformer.fit_transform(counts)
tfidfAPI = tfidf.toarray()

# b. mannual calculation. counts[4]

# term frequency. 
tf4 = np.array(counts[4])/int(sum(counts[4]))  # coulumn 5.

# inverse document frequency. 

N = len(counts) # no of documents.

# No zero terms across all documents.
dic = {}
rLen = len(counts)
cLen = len(counts[0])

for i in range(cLen):
	count = 0
	for j in range(rLen):
		if counts[j][i] > 0:
			count += 1
	dic[i] = count

idf4 = []

for i in range(cLen):

	idfValue = np.log(N/dic.get(i)) + 1
	idf4.append(idfValue)


# idf4 = np.array([np.log(6/6) + 1, np.log(6/1) + 1, np.log(6/1) +1])

# tf * idf calculation.

tfIdf4 = tf4 * idf4

# an important fianal step is normalisation (Euclidean Norm).

tfIdfNormalisaed = tfIdf4 / np.sqrt(sum(tfIdf4 ** 2))
tfIdfNormalisaed
# array([ 0.47330339,  0.88089948,  0.        ])

# c. validation.
assert(sum(tfidfAPI[4,:] - tfIdfNormalisaed) < 1e-10)

# Summarize: We can clear see that tfIdfNormalisaed is the same as the 5th row of results that produced by the TfidfTransformer class in the sklearn API.

# The End.



