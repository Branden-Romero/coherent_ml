import numpy as np

class Coherence():
	def __init__(self,M=20):
		self.M = M

	def score(self,classifier,X,y):
		classifier.fit(X,y)
		topics = classifier.classes_
		coherences = np.zeros(topics.shape[0])
		for topic in topics:
			coherences[topic] = coherence(classifier.estimators_[topic].coef_,X,self.M)
		return coherences


def coherence(coefs, X, M):
	topTokens = (-coefs).argsort()[:M][0]
	cohere = 0
	for m in xrange(1,M):
		for l in xrange(0,m-1):
			vm = np.nonzero(X[:,topTokens[m]])[0]
			vl = np.nonzero(X[:,topTokens[l]])[0]
			cohere += np.log((np.intersect1d(vm,vl).shape[0] + 1.0) / vl.shape[0])
	return cohere	
