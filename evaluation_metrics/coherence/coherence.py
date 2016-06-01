import numpy as np

class Coherence():
	def __init__(self,type='default',M=20):
		self.M = M
		self.type = type

	def score(self,classifier,X,y):
		classifier.fit(X,y)
		topics = classifier.classes_
		coherences = np.zeros(topics.shape[0])
		for topic in topics:
			coherences[topic] = coherence(classifier.estimators_[topic].coef_,X,self.M,self.type)
		return coherences

def topic_coherence(vmvl,vm,vl):
	return np.log((vmvl+1.0)/vl)

def oc_auto_pmi(Pwjwi,Pwi,Pwj):
	return np.log(Pwjwi/(Pwi*Pwj))

def oc_auto_npmi(Pwjwi,Pwi,Pwj):
	return (np.log(Pwjwi/(Pwi*Pwj)))/(-np.log(Pwjwi))

def oc_auto_lcp(Pwjwi,Pwi,Pwj):
	return np.log(Pwjwi/Pwi)

def coherence(coefs,X,M,type):
	numDocs = X.shape[0]
	coherenceType = {
			'default': topic_coherence,
			'OC-Auto-PMI': oc_auto_pmi,
			'OC-Auto-NPMI': oc_auto_npmi,
			'OC-Auto-LCP': oc_auto_lcp
	}

	coherenceMetric = coherenceType[type]
	topTokens = (-coefs).argsort()[:M][0]
	cohere = 0

	for m in xrange(1,M):
		for l in xrange(0,m-1):
			vm = np.nonzero(X[:,topTokens[m]])[0].shape[0]
			vl = np.nonzero(X[:,topTokens[l]])[0].shape[0]
			vmvl = np.intersect1d(vm,vl).shape[0]
			if type == 'default':
				cohere += coherenceMetric(vmvl,vm,vl)
			else:
				#Added smoothing factor
				Pwjwi = (vmvl+1)/float(numDocs)
				Pwi = (vl+1)/float(numDocs)
				Pwj = (vm+1)/float(numDocs)
				cohere += coherenceMetric(Pwjwi,Pwi,Pwj)

	return cohere	



