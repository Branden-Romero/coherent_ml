import numpy as np
import scipy.misc
import math

class Coherence():
	def __init__(self,M=20,type='default',dist='cosine'):
		self.M = M
		self.type = type
		self.dist = dist

	def score(self,classifier,X,y):
		classifier.fit(X,y)
		topics = classifier.classes_
		coherences = np.zeros(topics.shape[0])
		for topic in topics:
			coherences[topic] = coherence(classifier.estimators_[topic].coef_,X,self.M,type=self.type,dist=self.dist)
		return coherences

def topic_coherence(vmvl,vm,vl):
	return np.log((vmvl+1.0)/vl)

def oc_auto_pmi(Pwjwi,Pwi,Pwj):
	return np.log(Pwjwi/(Pwi*Pwj))

def oc_auto_npmi(Pwjwi,Pwi,Pwj):
	return (np.log(Pwjwi/(Pwi*Pwj)))/(-np.log(Pwjwi))

def oc_auto_lcp(Pwjwi,Pwi):
	return np.log(Pwjwi/Pwi)

def sim_cos(wi,wj):
	return (np.dot(wi,wj)+1)/(np.linalg.norm(wi)*np.linalg.norm(wj)+1)

def sim_dice(wi,wj):
	N = wi.shape[0]
	return (2*np.sum([min(wi[n],wj[n]) for n in xrange(N)])+1)/(np.sum(wi+wj)+1)
		
def sim_jaccard(wi,wj):
	N = wi.shape[0]
	return (np.sum([min(wi[n],wj[n]) for n in xrange(N)])+1)/(np.sum([max(wi[n],wj[n]) for n in xrange(N)])+1)

def oc_auto_ds(wi,wj,dist=sim_cos):
	return dist(wi,wj)
	
	

def coherence(coefs,X,M,type='default',dist='cosine'):
	numDocs = X.shape[0]
	coherenceType = {
			'default': topic_coherence,
			'OC-Auto-PMI': oc_auto_pmi,
			'OC-Auto-NPMI': oc_auto_npmi,
			'OC-Auto-LCP': oc_auto_lcp,
			'OC-Auto-DS' : oc_auto_ds
	}

	if type == 'OC-Auto-DS':
		distributionType = {
					'cosine': sim_cos,
					'dice': sim_dice,
					'jaccard': sim_jaccard
		}	

	coherenceMetric = coherenceType[type]
	topTokens = (-coefs).argsort()[:M][0]
	cohere = 0	

	for m in xrange(1,M):
		for l in xrange(0,m-1):
			if type == 'default':
				vm = np.nonzero(X[:,topTokens[m]])[0].shape[0]
				vl = np.nonzero(X[:,topTokens[l]])[0].shape[0]
				vmvl = np.intersect1d(vm,vl).shape[0]
				cohere += coherenceMetric(vmvl,vm,vl)

			elif type == 'OC-Auto-DS':
				wi = (X[:,topTokens[m]] > 0).astype(np.int).toarray().T[0]
				wj = (X[:,topTokens[l]] > 0).astype(np.int).toarray().T[0]
				distribution = distributionType[dist]
				cohere += coherenceMetric(wi,wj,distribution)

			elif type == 'OC-Auto-LCP':
				vm = np.nonzero(X[:,topTokens[m]])[0].shape[0]
				vl = np.nonzero(X[:,topTokens[l]])[0].shape[0]
				vmvl = np.intersect1d(vm,vl).shape[0]
				#Added smoothing factor
				Pwjwi = (vmvl+1)/float(numDocs)
				Pwi = (vl+1)/float(numDocs)
				cohere += coherenceMetric(Pwjwi,Pwi)
				
			else:
				vm = np.nonzero(X[:,topTokens[m]])[0].shape[0]
				vl = np.nonzero(X[:,topTokens[l]])[0].shape[0]
				vmvl = np.intersect1d(vm,vl).shape[0]
				#Added smoothing factor
				Pwjwi = (vmvl+1)/float(numDocs)
				Pwi = (vl+1)/float(numDocs)
				Pwj = (vm+1)/float(numDocs)
				cohere += coherenceMetric(Pwjwi,Pwi,Pwj)

	if type == 'OC-Auto-DS':
		cohere = cohere/scipy.misc.comb(M,2)
		
	return cohere	



