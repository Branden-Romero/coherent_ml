from datasets import multi_domain_sentiment
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from evaluation_metrics.coherence.coherence import Coherence
import numpy as np
import scipy.sparse as sps

def make_classifiers(clfs,penalties,Cs):
        clfDict = {
                "LogisticRegression":linear_model.LogisticRegression,
                "SGDClassifier":linear_model.SGDClassifier,
                "RidgeClassifier":linear_model.RidgeClassifier,
                "PassiveAggressiveClassifier":linear_model.PassiveAggressiveClassifier
        }
        classifiers = []
        for clf in clfs:
                if clf in clfDict:
                        for penalty in penalties:
                                for C in Cs:
                                        classifiers.append(OneVsRestClassifier(clfDict[clf](penalty=penalty,C=C)))
        return classifiers

multi_domain_sentiment.make()
domains = ['books','dvd','electronics','kitchen']
data = []
for domain in domains:
	data.append(multi_domain_sentiment.load(domain))

N = len(domains)
outData = []
penalties = ['l1','l2']
Cs = [0.01,0.1,1.0,10.0,100.0,1000.0]
clfs = make_classifiers(["LogisticRegression"],penalties,Cs)
cohere = Coherence(type='OC-Auto-NPMI')
folds = np.array(range(N))
for clf in clfs:
	for i in folds:
		inds = np.delete(folds,i,0)
		X = data[inds[0]].X
		y = data[inds[0]].y
		for j in range(1,inds.shape[0]):
			X = sps.vstack((X,data[inds[j]].X))
			y = np.concatenate((y,data[inds[j]].y))
		clf.fit(X,y)
		cScore = cohere.score(clf,X,y,fit=False)
		aScore = clf.score(data[i].X,data[i].y)
		clf.fit(data[i].X,data[i].y)
		cScores = np.append(cScore,cohere.score(clf,data[i].X,data[i].y,fit=False))
		outData.append(np.append(aScore,cScores))

np.savetxt("multi_domain_cv_l21.csv", outData, delimiter=",")
		
			
