from datasets import multi_domain_sentiment
from sklearn.multiclass import OneVsRestClassifier
from sklearn import linear_model
from evaluation_metrics.coherence.coherence import Coherence
import numpy as np

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


domains = ['books','dvd','electronics','kitchen']
data = []
for domain in domains:
	data.append(multi_domain_sentiment.load(domain))

N = len(domains)
outData = []
penalties = ['l1','l2']
Cs = [0.01,0.1,1.0,10.0,100.0,1000.0]
clfs = make_classifiers(["LogisticRegression"],penalties,Cs)
print(len(clfs))
cohere = Coherence(type='OC-Auto-NPMI')
for clf in clfs:
	for i in xrange(0,N-1):
		clf.fit(data[i].X,data[i].y)
		cScore = cohere.score(clf,data[i].X,data[i].y,fit=False)
		for j in xrange(i+1,N):
			aScore = clf.score(data[j].X,data[j].y)
			clf.fit(data[j].X,data[j].y)
			cScores = np.append(cScore,cohere.score(clf,data[j].X,data[j].y,fit=False))
			outData.append(np.append(aScore,cScores))


np.savetxt("multi_domain.csv", outData, delimiter=",")
		
			
