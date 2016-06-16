from datasets import multi_domain_sentiment
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from evaluation_metrics.coherence.coherence import Coherence
import numpy as np

multi_domain_sentiment.make()
domains = ['books','dvd','electronics','kitchen']
data = []
for domain in domains:
	data.append(multi_domain_sentiment.load(domain))

N = len(domains)
outData = []
clf = OneVsRestClassifier(LogisticRegression(penalty='l1'))
cohere = Coherence(type='OC-Auto-NPMI')
for i in xrange(0,N-1):
	clf.fit(data[i].X,data[i].y)
	cScore = cohere.score(clf,data[i].X,data[i].y,fit=False)
	for j in xrange(i+1,N):
		aScore = clf.score(data[j].X,data[j].y)
		outData.append(np.append(aScore,cScore))

np.savetxt("multi_domain2.csv", outData, delimiter=",")
		
			
