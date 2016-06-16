from datasets import multi_domain_sentiment
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from evaluation_metrics.coherence.coherence import Coherence
import numpy as np
import scipy.sparse as sps

multi_domain_sentiment.make()
domains = ['books','dvd','electronics','kitchen']
data = []
for domain in domains:
	data.append(multi_domain_sentiment.load(domain))

N = len(domains)
outData = []
clf = OneVsRestClassifier(LogisticRegression(penalty='l2'))
cohere = Coherence(type='OC-Auto-NPMI')
folds = np.array(range(N))
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
	outData.append(np.append(aScore,cScore))

np.savetxt("multi_domain_cv_l21.csv", outData, delimiter=",")
		
			
