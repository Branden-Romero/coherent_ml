import numpy as np
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from evaluation_metrics.pipelines import pipelines
from evaluation_metrics.coherence import coherence
from evaluation_metrics.datasets import twenty_newsgroups

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

def logistic_regressions():
	logreg = OneVsRestClassifier(linear_model.LogisticRegression())
	return logreg

def sgd_classifiers():
	sgd = OneVsRestClassifier(linear_model.SGDClassifier())
	return sgd	

def ridge_classifiers():
	ridge = OneVsRestClassifier(linear_model.RidgeClassifier())
	return ridge

def passive_aggressive_classifiers():
	pa = OneVsRestClassifier(linear_model.PassiveAggressiveClassifier())
	return pa

def get_classifers():
	clfTypes = [logistic_regressions(), sgd_classifiers(), ridge_classifiers(), passive_aggressive_classifiers()]
	classifiers = []
	for clfType in clfTypes:
		if not isinstance(clfType,list):
			clfType = [clfType]
		classifiers += clfType
	return classifers
	
def zip3(a1,a2,a3):
	zipped = []
	for i in xrange(len(a1)):
		for j in xrange(len(a2)):
			for k in xrange(len(a3)):
				zipped.append([a1[i],a2[j],a3[k]])
	return zipped

def main():
	data = twenty_newsgroups.TwentyNewsgroups()
	penalties = ['l1']#, 'l2']
	Cs = [0.1]#,1.0,10.0,100.0]
	clfs = ["LogisticRegression"]#,"SGDClassifier","RidgeClassifier","PassiveAggressiveClassifier"]
	classifers = make_classifiers(clfs,penalties,Cs)

	cv = cross_validation.KFold(data.X.shape[0],n_folds=5)
	cohere = [
			coherence.Coherence(type='OC-Auto-DS',dist='cosine'),
			coherence.Coherence(type='OC-Auto-DS',dist='dice'),
			coherence.Coherence(type='OC-Auto-DS',dist='jaccard')
		]

	testSet = zip3(classifers,[cv],cohere)
	pipe = pipelines.Pipeline(testSet)
	scores = pipe.evaluate(data.X[:250],data.y[:250])

	np.savetxt("ds_test.csv", scores, delimiter=",")

	

if __name__ == "__main__":
	main()
	
