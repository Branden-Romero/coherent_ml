import numpy as np
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier
from sklearn import cross_validation
from coherent_ml.evaluation_metrics.pipelines import pipelines
from coherent_ml.evaluation_metrics.coherence import coherence
from coherent_ml.evaluation_metrics.datasets import twenty_newsgroups

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
	classifers = [logistic_regression(), sgd_classifiers(), ridge_classifiers(), passive_aggressive_classifiers()]
	
def zip3(a1,a2,a3):
	zipped = []
	for i in xrange(len(a1)):
		for j in xrange(len(a2)):
			for k in xrange(len(a3)):
				zipped.append([a1[i],a2[j],a3[k]])
	return zipped

def main():
	data = twenty_newsgroups.TwentyNewsgroups()
	classifers = get_classifers()
	cv = cross_validation.KFold(data.X.shape[0],n_folds=5)
	cohere = coherence.Coherence()
	testSet = zip3(classifers,cv,cohere)
	pipe = pipelines.Pipeline(testSet)
	scores = pipe.evaluate(data.X,data.y)
	numpy.savetxt("../data/scores.csv", scores, delimiter=",")

	

if __name__ == "__main__":
	main()
	
