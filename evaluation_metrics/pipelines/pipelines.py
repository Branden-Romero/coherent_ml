from sklearn.cross_validation import cross_val_score
import numpy as np

class Pipeline():
	def __init__(self,steps):
		self.steps = steps

	def evaluate(self,X,y):
		steps = len(self.steps)
		scores = []
		for step in xrange(steps):
			classifier,cross_validation,cohere = self.steps[step]
			try:
				if self.steps[step-1][0].estimators_[0].get_params() == classifier.estimators_[0].get_params():
					classifier = self.steps[step-1][0]
			except:
				cvScores = cross_val_score(classifier,X,y,cv=cross_validation)
				classifier.fit(X,y)
		
			cScores = cohere.score(classifier,X,y,fit=False)
			scores.append(np.append(cvScores,cScores))
		return (np.array(scores))
