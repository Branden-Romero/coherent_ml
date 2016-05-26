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
			cvScores = cross_val_score(classifier,X,y,cv=cross_validation)
			cScores = cohere.score(classifier,X,y)
			scores.append(np.append(cvScores,cScores))
		return (np.array(scores))
