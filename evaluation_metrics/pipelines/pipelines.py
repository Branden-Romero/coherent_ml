from sklearn.cross_validation import cross_val_score
import numpy as np
#import time

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
				#startCV = time.time()
				cvScores = cross_val_score(classifier,X,y,cv=cross_validation)
				#endCV = time.time()
				#print("Cross Validation Time: {0} sec.".format(endCV-startCV))
				#startF = time.time()
				classifier.fit(X,y)
				#endF = time.time()
				#print("Fit Time: {0} sec.".format(endF-startF))
		
			#startC = time.time()
			cScores = cohere.score(classifier,X,y,fit=False)
			#endC = time.time()
			#print("Coherence Time: {0} sec.".format(endC-startC))
			scores.append(np.append(cvScores,cScores))
		return (np.array(scores))
