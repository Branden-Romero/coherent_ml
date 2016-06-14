import numpy as np

class OneVsRest():
	def __init__(self,clf):
		self.clf = clf

	def fit(self,X,y):
		numClf = np.unique(y).shape[0]
		self.clf = [self.clf] * numClf
		for i in xrange(numClf):
			y_temp = (y != i).astype(int)
			print(y_temp)
			self.clf[i].fit(X,y_temp)
			
	def predict(self,X):
		M,N = X.shape
		numClf = len(self.clf)
		clfPred = np.zeros(numClf)
		prediction = np.zeros(M)
		for m in xrange(M):
			for i in xrange(numClf):
				clfPred[i] = self.clf[i].predict(X,prob=True)
			prediction[m] = np.argmax(clfPred)
		return prediction
