import numpy as np
import scipy.sparse as sps
import time

def gradient_descent(X,y,W,classes,penalty,C,step_size=0.001,delta=.01,max_iterations=10):
	for i in xrange(max_iterations):
		print(i)
		W = gradient_descent_step(X,y,W,classes,penalty,C,step_size)
	return W
		
def gradient_descent_step(X,y,W,classes,penalty,C,step_size=0.001):
	for cls in classes:
		c = np.int(cls)
		g = 0
		for i in range(X.shape[0]):
			x = X[i]
			y_ic = np.int(y[i]==cls)
			mu_ic = Pyk(x,W,c,classes)	
			g += (mu_ic - y_ic)*x
		W[c] = W[c] - step_size*g
	return W
		

def Pyk(x,W,cls,classes):
	x = np.reshape(x,(-1,1))
	a = np.exp(np.dot(W[cls],x))
	b = np.sum(np.exp(np.dot(W,x)))
	return a/b


def zero(W,C):
	return 0
	
def l1(W,C):
	return 0

def l2(W,C): 
	return np.linalg.norm(W)

class LogisticRegression():
	def __init__(self,C=1.0,penalty='none'):	
		penalties = {
			'none': zero,
			'l1' : l1,
			'l2' : l2
		}
		self.C = C
		self.penalty = penalties[penalty]
		self.W = None
		self.classes = 0

	def fit(self, X, y,step_size=0.001,delta=.01,max_iterations=5):
		M,N = X.shape
		if sps.issparse(X) == True:
			X = sps.hstack((X,np.ones((M,1))))
			X = X.toarray()
		else:
			X = np.hstack((X,np.ones((M,1))))
		
		self.classes = np.unique(y)
		
		self.W = np.zeros((self.classes.shape[0],N+1))
		self.W = gradient_descent(X,y,self.W,self.classes,self.penalty,self.C,step_size,delta,max_iterations)

	def predict(self, X):
		M,N = X.shape
		if sps.issparse(X) == True:
			X = X.toarray()
		X = np.hstack((X,np.ones((M,1))))
		probabilities = []
		for cls in self.classes:
			probabilities.append(Pyk(X,self.W,cls,self.classes))
		probabilities = np.array([probabilities])[0,:,:]
		prediction = np.argmax(probabilities,axis=0)
		return prediction

	def score(self,X,y):
		predictions = self.predict(X)
		score = float(np.where(predictions==y)[0].shape[0])/y.shape[0]
		return score
