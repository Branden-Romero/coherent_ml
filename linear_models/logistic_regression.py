import numpy as np
import scipy.sparse as sps

def gradient_descent(X,y,W,penalty,C,step_size=0.001,delta=.01,max_iterations=1000):
	for i in xrange(max_iterations):
		W_old = W
		W = gradient_descent_step(X,y,W_old,penalty,C,step_size)
		if np.linalg.norm(W-W_old) <= delta:
			break
	return W
		
def gradient_descent_step(X,y,W,penalty,C,step_size=0.001):
	y = np.reshape(y,(-1,1))
	W = W + step_size*(np.sum(X*(y-Py1(X,W)),axis=0) - penalty(W,C))
	return W

def Py1(X,W):
	W = np.reshape(W,(-1,1))
	exp = np.exp(np.dot(X,W))
	return exp/(1.0+exp)

def Py0(X,W):
	W = np.reshape(W,(-1,1))
	exp = np.exp(np.dot(X,W))
	return 1.0/(1+exp)

def zero(W,C):
	return 0
	
def l1(W,C):
	return 0

def l2(W,C): 
	return 0

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

	def fit(self, X, y,step_size=0.001,delta=.01,max_iterations=1000):
		M,N = X.shape
		if sps.issparse(X) == True:
			X = X.toarray()
		
		X = np.hstack((X,np.ones((M,1))))

		self.W = np.zeros(N+1)
		self.W = gradient_descent(X,y,self.W,self.penalty,self.C,step_size,delta,max_iterations)

	def predict(self, X,prob=False):
		M,N = X.shape
		X = np.append(X,np.ones((M,1)),axis=1)
		if prob == False:
			prediction = (0 < np.dot(X,self.W)).astype(int)
		else:
			prediction = Py0(X,self.W)
		return prediction
