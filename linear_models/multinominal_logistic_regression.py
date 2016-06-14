import numpy as np
import scipy.sparse as sps

def gradient_descent(X,y,W,classes,penalty,C,step_size=0.001,delta=.01,max_iterations=1000):
	for i in xrange(max_iterations):
		W_old = W
		W = gradient_descent_step(X,y,W_old,classes,penalty,C,step_size)
		if np.linalg.norm(W-W_old) <= delta:
			break
	return W
		
def gradient_descent_step(X,y,W,classes,penalty,C,step_size=0.001):
	y = np.reshape(y,(-1,1))
	for cls in classes:
		W[cls] = W[cls] + step_size*(np.sum(X*((y==cls)-np.reshape(Pyk(X,W,cls,classes),(-1,1))),axis=0) - penalty(W[cls],C))
	return W

def Pyk(X,W,cls,classes):
	if cls == classes[-1]:
		num = 1.0
	else:
		Wk = W[cls]
		num = np.exp(np.dot(Wk,X.T))

	W = W[:-1]
	phi = np.array([np.exp(np.dot(Wi,X.T)) for Wi in W])
	dem = 1.0 + np.sum(phi,axis=0)
	return num/dem


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
		self.classes = 0

	def fit(self, X, y,step_size=0.001,delta=.01,max_iterations=1000):
		M,N = X.shape
		if sps.issparse(X) == True:
			X = X.toarray()
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
		probabilities = np.array([probabilities])
		prediction = np.argmax(probabilities,axis=0)
		return prediction
