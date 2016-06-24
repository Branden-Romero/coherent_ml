import numpy as np
import scipy.sparse as sps
import time

def shuffle(X,y):
	N = y.shape[0]
	inds = np.random.permutation(N)
	X_shuffled = X[inds,:]
	y_shuffled = y[inds]
	return (X_shuffled,y_shuffled)

def softmax(a,c):
	return np.exp(a[c])/np.sum(np.exp(a),axis=0)

def gradient(X,y,W,cls):
	y = np.reshape(y,(-1,1))
	return -np.sum(X*((y==cls)-np.reshape(softmax(np.dot(W,X.T),cls),(-1,1))),axis=0)
	
	
def GD(X,y,W,classes,penalty,C,step_size=0.001,delta=.01,max_iterations=10):
	for i in xrange(max_iterations):
		for cls in classes:
			c = np.int(cls)
			W[c] = W[c] - step_size * (gradient(X,y,W,c) - C*penalty(W,c))

	return W
		
def SGD(X,y,W,classes,penalty,C,step_size=0.001,delta=.01,max_iterations=10):
	for i in xrange(max_iterations):
		X,y = shuffle(X,y)
		for i in xrange(X.shape[0]):
			for cls in classes:
				c = np.int(cls)
				W[c] = W[c] - step_size * (gradient(X[i],y[i],W,c) - C*penalty(W,c))
	return W
		
def get_batches(X,y,batch_size):
	N = y.shape[0]
	if (N%batch_size) == 0:
		num_batches = np.int(N/batch_size)
	else:
		num_batches = np.int(N/batch_size) + 1
	batches = []
	for i in xrange(num_batches):
		try:
			batches.append((X[i*batch_size:(i+1)*batch_size],y[i*batch_size:(i+1)*batch_size]))
		except:
			batches.append(X[i*batch_size:],y[i*batch_size:])
	return batches

def mini_batch_GD(X,y,W,classes,penalty,C,step_size=0.001,delta=.01,max_iterations=10,batch_size=100):
	for i in xrange(max_iterations):
		X,y = shuffle(X,y)
		for batch in get_batches(X,y,batch_size):
			for cls in classes:
				c = np.int(cls)
				W[c] = W[c] - step_size * (gradient(batch[0],batch[1],W,c) - C*penalty(W,c))
	return W

def momentum(X,y,W,classes,penalty,C,step_size=0.001,delta=.01,max_iterations=10,batch_size=100,gamma=0.5):
	for i in xrange(max_iterations):
		X,y = shuffle(X,y)
		v = np.zeros(W.shape)
		for batch in get_batches(X,y,batch_size):
			for cls in classes:
				c = np.int(cls)
				v[c] = gamma * v[c] + step_size * (gradient(batch[0],batch[1],W,c) - C*penalty(W,c))
				W[c] = W[c] - v[c]
	return W

def NAG(X,y,W,classes,penalty,C,step_size=0.001,delta=.01,max_iterations=10,batch_size=100,gamma=0.5):
	for i in xrange(max_iterations):
		X,y = shuffle(X,y)
		v = np.zeros(W.shape)
		for batch in get_batches(X,y,batch_size):
			for cls in classes:
				c = np.int(cls)
				W_c = W[c]
				W[c] = W[c] - gamma*v[c]
				v[c] = gamma * v[c] + step_size * (gradient(batch[0],batch[1],W,c) - C*penalty(W,c))
				W[c] = W_c
				W[c] = W[c] - v[c]
	return W

def zero(W,c):
	return 0
	
def l1(W,c):
	return 0

def l2(W,c): 
	return np.linalg.norm(W[c])

class LogisticRegression():
	def __init__(self,C=.1,penalty='none'):	
		penalties = {
			'none': zero,
			'l1' : l1,
			'l2' : l2
		}
		self.C = C
		self.penalty = penalties[penalty]
		self.W = None
		self.classes = 0

	def fit(self, X, y,step_size=0.01,delta=.01,max_iterations=10,opt='mini_batch_GD'):
		optimizers = {
				'GD':GD,
				'SGD':SGD,
				'mini_batch_GD':mini_batch_GD,
				'momentum':momentum,
				'NAG':NAG
		}
		optimizer = optimizers[opt]
		M,N = X.shape
		if sps.issparse(X) == True:
			X = sps.hstack((X,np.ones((M,1))))
			X = X.toarray()
		else:
			X = np.hstack((X,np.ones((M,1))))
		
		self.classes = np.unique(y)
		
		self.W = np.zeros((self.classes.shape[0],N+1))
		self.W = optimizer(X,y,self.W,self.classes,self.penalty,self.C,step_size,delta,max_iterations)

	def predict(self, X):
		M,N = X.shape
		if sps.issparse(X) == True:
			X = X.toarray()
		X = np.hstack((X,np.ones((M,1))))
		probabilities = []
		for cls in self.classes:
			c = np.int(cls)
			probabilities.append(softmax(np.dot(self.W,X.T),c))
		probabilities = np.array([probabilities])[0,:,:]
		prediction = np.argmax(probabilities,axis=0)
		return prediction

	def score(self,X,y):
		predictions = self.predict(X)
		score = float(np.where(predictions==y)[0].shape[0])/y.shape[0]
		return score
