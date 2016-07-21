import tensorflow as tf
import numpy as np
import scipy.sparse as sps
import time
from sklearn.utils import shuffle

def get_batches(X,y,batch_size):
	X, y = shuffle(X,y)
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

def l2(W):
	return tf.nn.l2_loss(W)

def l1(W):
	return tf.reduce_sum(tf.abs(W))

def coherence1(W,X):
	m,n = X.shape
	m = np.float(m)
	cohere = tf.Variable(tf.zeros([1]))
	for i in xrange(0,n-1):
		for j in xrange(1,n):
				cohere = tf.add( cohere,(tf.mul( tf.constant(npmi(X[:,i],X[:,j],m) ,dtype=tf.float32),tf.reduce_sum( tf.pow( tf.sub( W[:,i],W[:,j] ),2 ) ) ) ) )
	return -tf.reduce_sum(cohere)

def clip(val):
	if val < 1e-10:
		val = 1e-10
	return val

def npmi(Xi,Xj,m):
	vm = np.nonzero(Xi)[0]
	vl = np.nonzero(Xj)[0]
	vmvl = np.intersect1d(vm,vl)

	Pwjwi = clip(vmvl.shape[0]/m)
	Pwi = clip(vl.shape[0]/m)
	Pwj = clip(vm.shape[0]/m)

	return np.log(Pwjwi/(Pwi*Pwj))/(-np.log(Pwjwi))
	

class LogisticRegression():
	def __init__(self):
		self.session = tf.InteractiveSession()
		self.__y = None
		self.__X = None
		self.__y_ = None
		self.coef_ = None
		self.intercept_ = None

	def fit(self, X_data, y_data,learning_rate=0.01,training_epochs=25,batch_size=100,C=1.0,loss='l2'):
		losses = {
			'l2': l2,
			'l1': l1,
			'coherence1': coherence1
		}
		loss_func = losses[loss]
		M,N = X_data.shape
		K = y_data.shape[1]

		if sps.issparse(X_data) == True:
			X_data = X_data.toarray()
	
		self.__X = tf.placeholder(tf.float32, [None,N])
		self.__y_ = tf.placeholder(tf.float32, [None,K])

		W = tf.Variable(tf.zeros([N,K]))
		b = tf.Variable(tf.zeros([K]))

		self.__y = tf.nn.softmax(tf.add(tf.matmul(self.__X,W),b))
		loss_function = -tf.reduce_sum(tf.mul(self.__y_,tf.log(tf.clip_by_value(self.__y,1e-10,1.0)))) + C * coherence1(W,X_data)
		train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss_function)
		
		tf.initialize_all_variables().run()

		
		for epoch in xrange(training_epochs):
			for batch in get_batches(X_data,y_data,batch_size):
				train_step.run({self.__X: batch[0], self.__y_:batch[1]})
		
		self.coef_ = W.eval()
		self.intercept_ = b.eval()


	def predict(self, X_data):
		if sps.issparse(X_data) == True:
			X_data = X_data.toarray()
		evaluation = tf.argmax(self.__y,1)
		prediction = self.__y.eval({self.__X: X_data})
		return prediction

	def score(self, X_data,y_data):
		if sps.issparse(X_data) == True:
			X_data = X_data.toarray()
		correct_prediction = tf.equal(tf.argmax(self.__y,1), tf.argmax(self.__y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		score = accuracy.eval({self.__X: X_data, self.__y_: y_data})
		return score
