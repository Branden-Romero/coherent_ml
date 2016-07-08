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
	return tf.nn.l2_loss(W)/2.0

def l1(W):
	return tf.reduce_sum(tf.abs(W))

def coherence1(W,X,n):
	cohere = tf.Variable(tf.zeros([n]))
	npmi_val, npmi_pos = npmi(X,80000)
	for i in xrange(npmi_val.shape[0]):
			W[npmi_pos[i,1],:]
			cohere = tf.add( cohere,(tf.mul( tf.constant(npmi_val[i],dtype=tf.float32),tf.pow( tf.sub( W[npmi_pos[i,0],:],W[npmi_pos[i,1],:] ),2 ) ) ) )
	return -tf.reduce_sum(cohere)

def clip(val,low,high):
	if val < low:
		val = low
	if val > high:
		val = high
	return val

def npmi(X,n):
	m = X.shape[1]
	top_n = np.zeros(n) - 2
	top_n_pos = np.zeros((n,2))
	for i in xrange(m):
		for j in xrange(i+1,m):
			c_word_i = np.sum((X[:,i]>0).astype(np.int))
			p_word_i = clip(c_word_i/np.float(m),1e-10,1)

			c_word_j = np.sum((X[:,j]>0).astype(np.int))
			p_word_j = clip(c_word_j/np.float(m),1e-10,1)

			c_word_ij = np.sum(((X[:,i]+X[:,j])==1).astype(np.int))
			p_word_ij = clip(c_word_ij/np.float(m),1e-10,1)
			
			npmi_ij = np.log(p_word_ij/(p_word_i*p_word_j))/-np.log(p_word_ij)
			
			if npmi_ij > np.min(top_n):
				min_pos = np.argmin(top_n)
				top_n[min_pos] = npmi_ij
				top_n_pos[min_pos] = np.array([i,j])

	return (top_n,top_n_pos)

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
			'l1': l1
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
		cohere = coherence1(W,X_data,K)

		self.__y = tf.nn.softmax(tf.add(tf.matmul(self.__X,W),b))
		loss_function = -tf.reduce_sum(tf.mul(self.__y_,tf.log(tf.clip_by_value(self.__y,1e-10,1.0)))) + C * cohere
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
