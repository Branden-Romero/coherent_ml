import tensorflow as tf
import numpy as np
import scipy.sparse as sps
import time

def shuffle(X,y):
	N = y.shape[0]
	inds = np.random.permutation(N)
	X = X[inds,:]
	y = y[inds,:]
	return (X,y)

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

class LogisticRegression():
	session = tf.InteractiveSession()
	def __init__(self):
		self.y = None
		self.X = None
		self.y_ = None

	def fit(self, X_data, y_data,learning_rate=0.01,training_epochs=25,batch_size=100,C=1.0):
		M,N = X_data.shape
		K = y_data.shape[1]

		if sps.issparse(X_data) == True:
			X_data = X_data.toarray()

		self.X = tf.placeholder(tf.float32, [None,N])
		self.y_ = tf.placeholder(tf.float32, [None,K])

		W = tf.Variable(tf.zeros([N,K]))
		b = tf.Variable(tf.zeros([K]))

		self.y = tf.nn.softmax(tf.add(tf.matmul(self.X,W,a_is_sparse=True),b))
		loss_function = -tf.reduce_sum(tf.mul(self.y_,tf.log(tf.clip_by_value(self.y,1e-10,1.0)))) + C / 2.0 * tf.nn.l2_loss(W) #C * tf.reduce_sum(tf.abs(W))#
		train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss_function)

		tf.initialize_all_variables().run()
		
		for epoch in xrange(training_epochs):
			for batch in get_batches(X_data,y_data,batch_size):
				train_step.run({self.X: batch[0], self.y_:batch[1]})


	def predict(self, X_data):
		if sps.issparse(X_data) == True:
			X_data = X_data.toarray()
		evaluation = tf.argmax(self.y,1)
		prediction = self.y.eval({self.X: X_data})
		return prediction

	def score(self,X_data,y_data):
		if sps.issparse(X_data) == True:
			X_data = X_data.toarray()
		correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		score = accuracy.eval({self.X: X_data, self.y_: y_data})
		return score
