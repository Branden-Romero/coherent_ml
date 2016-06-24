import tensorflow as tf
import numpy as np
import scipy.sparse as sps
import time

def shuffle(X,y):
	N = y.shape[0]
	inds = np.random.permutation(N)
	X_shuffled = X[inds,:]
	y_shuffled = y[inds,:]
	return (X_shuffled,y_shuffled)

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
	def __init__(self):	
		self.W = None
		self.b = None
		self.y = None
		self.X = None
		self.y_ = None
		self.classes = None

	def fit(self, X_data, y_data,learning_rate=0.01,training_epochs=25,batch_size=100):
		M,N = X_data.shape
		if sps.issparse(X_data) == True:
			X_data = X_data.toarray()
		self.classes = np.array(range(y_data.shape[1]))
		self.X = tf.placeholder(tf.float32, [None,N])
		self.y_ = tf.placeholder(tf.float32, [None,self.classes.shape[0]])
		self.W = tf.Variable(tf.zeros([N,self.classes.shape[0]]))
		self.b = tf.Variable(tf.zeros([self.classes.shape[0]]))

		self.y = tf.nn.softmax(tf.matmul(self.X,self.W)+self.b)
		cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_*tf.log(self.y), reduction_indices=[1]))
		train_step = tf.train.AdadeltaOptimizer().minimize(cross_entropy)

		init = tf.initialize_all_variables()

		with tf.Session() as sess:
			sess.run(init)
			for epoch in xrange(training_epochs):
				for batch in get_batches(X_data,y_data,batch_size):
					train_step.run(feed_dict={self.X: batch[0], self.y_:batch[1]})

	def predict(self, X_data):
		if sps.issparse(X_data) == True:
			X_data = X_data.toarray()

		evaluation = tf.argmax(self.y,1)
		init = tf.initialize_all_variables()

		with tf.Session() as sess:
			sess.run(init)
			prediction = sess.run(evaluation, feed_dict={self.X: X_data})

		return prediction

	def score(self,X_data,y_data):
		if sps.issparse(X_data) == True:
			X_data = X_data.toarray()

		correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(self.y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
		init = tf.initialize_all_variables()

		with tf.Session() as sess:
			sess.run(init)
			score = sess.run(accuracy, feed_dict={self.X: X_data, self.y_: y_data})

		return score
