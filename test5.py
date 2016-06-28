from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

num_labels = mnist.train.labels.shape[1]
num_features = mnist.train.images.shape[1]

x = tf.placeholder(tf.float32, [None,num_features])
W = tf.Variable(tf.zeros([num_features,num_labels]))
b = tf.Variable(tf.zeros([num_labels]))
y = tf.nn.softmax(tf.matmul(x,W)+b)

y_ = tf.placeholder(tf.float32, [None,num_labels])
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

tf.initialize_all_variables().run()
for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	train_step.run({x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accuracy.eval({x: mnist.test.images, y_:mnist.test.labels}))
