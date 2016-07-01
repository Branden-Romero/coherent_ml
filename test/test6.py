from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from sklearn import linear_model
from linear_models.tf_softmax_regression import LogisticRegression
clf = LogisticRegression()
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
Cs = [0.01,0.1,1.0,10.0,100.0]
for C in Cs:
	clf.fit(mnist.train.images,mnist.train.labels,C=C)
	print(clf.score(mnist.test.images,mnist.test.labels))
'''
clf = linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg')
mnist = input_data.read_data_sets("MNIST_data/",one_hot=False)
clf.fit(mnist.train.images,mnist.train.labels)
print(clf.score(mnist.test.images,mnist.test.labels))
'''
