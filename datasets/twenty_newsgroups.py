from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

def hot_one_conv(y):
	num_classes = np.unique(y).shape[0]
	N = y.shape[0]
	y_hot_one = np.zeros((N,num_classes))
	for i in xrange(N):
		y_hot_one[i,y[i]] = 1.0
	return y_hot_one

class TwentyNewsgroups():
	def __init__(self,hot_one=False):
		data = fetch_20newsgroups(subset='all')
		self.X, self.features = self.bag_x(data)
		self.y = data.target
		if hot_one==False:
			self.y = (self.y).astype(np.float32)
		else:
			self.y = hot_one_conv(self.y)
		

	def bag_x(self,data):
		vectorizer = CountVectorizer()
		X = vectorizer.fit_transform(data.data)
		features = vectorizer.get_feature_names()
		return (X.astype(np.float32), features)

