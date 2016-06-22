from sklearn.datasets import fetch_20newsgroups
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class TwentyNewsgroups():
	def __init__(self):
		data = fetch_20newsgroups(subset='all')
		self.X, self.features = self.bag_x(data)
		self.y = (data.target).astype(np.float32)

	def bag_x(self,data):
		vectorizer = CountVectorizer()
		X = vectorizer.fit_transform(data.data)
		features = vectorizer.get_feature_names()
		return (X.astype(np.float32), features)

