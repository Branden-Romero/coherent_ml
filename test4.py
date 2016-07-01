from datasets import twenty_newsgroups
from linear_models import tf_softmax_regression
from sklearn import linear_model
import numpy as np
from sklearn.datasets import make_classification
def hot_one_conv(y):
	num_classes = np.unique(y).shape[0]
	N = y.shape[0]
	y_hot_one = np.zeros((N,num_classes))
	for i in xrange(N):
		y_hot_one[i,y[i]] = 1.0
	return y_hot_one

#data = twenty_newsgroups.TwentyNewsgroups()
X,y = make_classification(n_samples=15000,n_classes=3,n_informative=3,n_features=10000)
X_train = X[:10000]
y_train = y[:10000]
X_test = X[10000:]
y_test = y[10000:]
'''
#clf2 = linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg')
clf2 = linear_model.SGDClassifier(loss='log')
clf2.fit(X_train,y_train,)
scoreclf2 = clf2.score(X_test,y_test)
print(scoreclf2)
'''
y_test = hot_one_conv(y_test)
y_train = hot_one_conv(y_train)
clf = tf_softmax_regression.LogisticRegression()
clf.fit(X_train,y_train,C=10)
scoreclf = clf.score(X_test,y_test)
print(scoreclf)
