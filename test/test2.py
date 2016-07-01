import numpy as np
from datasets import multi_domain_sentiment
from linear_models import tf_softmax_regression
from sklearn import linear_model

def hot_one_conv(y):
        num_classes = np.unique(y).shape[0]
        N = y.shape[0]
        y_hot_one = np.zeros((N,num_classes))
        for i in xrange(N):
                y_hot_one[i,y[i]] = 1.0
        return y_hot_one

dvd = multi_domain_sentiment.load('dvd')
change = np.where(dvd.y > 2)
dvd.y[change] = dvd.y[change] - 1
y = hot_one_conv((dvd.y-1).astype(np.int))
X_train = dvd.X[:-1000]
X_test = dvd.X[-1000:]
y_train = y[:-1000]
y_test = y[-1000:]
clf = tf_softmax_regression.LogisticRegression()
clf.fit(X_train,y_train)
print(clf.score(X_test,y_test))
