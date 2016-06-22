from datasets import twenty_newsgroups
from linear_models import softmax_regression
from sklearn import linear_model
import numpy as np
from sklearn.datasets import make_classification

#data = twenty_newsgroups.TwentyNewsgroups()
X,y = make_classification(n_samples=2500,n_classes=3,n_informative=3,n_features=10000)
X_train = X[:2000]
y_train = y[:2000]
X_test = X[2000:]
y_test = y[2000:]

clf = softmax_regression.LogisticRegression()
clf.fit(X_train,y_train,max_iterations=5)

clf2 = linear_model.LogisticRegression(multi_class='multinomial',solver='newton-cg')
clf2.fit(X_train,y_train)
scoreclf2 = clf2.score(X_test,y_test)

scoreclf = clf.score(X_test,y_test)
print(scoreclf)
print(scoreclf2)
