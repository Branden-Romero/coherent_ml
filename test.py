from evaluation_metrics.datasets import twenty_newsgroups
from linear_models import multinominal_logistic_regression
from sklearn import linear_model
import numpy as np

data = twenty_newsgroups.TwentyNewsgroups()
X_train = data.X[:2000]
y_train = data.y[:2000]
X_test = data.X[2000:2500]
y_test = data.y[2000:2500]

clf = multinominal_logistic_regression.LogisticRegression()
clf.fit(X_train,y_train)

clf2 = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
clf2.fit(X_train,y_train)
scoreclf2 = clf2.score(X_test,y_test)

pred = clf.predict(X_test)
print(float(np.where(pred==y_test)[0].shape[0])/y_test.shape[0])
print(scoreclf2)
