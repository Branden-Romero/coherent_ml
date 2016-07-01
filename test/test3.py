from sklearn.datasets import make_classification
from linear_models.logistic_regression import LogisticRegression
import numpy as np

X,y = make_classification(n_samples=2500)
X_train = X[:2000]
y_train = y[:2000]
X_test = X[2000:]
y_test = y[2000:]

clf = LogisticRegression()
clf.fit(X_train,y_train)
pred = clf.predict(X_test)
print(pred.shape)
print(np.where(y_test==pred)[0].shape)
