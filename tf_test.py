from datasets import twenty_newsgroups
from linear_models import tf_softmax_regression
from sklearn import linear_model
import numpy as np

data = twenty_newsgroups.TwentyNewsgroups(hot_one=True)
ind_train = np.random.permutation(data.y.shape[0])[:2000]
ind_test = np.random.permutation(data.y.shape[0])[:500]
X_train = data.X[ind_train,:]
y_train = data.y[ind_train]
X_test = data.X[ind_test,:]
y_test = data.y[ind_test]

clf = tf_softmax_regression.LogisticRegression()
clf.fit(X_train,y_train)
scoreclf = clf.score(X_test,y_test)
prediction = clf.predict(X_test)

#clf2 = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
#clf2.fit(X_train,y_train)
#scoreclf2 = clf2.score(X_test,y_test)

print(scoreclf)
print(prediction)
#print(scoreclf2)
