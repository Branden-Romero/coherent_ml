from datasets import twenty_newsgroups
from linear_models import softmax_regression
from sklearn import linear_model
import numpy as np

data = twenty_newsgroups.TwentyNewsgroups()
X_train = data.X[:2000]
y_train = data.y[:2000]
X_test = data.X[2000:2500]
y_test = data.y[2000:2500]

clf = softmax_regression.LogisticRegression()
clf.fit(X_train,y_train,max_iterations=1,step_size=.0001)
scoreclf = clf.score(X_test,y_test)

#clf2 = linear_model.LogisticRegression(multi_class='multinomial',solver='lbfgs')
#clf2.fit(X_train,y_train)
#scoreclf2 = clf2.score(X_test,y_test)

print(scoreclf)
#print(scoreclf2)
