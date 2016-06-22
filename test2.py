import numpy as np
from datasets import multi_domain_sentiment
from linear_models import multinomial_logistic_regression

dvd = multi_domain_sentiment.load('dvd')
clf = multinomial_logistic_regression.LogisticRegression()
clf.fit(dvd.X,dvd.y)
