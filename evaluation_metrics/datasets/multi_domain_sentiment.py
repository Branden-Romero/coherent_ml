import re
import numpy as np

with open('processed_stars/books/all_balanced.review') as f:
	lines = f.readlines()

M = len(lines)
y = np.zeros(M)
vocab = dict()

for line in xrange(M):
	for pair in lines[line].split():
		word = re.search('.*:',pair).group()[:-1]
		if word != '#label#':
			if word not in vocab:
				vocab[word] = len(vocab)
		else:
			value = np.float16(re.search(':.*',pair).group()[1:])
			y[line] = value

N = len(vocab)
X = np.zeros((M,N))

for line in xrange(M):
	for pair in lines[line].split():	
		word = re.search('.*:',pair).group()[:-1]
		if word != '#label#':
			wordInd = vocab[word]
			value = np.uint8(re.search(':.*',pair).group()[1:])
			X[line,wordInd] = value

print(X.shape)
