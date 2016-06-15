import re
import numpy as np
import pickle
import operator
import scipy.sparse as sps

class MultiDomainSentiment():
	def __init__(self,data):
		self.data = data
		

class SentimentData():
	def __init__(self,X,y,vocab):
		self.X = X
		self.y = y
		self.vocab = vocab

#file = '~/processed_stars/books/all_balanced.review'
#file = '~/processed_stars/dvd/all_balanced.review'
#file = '~/processed_stars/electronics/all_balanced.review'
file = '~/processed_stars/kitchen/all_balanced.review'

with open(file) as f:
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
#X = np.zeros((M,N))
row = []
col = []
data = []
index = 0
for line in xrange(M):
	for pair in lines[line].split():	
		word = re.search('.*:',pair).group()[:-1]
		if word != '#label#':
			wordInd = vocab[word]
			value = np.uint8(re.search(':.*',pair).group()[1:])
			row.append(line)
			col.append(wordInd)
			data.append(value)
			#X[line,wordInd] = value
row = np.array(row)
col = np.array(col)
data = np.array(data)
X = sps.csr_matrix((data, (row,col)),shape=(M,N), dtype=np.uint8)
sorted_vocab = sorted(vocab.items(), key=operator.itemgetter(1))
vocab_list = list(sorted_vocab)

sData = SentimentData(X,y,vocab_list)

#pickle.dump(sData, open("books.pkl", "wb"))
#pickle.dump(sData, open("dvd.pkl", "wb"))
#pickle.dump(data, open("electronics.pkl", "wb"))
pickle.dump(data, open("kitchen.pkl", "wb"))
