import re
import numpy as np
import pickle
import operator
import scipy.sparse as sps
import getpass
import os.path

class SentimentData():
	def __init__(self,X,y,vocab):
		self.X = X
		self.y = y
		self.vocab = vocab

def load(domain):
	user = getpass.getuser()
	dir = '/home/{0}/coherent_ml/evaluation_metrics/datasets/{1}'
	domains = {
		'books':'books.pkl',
		'dvd':'dvd.pkl',
		'electronics':'electronics.pkl',
		'kitchen':'kitchen.pkl'
	}
	file = domains[domain]
	data = pickle.load(open(dir.format(user,file),"rb"))
	return SentimentData(data[0],data[1],data[2])

def make():
	user = getpass.getuser()
	domains = ['books','dvd','electronics','kitchen']
	fileIn = '/home/{0}/processed_stars/{1}/all_balanced.review'
	fileOut = '/home/{0}/coherent_ml/evaluation_metrics/datasets/{1}.pkl'
	if os.path.isfile(fileOut.format(user,domains[-1])) == False:
		for domain in domains:
			with open(fileIn.format(user,domain)) as f:
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
			vocab_list = [v[0] for v in sorted_vocab]

			sData = (X,y,vocab_list)

			pickle.dump(sData, open(fileOut.format(user,domain), "wb"))

