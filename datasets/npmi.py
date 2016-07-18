import pickle
import numpy as np

def clip(x):
	if x < 1e-10:
		x = 1e-10
	return x

def npmi_dict(X,fileName):
	X = X.toarray()
	m,n = X.shape
	m = np.float(m)
	data = dict()
	for i in xrange(0,n-1):
		for j in xrange(i+1,n):
			data[(i,j)] = npmi(X[:,i],X[:,j],m)
	pickle.dump(data, open(fileName,"wb"))

def npmi(Xi,Xj,m):
	vm = np.nonzero(Xi)[0]
	vl = np.nonzero(Xj)[0]
	vmvl = np.intersect1d(vm,vl)

	Pwjwi = clip(vmvl.shape[0]/m)
	Pwi = clip(vl.shape[0]/m)
	Pwj = clip(vm.shape[0]/m)

	return np.log(Pwjwi/(Pwi*Pwj))/-np.log(Pwjwi)


