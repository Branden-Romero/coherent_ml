from sklearn.cross_validation import cross_val_score

class Pipeline():
	def __init__(self,objects):
		self.classifier = objects[0]
		self.cross_validation = objects[1]
		self.coherence = objects[2]

	def evaluate(self,X,y):
		scores = cross_val_score(self.classifier,X,y,cv=self.cross_validation)
		coherence = 100
		return (scores,coherence)
