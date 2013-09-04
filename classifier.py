# -*- coding: utf-8 -*-
import pickle

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn import metrics

from stemcountvectorizer import StemCountVectorizer

CATEGORIES = ["alt.atheism", "comp.graphics", "rec.sport.hockey", "sci.electronics"]


class Preprocessor(object):
	def __init__(self, categories=None, vocabulary=None):
		"""Setting train set, vectorizer(ie, how to find tokens such as words), categories selected"""
		self.ng_train = fetch_20newsgroups(subset='train', categories=categories,
                                shuffle=True, random_state=42)
		self.vectorizer = StemCountVectorizer(vocabulary=vocabulary)
		self.categories = categories

	def preprocess(self, file_type):
		"""Calculate token counts for train and test set"""

		if(file_type == 'train'):
			filenames = self.ng_train.filenames
			documents = [open(f).read() for f in filenames]
			counted = self.vectorizer.fit_transform(documents)#here we need to store returned object
		elif(file_type == 'test'):
			filenames = self.ng_test.filenames
			documents = [open(f).read() for f in filenames]
			counted = self.vectorizer.transform(documents)

		self.counted = counted


class Classifier(Preprocessor):
	def __init__(self, categories=None):
		"""More than a Preprocessor, a classifier need tfidf and a classifier"""
		super(Classifier, self).__init__(categories)
		self.ng_test = fetch_20newsgroups(subset='test', categories=self.categories,
                               shuffle=True, random_state=42)
		self.transformer = TfidfTransformer()
#		self.classifier =  NearestCentroid()
		self.classifier = GaussianNB()
#		self.classifier = BernoulliNB()
		print self.classifier

	def preprocess(self, file_type):
		"""Calculate tfidf value"""
		super(Classifier, self).preprocess(file_type)
		self.tfidf = self.transformer.fit_transform(self.counted)
	
	def learn_or_predict(self, file_type):
		"""learning of classifier based on its categories OR predicting category based on learned classifier
		tfidf calculated from previous step is used in this function."""
		if(file_type == 'train'):
			ng_set = self.ng_train
			self.t_categories =  [ng_set.target[x] for x in range(len(ng_set.target))]
			self.classifier.fit(self.tfidf.toarray(), self.t_categories)
		elif(file_type == 'test'):
			ng_set = self.ng_test
			self.t_categories =  [ng_set.target[x] for x in range(len(ng_set.target))]
			self.c_categories = self.classifier.predict(self.tfidf.toarray())
		return self.t_categories
		
	def evaluate(self):
		"""Find how effectively our classifier predict category of an un-classified documents"""

		c_categories = self.c_categories
		t_categories = self.t_categories

		print ("f1_score")
		score = metrics.f1_score(c_categories, t_categories)
		print(score)

		precision_score = metrics.precision_score(c_categories, t_categories, average=None)
		recall_score = metrics.recall_score(c_categories, t_categories, average=None)
		f1_score = metrics.f1_score(c_categories, t_categories, average=None)

		print("classification report:")
		classification_report = metrics.classification_report(c_categories, t_categories, target_names=self.categories)
		print(classification_report)

		print("confusion matrix:")
		confusion_matrix = metrics.confusion_matrix(c_categories, t_categories)
		print(confusion_matrix)

		return (precision_score, recall_score, f1_score)

#		import pylab as pl

#		pl.matshow(confusion_matrix)
#		pl.title('Confusion matrix')
#		pl.colorbar()
#		pl.ylabel('True label')
#		pl.xlabel('Predicted label')
#		pl.show()


if __name__=="__main__":
	clf = Classifier(CATEGORIES)
	print("TRAINING")
	print("========")
	print("Loading 20 newsgroups dataset for categories:")
	print(CATEGORIES)


	
	clf.preprocess('train')
	pickle.dump(clf.vectorizer, open('../temp/vectorizer.p', "wb"))
	pickle.dump(clf.vectorizer.vocabulary_, open('../temp/vocabulary.p', "wb"))
	pickle.dump(clf.counted, open('../temp/counted.p', "wb"))
	X_train = clf.tfidf
	pickle.dump(X_train, open('../temp/X_train.p', "wb"))
	print("Reading files and undergoing stemming.")
	print("Tokens extracted.\ntfidf value from train set is calculated.")

	t_categories = clf.learn_or_predict('train')
	pickle.dump(t_categories, open('../temp/t_categories.p', "wb"))
	print("Centroids of each categories have been calculated.")


	print("TESTING")
	print("========")
	print("Loading 20 newsgroups dataset for categories:")
	print(CATEGORIES)

	
	clf.preprocess('test')
	pickle.dump(clf.counted, open('../temp/counted_test.p', "wb"))
	X_test = clf.tfidf
	print("Reading files and undergoing stemming.")
	print("Tokens extracted.\ntfidf value from test set is calculated.")

	clf.learn_or_predict('test')
	print("Predicted category of each files")


	print("EVALUATION")
	print("==========")
	pickle.dump(clf.evaluate(), open('../temp/evaluate.p', "wb"))

