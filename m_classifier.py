# -*- coding: utf-8 -*-
import pickle

import numpy as np
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import metrics

from classifier import Classifier, CATEGORIES


class ModifiedClassifier(Classifier):

	def preprocess(self, file_type):
		if file_type=='train':
			counted = pickle.load(open('../temp/counted.p', "r"))
			N_SAMPLES, N_FEATURES_COUNTED = counted.shape
			counted = counted.tocoo()

			occurring_terms_set = pickle.load(open('../temp/occurring_terms_set.p', "r"))

			co_occurrence_column_set = pickle.load(open('../temp/co_occurrence_column_set.p', "r"))
			N_FEATURES_COOCCURRENCE = co_occurrence_column_set.shape[1]

			TOTAL_FEATURES = N_FEATURES_COUNTED+N_FEATURES_COOCCURRENCE

			data = np.concatenate((counted.data, co_occurrence_column_set.data))
			rows = np.concatenate((counted.row, co_occurrence_column_set.row))
			cols = np.concatenate((counted.col, [N_FEATURES_COUNTED+x for x in co_occurrence_column_set.col]))

			X = coo_matrix((data,(rows,cols)),shape=(N_SAMPLES, TOTAL_FEATURES))

			self.tfidf = self.transformer.fit_transform(X)
			self.occurring_terms_set = occurring_terms_set
			self.N_FEATURES_COOCCURRENCE = N_FEATURES_COOCCURRENCE
			self.N_FEATURES_COUNTED = N_FEATURES_COUNTED

		elif file_type=='test':#here we need all variables N_SAMPLES, counted etc. again calculated.
			from co_occurrence import get_term_columns

			vocabulary = pickle.load(open('../temp/vocabulary.p', "r"))
			counted = pickle.load(open('../temp/counted_test.p', "r"))
			N_SAMPLES = counted.shape[0]

			N_FEATURES_COOCCURRENCE = self.N_FEATURES_COOCCURRENCE
			N_FEATURES_COUNTED = self.N_FEATURES_COUNTED
			TOTAL_FEATURES = N_FEATURES_COUNTED+N_FEATURES_COOCCURRENCE

			occurring_terms_set = self.occurring_terms_set

			co_occurrence_column_set = np.empty((N_SAMPLES, N_FEATURES_COOCCURRENCE), dtype=np.int32)

			for f_number,item in enumerate(occurring_terms_set):
				feature_columns = get_term_columns(item, vocabulary, counted).values()
				for s_number in xrange(N_SAMPLES):
					element = reduce(lambda a,b: a and b, [x[s_number] for x in feature_columns])
					co_occurrence_column_set[s_number, f_number] = element

			co_occurrence_column_set = coo_matrix(co_occurrence_column_set)
			counted = counted.tocoo()
			data = np.concatenate((counted.data, co_occurrence_column_set.data))
			rows = np.concatenate((counted.row, co_occurrence_column_set.row))
			cols = np.concatenate((counted.col, [N_FEATURES_COUNTED+x for x in co_occurrence_column_set.col]))

			X_test = coo_matrix((data,(rows,cols)),shape=(N_SAMPLES, TOTAL_FEATURES))
			self.tfidf = self.transformer.fit_transform(X_test)

if __name__=="__main__":
	clf = ModifiedClassifier(CATEGORIES)
	print("TRAINING")
	print("========")
	print("Loading 20 newsgroups dataset for categories:")
	print(CATEGORIES)

	clf.preprocess('train')
	t_categories = clf.learn_or_predict('train')
	print("Centroids of each categories have been calculated using the new approach.")

	print("TESTING")
	print("========")
	print("Loading 20 newsgroups dataset for categories:")
	print(CATEGORIES)

	clf.preprocess('test')
	X_test = clf.tfidf
	print("Reading files and undergoing stemming.")
	print("Tokens extracted.\ntfidf value from test set is calculated.")

	clf.learn_or_predict('test')
	print("Predicted category of each files")


	print("EVALUATION")
	print("==========")
	pickle.dump(clf.evaluate(), open('../temp/mevaluate.p', "wb"))


#	results.append(benchmark(clf.classifier, clf.t_categories, clf.c_categories, metrics.f1_score(clf.c_categories, clf.t_categories)))

#	indices = np.arange(len(results))

#	results = [[x[i] for x in results] for i in range(2)]

#	clf_names, score = results
#	print results

#	import pylab as pl

#	pl.figure(figsize=(12,10))
#	pl.title("Score")
#	pl.barh(indices, score, .2, label="score", color='r')
#	pl.yticks(())
#	pl.legend(loc='best')
#	pl.subplots_adjust(left=.25)
#	pl.subplots_adjust(top=.95)
#	pl.subplots_adjust(bottom=.05)

#	for i, c in zip(indices, clf_names):
#		pl.text(-.3, i, c)

#	pl.show()


