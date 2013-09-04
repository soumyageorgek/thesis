# -*- coding: utf-8 -*-
import pickle
from operator import add
from itertools import combinations

import numpy as np
from scipy.sparse import coo_matrix

def get_term_columns(term_set, vocabulary, counted):
	dictionary = {}
	for x in term_set:
		index = vocabulary.get(x, -1)
		dictionary[x] = counted[:,index].toarray()

	return dictionary


class Cooccurrence:

	def find_cooccurring_terms(self):
		def already_checked(not_set, big):
			for small in not_set:
				if set(small).issubset(set(big)):
					return True
			return False

		def in_occuring_term_set(term_set, occurring_terms_set):
			for element in occurring_terms_set:
				if set(term_set)==(set(element)):
					return True
			return False

		from classifier import Preprocessor, CATEGORIES

		co_occurrence_file = "../temp/co_terms.p"
		co_occurrence = pickle.load(open(co_occurrence_file, "rb"))
		vocabulary_list = [list(set(el[0])) for el in co_occurrence]
		vocabulary_list = filter(lambda x: len(x)>1, vocabulary_list)
		vocabulary = reduce(add, vocabulary_list)
		vocabulary = set(vocabulary)
		preprocessor = Preprocessor(CATEGORIES, vocabulary=vocabulary)
		preprocessor.preprocess('train')
		counted = preprocessor.counted
		N_SAMPLES, N_FEATURES = counted.shape
		vocabulary = preprocessor.vectorizer.vocabulary_

		occurring_terms_set = []
		co_occurrence_column_set = None
		refined = None

		for co_occurrence in vocabulary_list:
			refined = [x for x in co_occurrence if x in vocabulary] #and x not in set([u'histori', u'articl'])]
			if not refined:
				continue
			dictionary = get_term_columns(refined, vocabulary, counted)
			not_set = []
			length = len(refined)
			combination = [list(combinations(refined, x)) for x in xrange(2, length+1)]
			for item in combination:
				for term_set in item:
			
					if already_checked(not_set, term_set):
						continue

					feature_columns = [dictionary[x] for x in term_set]
					co_occurrence_column = np.empty((N_SAMPLES, 1), dtype=np.int32)
					flag = 0
					for i in xrange(N_SAMPLES):
						element = reduce(lambda a,b: a and b, [x[i] for x in feature_columns])
						co_occurrence_column[i,0] = element
						if element:
							flag+=1

					if flag>3:
						if in_occuring_term_set(term_set, occurring_terms_set):
							continue
						occurring_terms_set.append(term_set)
						if co_occurrence_column_set is None:
							co_occurrence_column_set = co_occurrence_column
						else:
							co_occurrence_column_set = np.hstack((co_occurrence_column_set, co_occurrence_column))
					else:
						not_set.append(term_set)

		return (coo_matrix(co_occurrence_column_set), occurring_terms_set)


if __name__=="__main__":
	cooccurrence = Cooccurrence()
	co_occurrence_column_set, occurring_terms_set = cooccurrence.find_cooccurring_terms()
	pickle.dump(occurring_terms_set, open('../temp/occurring_terms_set.p', "wb"))
	pickle.dump(co_occurrence_column_set, open('../temp/co_occurrence_column_set.p', "wb"))

