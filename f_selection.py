# -*- coding: utf-8 -*-
import pickle

from sklearn.feature_selection import SelectKBest, chi2

def select_master_unigram(t_categories_file, vocabulary_file, X_train_file, master_unigram_file):
	t_categories = pickle.load(open(t_categories_file, "r"))
	vocabulary = pickle.load(open(vocabulary_file, "r"))
	X_train = pickle.load(open(X_train_file, "r"))
	print("Selecting master unigrams by a chi-squared test")
	ch2 = SelectKBest(chi2, k=1500)
	X_train = ch2.fit(X_train, t_categories)
	mask = ch2._get_support_mask()
	master_unigram_index = [i for i, e in enumerate(mask) if e==True]
	inv_vocabulary = {v:k for k, v in vocabulary.items()}
	master_unigram = [inv_vocabulary[x] for x in master_unigram_index]
	pickle.dump(master_unigram, open(master_unigram_file, "wb"))

if __name__=="__main__":
	select_master_unigram("../temp/t_categories.p", "../temp/vocabulary.p", "../temp/X_train.p", "../temp/master_unigram.p")
