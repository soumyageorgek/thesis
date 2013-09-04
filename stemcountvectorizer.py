# -*- coding: utf-8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import VectorizerMixin

class StemCountVectorizer(CountVectorizer):
	def __init__(self, vocabulary=None):
		super(StemCountVectorizer, self).__init__(stop_words="english", charset_error='ignore', vocabulary=vocabulary)


	def build_stemmer(self):
		from nltk.stem import SnowballStemmer
		
		english_stemmer = SnowballStemmer('english')
		return lambda tokens: [english_stemmer.stem(token) for token in tokens] 

	def build_analyzer(self):
		if self.analyzer == 'word':
			stop_words = self.get_stop_words()
			stem = self.build_stemmer()
			tokenize = self.build_tokenizer()
			preprocess = self.build_preprocessor()
	
			return lambda doc: self._word_ngrams(stem(tokenize(preprocess(self.decode(doc)))), stop_words)


