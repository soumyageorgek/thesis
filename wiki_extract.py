# -*- coding: utf-8 -*-
import pickle
import urllib2

from bs4 import BeautifulSoup

from stemcountvectorizer import StemCountVectorizer

DATASET_FOLDER = "../PROJECT_DATASET/wiki_20news-bydate/"
WIKI_PAGES = ["Computer_graphics.html", "Computer_graphics2.html", "Atheism.html", "Hockey.html", "Electronics.html"]

class WikiExtractor:
	def __init__(self, f):
		self.soup = BeautifulSoup(f.read())

	def find_all_links(self):
		content = self.soup.find("div", {"id": "content"})
		content = content.extract()
		links = content.find_all('a')
		self.links = [link for link in links if link.getText().find("http")==-1 and link.getText()!='']
		return self.links

	def preprocess_atext(self):
		"""stem_split_atext: anchor text after preprocessing"""
		text = map(lambda x: x.getText(), self.links)
		vectorizer = StemCountVectorizer()
		analyzer = vectorizer.build_analyzer()
		text = [analyzer(x) for x in text]
		text = filter(lambda x: len(x)>1, text)
		self.stem_split_atext = text
	
	def filter_atext(self, master_unigram):
		"""master_unigram: set of selected unigrams from chisquare
		master_unigram_atext: master_unigrams corresponding to an anchor text
		"""
		self.preprocess_atext()
		filtered_text = []
		for el in self.stem_split_atext:
			if u'disambigu' in el:
				el.remove(u'disambigu')
			if u'articl' in el:
				el.remove(u'articl')
			if len(el)>1:#check this with filter and len(co_occurrence)==1
				master_unigram_atext = []
				for elem in el:
					if elem in master_unigram:#check whether elm is an element in master_unigram list
						master_unigram_atext.append(elem)
				if master_unigram_atext!=[]:
					filtered_text.append((el, master_unigram_atext))

		self.filtered_text=filtered_text
		return self.filtered_text

def get_urls(href):
	print href
	import urllib2

	req = urllib2.Request('http://docs.python.org/3/howto/urllib2.html')
	response = urllib2.urlopen(req)
	the_page = response.read()


if __name__=="__main__":
	master_unigram = pickle.load(open("../temp/master_unigram.p", "rb"))
	co_terms = []

	for item in WIKI_PAGES:
		f = open(DATASET_FOLDER+item, 'r')
		wikiextracter = WikiExtractor(f)
		first_links = wikiextracter.find_all_links()
		co_terms.extend((wikiextracter.filter_atext(master_unigram)))

	pickle.dump(co_terms, open('../temp/co_terms.p', "wb"))

