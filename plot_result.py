# -*- coding: utf-8 -*-
import pickle

import numpy as np

from classifier import CATEGORIES

def draw_barchart(score_set):
	score, mscore, title = score_set
	import pylab as pl
	fig = pl.figure()
	fig.canvas.set_window_title(title)
	ax = pl.subplot(111)
	width = 0.8

	length = len(score)
	ordinary = range(0, length*2, 2)
	modified = range(1, length*2, 2)
	pl.ylabel("F1 score")
	ax.bar(ordinary, score, width=width, label="BOW", align="center")
	ax.bar(modified, mscore, width=width, color="red", label="BOW+co-occurrence", align="center")
	ax.set_xticks(np.arange(15) + width/2)
	ax.set_yticks(np.arange(0, 1, 0.1))
	categories = ["atheism", "", "graphics", "", "hockey", "", "electronics", "","","","","","","",""]
	ax.set_xticklabels(categories, rotation=0)
	ax.legend()

	pl.show()

evaluate = pickle.load(open('../temp/evaluate.p', "r"))
mevaluate = pickle.load(open('../temp/mevaluate.p', "r"))

print("precision_score, recall_score, f1_score")
print("=======================================")

for score_set in zip(evaluate, mevaluate, ["precision_score", "recall_score", "f1_score"]):
	score, mscore, _ = score_set
	print score
	print mscore
	print score<=mscore
	for c,x,y in zip(["atheism", "graphics", "hockey", "electronics"], score, mscore):
		print c+" & "+str(x)+" & "+str(y)+"\\\\"
#print numbers with 4 digit precision.
	draw_barchart(score_set)
	


