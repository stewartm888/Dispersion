#Imports
#################
import pandas as pd
import numpy as np
from yellowbrick.text import DispersionPlot
from yellowbrick.datasets import load_hobbies
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.pyplot as plt
from yellowbrick.style.palettes import PALETTES, SEQUENCES, color_palette
import re
import argparse
import matplotlib.axes as axes


#Read csv code into list
#################
df = pd.read_csv("connect4.csv",header=None) 


#Target words
#################
word1 = 'win'
word2 = 'loss'
word3 = 'draw'

class CommandError(Exception):
	pass


def main(args):
	column = df[42].values.tolist()
	if args.length!='':
		column = column[0:int(args.length)]

	#TRAIN & TEST LISTS
	train_list = column[:int(len(column)*.7)]
	test_list = column[int(len(column)*.7+1):]

	w1count_train = train_list.count(word1)
	w2count_train = train_list.count(word2)
	w3count_train = train_list.count(word3)
	w1count_test = test_list.count(word1)
	w2count_test = test_list.count(word2)
	w3count_test = test_list.count(word3)

	w1percent_train = int(100*w1count_train/len(train_list))
	w2percent_train = int(100*w2count_train/len(train_list))
	w3percent_train = int(100*w3count_train/len(train_list))
	w1percent_test = int(100*w1count_test/len(test_list))
	w2percent_test = int(100*w2count_test/len(test_list))
	w3percent_test = int(100*w3count_test/len(test_list))

	# generate some multi-dimensional data & arbitrary labels
	data = np.array([w1count_train,w2count_train,w3count_train,w1count_test,w2count_test,w3count_test])
	percentages = np.array([[w1percent_train,w2percent_train,w3percent_train,w1percent_test,w2percent_test,w3percent_test]])
	y_pos = np.arange(1)	 
	patch_handles = []
	colors ='orange','cyan','tan'
	left = np.zeros(1) # left alignment of data starts at zero



	### PLOT LINES PREP ###
	d = list()
	for i in range(0,len(column)):
		if column[i] == word1:
			d.append(i)

	e = list()
	for i in range(0,len(column)):
		if column[i] == word2:
			e.append(i)

	f = list()
	for i in range(0,len(column)):
		if column[i] == word3:
			f.append(i)

	### PLOT ###
	fig, ((ax1), (ax2)) = plt.subplots(2, 1, sharex=True, sharey=True)
	ax2.vlines(d, 0, 1, edgecolor="orange",label=word1)
	ax2.vlines(e, 0, 1, edgecolor="cyan",label=word2)
	ax2.vlines(f, 0, 1, edgecolor="tan",label=word3)
	#ax1.set_xlim([0, len(column)+len(column)/6]) # set the lower and upper limits of graph
	ax1.set_xlabel('Train/Test Split (70:30)')
	ax2.set_xlabel('Word Offset')
	ax2.set_xticks([0],minor=True) # turn off: ax.set_xticks([])
	ax2.set_yticks([])
	ax2.legend(loc = 'right',frameon=1,framealpha=1,edgecolor='black')
	axes.Axes.axvline(ax1,x=len(column)*.7,linewidth=9)
	for i, d in enumerate(data):
		patch_handles.append(ax1.barh(y_pos, d, 
		  color=colors[i%len(colors)], align='edge', 
		  left=left))
		# accumulate the left-hand offsets
		left += d
	 
	# go through all of the bar segments and annotate
	for j in range(len(patch_handles)):
		for i, patch in enumerate(patch_handles[j].get_children()):
			bl = patch.get_xy()
			x = 0.5*patch.get_width() + bl[0]
			y = 0.5*patch.get_height() + bl[1]
			ax1.text(x,y, "%d%%" % (percentages[i,j]), ha='center',size='large')


	plt.show()	


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="creates a dispersion plot"
	)


	args = {
		("-l", "--length"): {
			"type": str, "default": '',
			"help": "Size of sample to make into dispersion plot; defaults to entire sample",
		},

	}


#parser code
	for pargs, kwargs in args.items():
		if not isinstance(pargs, tuple):
			pargs = (pargs,)
		parser.add_argument(*pargs, **kwargs)

	try:
		main(parser.parse_args())

	except CommandError as e:
		parser.error(str(e))
