## SVM Classification
## Test/Train code

import pandas as pd
import datetime
import numpy as np
from collections import defaultdict 
from math import log
from sklearn import svm
import sys



def label_to_num(label):
	if label == 'REAL':
		return 0
	return 1

def split(text):
    df = pd.read_csv(text, quotechar='|')
    df['split'] = np.random.randn(df.shape[0], 1)
    split = np.random.rand(len(df)) <= 0.8
    train = df[split]
    test = df[~split]
    train.to_csv('train5.csv', index=False)
    test.to_csv('test5.csv', index=False)

class SVMClassifier:

	def __init__(self):
		# if there is a max dictionary size specified in the arguments, set the size
		if len(sys.argv) == 2:
			self.shortDictLen = int(sys.argv[1])
		else:
			self.shortDictLen = -1

	def train(self, trainingset, col):
		# init vars 
		self.dict = {}
		self.classified = [0]*2 # 0 is real, 1 is fake

		# read data
		df = pd.read_csv(trainingset)
		articles = df[col]
		labels = df['label']
		len_data = len(articles)

		# fill initial dictionary
		total_wordcount = 0
		wordcounts =[[], []]
		for i in range(len_data):
			article = articles[i]
			label = label_to_num(labels[i])
			
			#split the string of words of the article in to a list of words
			wordlist = article.split()

			for word in wordlist:
				if word not in self.dict:
					# adding new word to dictionary
					self.dict[word] = total_wordcount
					total_wordcount += 1
					wordcounts[0].append(0)
					wordcounts[1].append(0)
				
				# add to the wordcount 1/(num of words in article) to add a frequency based value
				wordcounts[label][self.dict[word]] += 1.0/len(wordlist)

		# only save most different words b/w labels in dictionary
		if self.shortDictLen != -1:
			# initialize difference list
			wordcount_diff = [0] * len(wordcounts[0])

			# calculate the difference in frequencies between each word
			for i in range(len(wordcounts[0])):
				wordcount_diff[i] = abs(wordcounts[1][i]-wordcounts[0][i])

			# create temp dictionary to sort them
			ranked_words = {}
			for word in self.dict:
				ranked_words[word] = wordcount_diff[self.dict[word]]

			# sort the temp dictionary, those with the greatest difference appear first
			self.sorted_words = sorted(ranked_words.iteritems(), key=lambda (k, v): (v, k), reverse = True)
			
			# get the top XX number of words based on the maxdiciontarysize
			self.sorted_words = self.sorted_words[:self.shortDictLen]

			# repopulate 'short' dictionary used by the rest of the code with moost 'different' words
			self.shortDict = {}
			wordcount = 0
			for k,v in self.sorted_words:
				self.shortDict[k] = wordcount
				wordcount = wordcount + 1

		# save all words
		else:
			self.shortDict = self.dict

		# fill counts
		self.datapoints = [[0.0] * len(self.shortDict) for i in range(len_data)]
		self.labelset = [0] * len_data

		# for article in article list
		for i in range(len_data):
			label = label_to_num(labels[i])
			self.labelset[i] = label
			article = articles[i]

			# split the string of the article into a list of words
			wordlist = article.split()

			for word in wordlist:
				# only count words that are in the dictionary
				if word in self.shortDict:
					# add the frequency to the datapoint
					self.datapoints[i][self.shortDict[word]] += (1.0 / len(wordlist))
		

	def fit(self):
		self.clf = svm.SVC(gamma='scale', kernel='linear')
		self.clf.fit(self.datapoints, self.labelset)

	def test(self, testset, col):
		# read data
		df = pd.read_csv(testset)
		articles = df[col]
		labels = df['label']
		len_data = len(articles)

		# initialize variables
		accurate, total = 0.,0.
		pred = []

		# for each article in article list
		for i in range(len_data):
			label = label_to_num(labels[i])
			article = articles[i]
			test_point = [0 for _ in range(len(self.shortDict))]

			# split article string in to list of words
			wordlist = article.split()

			for word in wordlist:
				# if word is in dictionary, add frequency to testpoint total for that word
				if word in self.shortDict:
					test_point[self.shortDict[word]] += (1.0/len(wordlist))

			# get prediction from SVM model
			p = self.clf.predict([test_point])
			# add to prediction list
			pred.append(p)
			# check if accurate
			if p == label:
				accurate += 1
			total += 1
		return (pred, accurate/total)


if __name__ == '__main__':

	if len(sys.argv) != 2 and len(sys.argv) != 1:
		print "Usage: svm-new.py [Max dictionary size]"
		sys.exit(0)

	c = SVMClassifier()
	starttime = datetime.datetime.now()
	print "starttime:", starttime
	print "Processing training set..."
	c.train('train.csv', 'text')
	print len(c.dict), "words in dictionary"
	print "but using", len(c.shortDict), "words"
	print "timestamp:", datetime.datetime.now()
	print "Fitting model..."
	c.fit()
	print "timestamp:", datetime.datetime.now()
	print "Accuracy on validation set:", c.test('test.csv', 'text')[1]
	print "endtime:", datetime.datetime.now()
	duration = datetime.datetime.now() - starttime
	print "duration in seconds:", duration.total_seconds()
	# print "Good alpha:", c.improve_alpha('test.csv', 'text')