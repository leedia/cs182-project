import pandas as pd
import datetime
import numpy as np
from collections import defaultdict 
from math import log
from sklearn import svm



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
    train.to_csv('train4.csv', index=False)
    test.to_csv('test4.csv', index=False)

class SVMClassifier:

	def train(self, trainingset, col):
		# init vars 
		self.dict = {}
		self.classified = [0]*2 # 0 is real, 1 is fake

		# read data
		df = pd.read_csv(trainingset)
		articles = df[col]
		labels = df['label']
		len_data = len(articles)

		# fill dict and classified
		total_wordcount = 0
		label_wordcounts =[0, 0]
		wordcounts =[[], []]
		for i in range(len_data):
			article_wordcount = 0
			article = articles[i]
			label = label_to_num(labels[i])
			for word in article.split():
				print word
				if word not in self.dict:
					print "adding to dictionary"
					self.dict[word] = total_wordcount
					total_wordcount += 1
					wordcounts[0].append(0)
					wordcounts[1].append(0)
				
				label_wordcounts[label] = label_wordcounts[label] + 1.0
				wordcounts[label][self.dict[word]] = wordcounts[label][self.dict[word]] + 1.0

				article_wordcount = article_wordcount + 1.0

			print "article_wordcount:", article_wordcount
			for j in range(len(self.dict)):
				if article_wordcount != 0:
					wordcounts[label][j] = wordcounts[label][j] / article_wordcount
				else:
					wordcounts[label][j] = 0

		for word in self.dict:
			print word, wordcounts[0][self.dict[word]]


		wordcount_diff = [0] * len(wordcounts[0])
		for i in range(len(wordcounts[0])):
			wordcount_diff[i] = abs(wordcounts[1][i]-wordcounts[0][i])

		ranked_words = {}
		for word in self.dict:
			ranked_words[word] = wordcount_diff[self.dict[word]]

		# print ranked_words

		self.sorted_words = sorted(ranked_words.iteritems(), key=lambda (k, v): (v, k), reverse = True)

		self.sorted_words = self.sorted_words

		print self.sorted_words

		self.shortDict = {}
		wordcount = 0
		for k,v in self.sorted_words:
			self.shortDict[k] = wordcount
			wordcount = wordcount + 1


		# print self.shortDict

		# fill counts
		self.datapoints = [[0.0] * len(self.shortDict)] * len_data
		print self.datapoints
		print "len", len(self.datapoints), len(self.datapoints[0])


		self.labelset = [0] * len_data
		for i in range(len_data):
			label = label_to_num(labels[i])
			self.labelset[i] = label
			article = articles[i]

			article_wordcount = 0

			print "---"
			print "i:", i

			for word in article.split():
				
				article_wordcount = article_wordcount + 1.0

				if word in self.shortDict:
					print self.datapoints[i][self.shortDict[word]]
					self.datapoints[i][self.shortDict[word]] += 1.0
					print word, "i:", i, "adding to datapoint count", self.datapoints[i][self.shortDict[word]]

			for word in self.shortDict:
				if article_wordcount != 0:
					print "before:\t", word, "\t", self.datapoints[i][self.shortDict[word]]
					self.datapoints[i][self.shortDict[word]] = self.datapoints[i][self.shortDict[word]] / article_wordcount
					print "after-:\t", word, "\t", self.datapoints[i][self.shortDict[word]]
					print self.datapoints
				else:
					self.datapoints[i][self.shortDict[word]] = 0


			print "datapoints:"
			for word in self.shortDict:
				print word, "\t", self.datapoints[0][self.shortDict[word]], "\t", self.datapoints[1][self.shortDict[word]]




		

	def fit(self):
		self.clf = svm.SVC(gamma='scale', kernel='linear')
		self.clf.fit(self.datapoints, self.labelset)

	def test(self, testset, col):
		# read data
		df = pd.read_csv(testset)
		articles = df[col]
		labels = df['label']
		len_data = len(articles)

		accurate, total = 0.,0.
		pred = []
		for i in range(len_data):
			label = label_to_num(labels[i])
			article = articles[i]
			test_point = [0 for _ in range(len(self.shortDict))]
			article_wordcount = 0
			for word in article.split():
				article_wordcount = article_wordcount + 1.0
				if word in self.shortDict:
					test_point[self.shortDict[word]] += 1.0
			for word in self.shortDict:
				if article_wordcount != 0:
					test_point[self.shortDict[word]] = test_point[self.shortDict[word]] / article_wordcount
				else:
					test_point[self.shortDict[word]] = 0
			p = self.clf.predict([test_point])
			pred.append(p)
			if p == label:
				accurate += 1
			total += 1
		return (pred, accurate/total)


if __name__ == '__main__':
	split('lowerandwords.csv')

	c = SVMClassifier()
	starttime = datetime.datetime.now()
	print "starttime:", starttime
	print "Processing training set..."
	c.train('tiny.csv', 'text')
	print len(c.dict), "words in dictionary"
	print "timestamp:", datetime.datetime.now()
	print "Fitting model..."
	c.fit()
	print "timestamp:", datetime.datetime.now()
	print "Accuracy on validation set:", c.test('tinytext.csv', 'text')[1]
	print "endtime:", datetime.datetime.now()
	duration = datetime.datetime.now() - starttime
	print "duration in seconds:", duration.total_seconds()
	# print "Good alpha:", c.improve_alpha('test.csv', 'text')