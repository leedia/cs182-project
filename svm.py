import pandas as pd
import numpy as np
from collections import defaultdict 
from math import log    
from sklearn import svm
from collections import Counter


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
    train.to_csv('train1.csv', index=False)
    test.to_csv('test1.csv', index=False)

class SVMClassifier:

    def train(self, trainingset, col):
        # init vars 
        self.dict = {}
        self.classified = [0]*2 # 0 is real, 1 is fake

        # read data
        df = pd.read_csv(trainingset)
        atemp = df[col]
        labels = df['label']
        len_data = len(atemp)

        articles = []
        for a in atemp:
            articles.append(a)

        # parse common
        temp = defaultdict(int)
        arts = []
        labs = []
        for article in articles:
            for word in article.split():
                temp[word] += 1
        tempc = Counter(temp)
        for k,v in tempc.most_common(100):
            print k
            print v
            arts.append(k)
            labs.append(labels[articles.index(k)])# parse common
        temp = defaultdict(int)
        arts = []
        labs = []
        for article in articles:
            for word in article.split():
                temp[word] += 1
        tempc = Counter(temp)
        for k,v in tempc.most_common(100):
            arts.append(k)
            labs.append(labels[articles.index(k)])

        # fill dict and classified
        wordcount = 0
        for i in range(len_data):
            article = arts[i]
            label = label_to_num(labs[i])
            for word in article.split():
                if word not in self.dict:
                    self.dict[word] = wordcount
                    wordcount += 1
            self.classified[label] += 1

        # fill counts
        self.datapoints = [[0] * len(self.dict)] * len_data
        print "len", len(self.datapoints), len(self.datapoints[0])
        self.labset = [0] * len_data
        article_count = 0;
        for i in range(len_data):
            label = label_to_num(labs[i])
            self.labset[i] = label
            article = arts[i]

            for word in article.split():

                if word in self.dict:
                    self.datapoints[i][self.dict[word]] += 1

        

    def fit(self):
        self.clf = svm.SVC(gamma='scale')
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
            test_point = [0 for _ in range(len(self.dict))]
            for word in article.split():
                if word in self.dict:
                    test_point[self.dict[word]] += 1
            p = self.clf.predict(test_point)
            pred.append(p)
            if p == label:
                accurate += 1
            total += 1
        return (pred, accurate/total)


if __name__ == '__main__':
    split('lowerandwords.csv')

    c = SVMClassifier()
    print "Processing training set..."
    c.train('train1.csv', 'text')
    print len(c.dict), "words in dictionary"
    print "Fitting model..."
    c.fit()
    print "Accuracy on validation set:", c.test('test1.csv', 'text')[1]
    # print "Good alpha:", c.improve_alpha('test.csv', 'text')