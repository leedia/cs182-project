import pandas as pd
import numpy as np
from collections import defaultdict 
from math import log
import time

def split(text):
    df = pd.read_csv(text, quotechar='|')
    df['split'] = np.random.randn(df.shape[0], 1)
    split = np.random.rand(len(df)) <= 0.8
    train = df[split]
    test = df[~split]
    train.to_csv('outtrain.csv', index=False)
    test.to_csv('outtest.csv', index=False)

def label_to_num(label):
    if label == 'REAL':
        return 0
    else:
        return 1

class NaiveBayesClassifier:

    def train(self, trainingset, col):
        # init vars 
        self.dict = defaultdict(int)
        self.classified = [0]*2 # 0 is real, 1 is fake

        # read data
        df = pd.read_csv(trainingset)
        articles = df[col]
        labels = df['label']
        len_data = len(articles)

        # fill dict and classified
        marker = 0
        for i in range(len_data):
            article = articles[i]
            label = label_to_num(labels[i])
            for word in article.split():
                if word not in self.dict:
                    self.dict[word] = marker
                    marker += 1
            self.classified[label] += 1

        # fill counts
        self.counts = [[0] * len(self.dict) for _ in range(2)]
        for i in range(len_data):
            label = label_to_num(labels[i])
            article = articles[i]
            for word in article.split():
                self.counts[label][self.dict[word]] += 1

        #print(sorted(self.dict.items(), key=lambda(k,v): v))

    def fit(self, alpha=1):
        self.F = [[0] * len(self.dict) for _ in range(2)]
        for label in range(2):
            for word in range(len(self.dict)):
                p = float(alpha + self.counts[label][word])/(sum(self.counts[label]) + alpha * len(self.counts[label]))
                if p <= 0:
                    self.F[label][word] = 0
                else:
                    self.F[label][word] = -log(p)

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
            test_labels = [0 for _ in range(2)]
            for word in article.split():
                if word in self.dict:
                    for j in range(2):
                        test_labels[j] += self.F[j][self.dict[word]]
            p = test_labels.index(min(test_labels))
            pred.append(p)
            if p == label:
                accurate += 1
            total += 1
        return (pred, accurate/total)

    def improve_alpha(self, testset, col):
        alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 0.9, 1]
        best_alpha = alphas[0]
        best_acc = 0
        for a in alphas:
            self.fit(alpha=a)
            acc = self.test(testset,col)[1]
            if acc > best_acc:
                best_alpha = a
                best_acc = acc
        return best_alpha, best_acc

if __name__ == '__main__':
    #split('out6.csv')

    t0 = time.time()
    c = NaiveBayesClassifier()
    print "Processing training set..."
    c.train('outtrain.csv', 'text')
    print len(c.dict), "words in dictionary"
    print "Fitting model..."
    c.fit(alpha=0.1)
    print "Accuracy on validation set:", c.test('outtest.csv', 'title')[1]
    t1 = time.time()
    alphatest = c.improve_alpha('outtest.csv', 'title')
    print "Good alpha:", alphatest[0]
    print "Improves accuracy to:", alphatest[1]
    print "Code Duration:", t1-t0, "seconds"
    