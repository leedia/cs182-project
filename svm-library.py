from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import datetime

def label_to_num(label):
	if label == 'REAL':
		return 0
	return 1

startime = datetime.datetime.now()
print("startime:", startime)

df = pd.read_csv('train4.csv')
articles = df['text']
labels = df['label']

dftest = pd.read_csv('test4.csv')
articlestest = dftest['text']
labelstest = dftest['label']

twenty_test = []
twenty_train = []

for i in range(len(articlestest)):
	twenty_test.append((articlestest[i], label_to_num(labelstest[i])))

for i in range(len(articles)):
	twenty_train.append((articles[i], label_to_num(labels[i])))



text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])

out = text_clf_svm.fit(articles, labels)

predicted_svm = text_clf_svm.predict(articlestest)
print("accuracy:", np.mean(predicted_svm == labelstest))
endtime = datetime.datetime.now()
duration = endtime - startime
print("seconds:", duration.total_seconds())