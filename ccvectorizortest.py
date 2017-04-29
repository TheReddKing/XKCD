import numpy as np
from nltk import regexp_tokenize
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.externals import joblib
from sortedcontainers import SortedSet
from functools import partial
from sklearn.linear_model import SGDClassifier
import pickle
import pdb

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

import csv

train = []
with open('2016_03_clean.csv') as f:
    train.extend([line for line in csv.reader(f)])

X_train = [x[0] for x in train]
y_train = [x[1] for x in train]

count_vect = CountVectorizer(tokenizer=LemmaTokenizer(),max_features=80000,max_df=.3)
# count_vect.fit(train)
analyze = count_vect.build_analyzer()
count = 0
for text in X_train:
    print analyze(text)
    count += 1
    if count == 10:
        break
