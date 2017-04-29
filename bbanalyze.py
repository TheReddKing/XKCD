#LINEAR SVM
import numpy as np
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

# with open('2015above_clean.csv') as f:
#     train.extend([line for line in csv.reader(f)])
with open('2014_clean.csv') as f:
    train.extend([line for line in csv.reader(f)])
# with open('2013_clean.csv') as f:
#     train.extend([line for line in csv.reader(f)])
# with open('2016_03_clean.csv') as f:
#     train.extend([line for line in csv.reader(f)])
# with open('2016_04_clean.csv') as f:
#     train.extend([line for line in csv.reader(f)])


X_train = [x[0] for x in train]
y_train = [x[1] for x in train]

# Multiple passes to make stuff better.
# count_vect = CountVectorizer()
count_vect = CountVectorizer(tokenizer=LemmaTokenizer(),max_features=100000,max_df=.5)
count_vect.fit(X_train)

#Order y_train results alphabetically
y_map = SortedSet(y_train)

#epochs automatically added :P
X_train_counts = count_vect.transform(X_train)
print X_train_counts.shape

# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf = X_train_counts

# print y_train.shape
clf = SGDClassifier(loss='log', penalty='l2',
                                           alpha=3e-4, n_iter=5, random_state=42).fit(X_train_tfidf, y_train)

docs_new = ['ocean',"bobby","drop tables","phone call", "phones","terrible story"] #,'little bobby tables','little','bobby','tables','drop table','drop me','phone call table']
X_new_counts = count_vect.transform(docs_new)
# X_new_tfidf = tfidf_transformer.transform(X_new_counts)
X_new_tfidf = X_new_counts

predictPure = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predictPure):
    print('%r => %s' % (doc, category))

predicted = clf.predict_proba(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    ind = 0
    for val in category:
        if(val > .01):
            print('%r => %s (%i%%)' % (doc, y_map[ind],int(val*100)))
        ind += 1
test = []
with open('2016_01_clean.csv') as f:
    test.extend([line for line in csv.reader(f)])

X_test = [x[0] for x in test]
y_test = [x[1] for x in test]

X_test_counts = count_vect.transform(X_test)
# X_test_tfidf = tfidf_transformer.transform(X_test_counts)
X_test_tfidf = X_test_counts

print "Awaiting results"

total = 0
accurate = 0
predicted = clf.predict_proba(X_test_tfidf)
for answer, prob in zip(y_test, predicted):
    ind = 0
    for val in prob:
        if(val > .01):
            if(int(y_map[ind]) == int(answer)):
                accurate += 1
                break
        ind += 1
    total += 1
print accurate*100.0/total
