'''
Author: your name
Date: 2021-03-10 15:36:55
LastEditTime: 2021-05-18 14:30:04
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /twitterSententAnakyse/use_tfidf.py
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
import math
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
import joblib
from gensim.models import word2vec
import gensim.downloader
import pyprind

LABELS = ['positive','neutral','negative']
def getData(filename):
    X = []
    y = []
    with open(filename) as fr:
        for line in fr:
            line_split = line.strip().split('\t')
            if len(line_split) == 2:
                X.append(line_split[0].strip())
                y.append(LABELS.index(line_split[1].strip()))

    return X, y

def sentence2embedding(model, sentence):
    sum_embedding = np.zeros(100)
    for word in sentence:
        word_embedding = model[word]
        sum_embedding += word_embedding
    return sum_embedding / len(sentence)

    

X, y = getData('./filter_txt_train_data.txt')
#  model = word2vec.Word2Vec()
if not os.path.exists('models/word2vec.model'):
    model = word2vec.Word2Vec(sentences=X, vector_size=100, window=5, min_count=1, workers=4)
    model.save('models/word2vec.model')
else:
    model = word2vec.Word2Vec.load('models/word2vec.model')
#  wv = gensim.downloader.load('word2vec-google-news-300')
model = model.wv



#  model.wv = wv
#  model.train(X, total_examples = len(X), epochs = 3)
#  vectorizer = Word2Vec.load('word2vec.model')
#  features = []
#  for sentence in X:
#      s_embedding = sentence2embedding(model, sentence)
#      features.append(s_embedding)
#  features = np.array(features)
#  joblib.dump(vectorizer, "./models/tfidf_model.m")

#  lm = joblib.load("./models/tfidf_model.m")

def split_train_test(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test(X, y)

wv_X_train = []
wv_X_test = []
for sentence in X_train:
    s_embedding = sentence2embedding(model, sentence)
    wv_X_train.append(s_embedding)
wv_X_train = np.array(wv_X_train)
for sentence in X_test:
    s_embedding = sentence2embedding(model, sentence)
    wv_X_test.append(s_embedding)
#  wv_X_train = lm.transform(X_train)
#  wv_X_test = lm.transform(X_test)
wv_X_test = np.array(wv_X_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

n_features = 2

clf1 = DecisionTreeClassifier(max_depth=32, min_samples_split=2)
clf2 = RandomForestClassifier(n_estimators=32)
clf3 = ExtraTreesClassifier(n_estimators=32)
clf4 = SVC()

clf1.fit(wv_X_train, y_train)
preds1 = clf1.predict(wv_X_test)
print(preds1)
print('classification_report DecisionTreeClassifier:\n')
print(classification_report(preds1, y_test))

clf2.fit(wv_X_train, y_train)
preds2 = clf2.predict(wv_X_test)
print(preds2)
print('classification_report RandomForestClassifier:\n')
print(classification_report(preds2, y_test))

clf3.fit(wv_X_train, y_train)
preds3 = clf3.predict(wv_X_test)
print(preds3)
print('classification_report ExtraTreesClassifier:\n')
print(classification_report(preds3, y_test))

clf4.fit(wv_X_train, y_train)
preds4 = clf4.predict(wv_X_test)
print(preds4)
print('classification_report_SVC: \n')
print(classification_report(preds4, y_test))

finalpreds = []
for i, j, k, l in zip(preds1, preds2, preds3, preds4):
	tmp = {}
	tmp[i] = tmp.setdefault(i, 0) + 1
	tmp[j] = tmp.setdefault(j, 0) + 1
	tmp[k] = tmp.setdefault(k, 0) + 1
	tmp[l] = tmp.setdefault(l, 0) + 1
	lab = sorted(tmp.items(), key=lambda k:k[1], reverse=True)[0][0]
	finalpreds.append(lab)
print('classification_report cl_1234:\n')
print(classification_report(finalpreds, y_test))

# 5-fold cross validation
from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(clf1, wv_X_train, y_train)
scores2 = cross_val_score(clf2, wv_X_train, y_train)
scores3 = cross_val_score(clf3, wv_X_train, y_train)
scores4 = cross_val_score(clf4, wv_X_train, y_train)
print('DecisionTreeClassifier:'+str(scores1.mean()))
print('RandomForestClassifier:'+str(scores2.mean()))
print('ExtraTreesClassifier:'+str(scores3.mean()))
print('SVC:' + str(scores4.mean()))
