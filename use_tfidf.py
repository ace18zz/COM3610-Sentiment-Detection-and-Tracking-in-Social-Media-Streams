'''
Author: your name
Date: 2021-03-10 15:36:55
LastEditTime: 2021-05-19 04:03:47
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /twitterSententAnakyse/use_tfidf.py
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import math
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# K fold split
# k_number = 5
# kf = KFold(k_number)
# def split(dataset): 
#     split_gen = kf.split(dataset)
#     return split_gen

LABELS = ['positive','neutral','negative']
def getData(filename):
	X = []
	y = []
	with open(filename) as fr:
		# fr.readline()
		for line in fr:
			line_split = line.strip().split('\t')
			if len(line_split) == 2:

				X.append(line_split[0].strip())
				y.append(LABELS.index(line_split[1].strip()))

	return X, y

X, y = getData('./filter_txt_train_data.txt')
# print(type(full_X))
# print(y[:3])

# for train_idx, test_idx in split(full_X):
# X_train = full_X[train_idx]
# y_train = full_y[train_idx]
# X_test = full_X[test_idx]
# y_test = full_X[test_idx]

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
features = vectorizer.fit_transform(X)
joblib.dump(vectorizer, "./models/tfidf_model.m")


lm = joblib.load("./models/tfidf_model.m")

def split_train_test(X, y):
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
	return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_train_test(X, y)

tf_X_train = lm.transform(X_train)
tf_X_test = lm.transform(X_test)


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

n_features = 2

clf1 = DecisionTreeClassifier(max_depth=32, min_samples_split=2)
clf2 = RandomForestClassifier(n_estimators=32)
clf3 = ExtraTreesClassifier(n_estimators=32)
clf4 = SVC()

clf1.fit(tf_X_train, y_train)
preds1 = clf1.predict(tf_X_test)
print(preds1)
print('classification_report DecisionTreeClassifier:\n')
print(classification_report(preds1, y_test))

clf2.fit(tf_X_train, y_train)
preds2 = clf2.predict(tf_X_test)
print(preds2)
print('classification_report RandomForestClassifier:\n')
print(classification_report(preds2, y_test))

clf3.fit(tf_X_train, y_train)
preds3 = clf3.predict(tf_X_test)
print(preds3)
print('classification_report ExtraTreesClassifier:\n')
print(classification_report(preds3, y_test))

clf4.fit(tf_X_train, y_train)
preds4 = clf4.predict(tf_X_test)
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
print('classification_report cl_1234：\n')
print(classification_report(finalpreds, y_test))

from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(clf1, tf_X_train, y_train)
scores2 = cross_val_score(clf2, tf_X_train, y_train)
scores3 = cross_val_score(clf3, tf_X_train, y_train)
scores4 = cross_val_score(clf4, tf_X_train, y_train)
print('DecisionTreeClassifier:'+str(scores1.mean()))
print('RandomForestClassifier:'+str(scores2.mean()))
print('ExtraTreesClassifier:'+str(scores3.mean()))
print('SVC:' + str(scores4.mean()))


# [1 0 1 ... 1 1 1]
# classification_report DecisionTreeClassifier：

#              precision    recall  f1-score   support

#           0       0.37      0.71      0.49       725
#           1       0.89      0.58      0.70      3166
#           2       0.20      0.54      0.29       235

# avg / total       0.76      0.60      0.64      4126

# [0 0 1 ... 1 1 1]
# classification_report RandomForestClassifier：

#              precision    recall  f1-score   support

#           0       0.52      0.68      0.59      1090
#           1       0.85      0.61      0.71      2883
#           2       0.16      0.65      0.25       153

# avg / total       0.74      0.63      0.66      4126

# [0 0 1 ... 1 0 2]
# classification_report ExtraTreesClassifier：

#              precision    recall  f1-score   support

#           0       0.54      0.68      0.60      1124
#           1       0.84      0.61      0.71      2849
#           2       0.16      0.67      0.26       153

# avg / total       0.73      0.63      0.66      4126

# classification_report cl_123：

#              precision    recall  f1-score   support

#           0       0.49      0.70      0.58       976
#           1       0.87      0.60      0.71      3000
#           2       0.15      0.64      0.25       150

# avg / total       0.75      0.63      0.66      4126

# DecisionTreeClassifier:0.5947043808806626
# RandomForestClassifier:0.619062244677851
# ExtraTreesClassifier:0.6289383556778382