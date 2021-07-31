'''
Author: your name
Date: 2021-05-17 03:11:15
LastEditTime: 2021-05-18 14:46:09
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /twitterSententAnalyse 2/data_helper.py
'''
#coding:utf-8
from ext_en import *
import sys

def loadStops():
    stops = []
    with open('./stopword.txt', encoding='utf-8') as fr:
        for line in fr:
            stops.append(line.strip())
    return stops

def getTxt():
    y = []
    X = []
    with open('./data/{}'.format(sys.argv[1])) as fr:
        for line in fr:
            try:
                newline = line.strip().split('\t')
                
                if newline[-2] in ['positive', 'negative', 'neutral']:
                    y.append(newline[-2])
                    X.append(newline[-1])
            except:
                pass
    return X, y

stops = loadStops()
X, y = getTxt()

X_feather = []

fw = open('./filter_txt_train_data.txt', 'w')
for x,y in zip(X, y):
    text = x.strip('"')
    token_words = tokenize(text)
    token_words = stem(token_words)
    token_words = delete_stopwords(token_words)
    token_words = delete_characters(token_words)
    token_words = to_lower(token_words)
    fw.write(' '.join(token_words)+'\t'+y+'\n')
    fw.flush()

