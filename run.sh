#! /bin/bash
###
 # @Author: your name
 # @Date: 2021-05-17 03:12:17
 # @LastEditTime: 2021-05-18 06:52:50
 # @LastEditors: Please set LastEditors
 # @Description: In User Settings Edit
 # @FilePath: /twitterSententAnalyse 2/run.sh
### 

stage=$1

dataset='SemEval2017-task4-dev.subtask-A.english.INPUT.txt'
if [ $stage -le 0 ]; then
    python3 data_helper.py $dataset
fi

if [ $stage -le 1 ]; then
    echo "Use word bag preprocessor"
    python3 use_word_bag.py
fi

if [ $stage -le 2 ]; then
    echo "Use tf-idf preprocessor"
    python3 use_tfidf.py
fi

if [ $stage -le 3 ]; then
    echo "Use word2vec preprocessor"
    python3 use_word2vec.py 
fi

stage=$2
dataset='dataset3000.txt'
if [ $stage -le 0 ]; then
    python3 data_helper.py $dataset
fi

if [ $stage -le 1 ]; then
    echo "Use word bag preprocessor"
    python3 use_word_bag.py
fi

if [ $stage -le 2 ]; then
    echo "Use tf-idf preprocessor"
    python3 use_tfidf.py
fi

if [ $stage -le 3 ]; then
    echo "Use word2vec preprocessor"
    python3 use_word2vec.py 
fi
stage=$3
dataset='dataset6000.txt'
if [ $stage -le 0 ]; then
    python3 data_helper.py $dataset
fi

if [ $stage -le 1 ]; then
    echo "Use word bag preprocessor"
    python3 use_word_bag.py
fi

if [ $stage -le 2 ]; then
    echo "Use tf-idf preprocessor"
    python3 use_tfidf.py
fi

if [ $stage -le 3 ]; then
    echo "Use word2vec preprocessor"
    python3 use_word2vec.py 
fi
stage=$4
dataset='dataset9000.txt'
if [ $stage -le 0 ]; then
    python3 data_helper.py $dataset
fi

if [ $stage -le 1 ]; then
    echo "Use word bag preprocessor"
    python3 use_word_bag.py
fi

if [ $stage -le 2 ]; then
    echo "Use tf-idf preprocessor"
    python3 use_tfidf.py
fi

if [ $stage -le 3 ]; then
    echo "Use word2vec preprocessor"
    python3 use_word2vec.py 
fi
