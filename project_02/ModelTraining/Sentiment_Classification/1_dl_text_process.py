# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:15:32 2020

@author: Jerry
"""
import pandas as pd
import numpy as np
import os, sys
import logging
import re
import jieba
from sklearn.model_selection import train_test_split                #划分训练/测试集
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
import xgboost as xgb    
from bayes_opt import BayesianOptimization
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score,roc_curve,auc
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.utils import shuffle


# 读取数据并使用tokens函数提取中文
def tokens(text):   
    return ''.join(re.findall('[\u4e00-\u9fff]', text))


# 读取评价数据函数
def read_ratings_data():
    print('开始读取文件。。。')
    # 用pd.read_csv方法读取语料
    pd_ratings = pd.read_csv('yf_dianping/ratings.csv')
    ratings_data = pd_ratings[(pd_ratings['rating'].notnull()) & (pd_ratings['comment'].notnull())][['comment','rating']]
    del pd_ratings
    ratings_data = ratings_data.loc[:, ['comment', 'rating']]
    # 设置标签,0为负面情感，1为正面情感
    ratings_data.loc[ratings_data['rating'] < 3, 'rating'] = 0
    ratings_data.loc[ratings_data['rating'] >= 3, 'rating'] = 1
    ratings_data['rating'] = ratings_data['rating'].astype(int)
    
    data0 = ratings_data[(ratings_data['rating']==0) & (ratings_data['comment'].str.len()>20)][:250000]
    data1 = ratings_data[(ratings_data['rating']==1) & (ratings_data['comment'].str.len()>20)][:len(data0)]
    
    data = pd.concat([data0,data1])
    # 把数据打乱
    data = shuffle(data,random_state = 0)   
    print('读取完毕。。。')
    # 使用apply对comment列去掉多余符号
    #data['comment'] = data['comment'].apply(cut_text)
    
    return data




if __name__ == '__main__':
    # print('开始读取文件')
    data = read_ratings_data()
    # 使用apply对comment列去掉多余符号
    print('正在处理文本，请耐心等待。。。')
    data['comment'] = data['comment'].apply(tokens)
    print('处理完毕，正在导出csv文件。。。')
    # 把数据导出为txt文本，index和header为空，分隔符为4个空格，即"\t"
    data[['comment','rating']][:438000].to_csv('train_data/bert_train.csv',index=None)
    data[['comment','rating']][438000:].to_csv('train_data/bert_valid.csv',index=None)
    #data[['comment','rating']][440000:].to_csv('/content/drive/My Drive/4-项目2/bert_test.csv',index=None)
    print('导出完毕。。。')





