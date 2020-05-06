# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 22:18:02 2020

@author: Jerry
"""

import pandas as pd
import numpy as np
import re
import jieba
from sklearn.model_selection import train_test_split                #划分训练/测试集
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB     # 从sklean.naive_bayes里导入朴素贝叶斯模型
from sklearn.preprocessing import StandardScaler
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
import xgboost as xgb    
from bayes_opt import BayesianOptimization
from bayes_opt.util import Colours
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score,roc_curve,auc,accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") 
import os,json,pickle
from sklearn.externals import joblib
from sklearn.calibration import CalibratedClassifierCV


# 中文分词函数，用正则去除多余的符号
def cut_text(text):
    text = str(text)
    stopwords = [line.strip() for line in open('chinese_stopwords.txt',encoding='UTF-8').readlines()]
    text = ''.join(re.findall('[\u4e00-\u9fff]', text))
    seg_list = jieba.cut(text)            
    sentence_segment=[] 
    for word in seg_list:
        if word not in stopwords:
            sentence_segment.append(word)
    #sentence_segment.append(word)        
    # 把已去掉停用词的sentence_segment，用' '.join()拼接起来
    seg_res = ' '.join(sentence_segment)
    return seg_res


def get_sentiment(txt, model = None):
    """ 获取文本情感
    :param txt: 输入的文本
    :return: 情感分析的结果，json格式
    """
    text = str(txt)
    text = cut_text(text)
    text_matrix = tfidf_vec.transform([text])
    text_prod = model.predict(text_matrix)
    #text_prod = model.predict_proba(text_matrix)
    #print('预测结果',test_model_pred)
    #print(np.argmax(test_model_pred)) 
    if text_prod[0] < 0.5:
        sentiment_label = 0
        sentiment_classification = '负面情感'
    else:
        sentiment_label = 1
        sentiment_classification = '正面情感'
    negative_prob = str(1 - text_prod[0])
    positive_prob = str(text_prod[0])
    result = {'text':txt,
              'sentiment_label':sentiment_label,
              'sentiment_classification':sentiment_classification,
              'negative_prob':negative_prob,
              'positive_prob':positive_prob}
    return json.dumps(result, ensure_ascii=False) 

        
if __name__ == "__main__":
    # 读取分词文件
    data = pd.read_csv('seg_ratings_data.txt',sep='\t')    
    # TfidfVectorizer 是 CountVectorizer + TfidfTransformer的组合，输出的各个文本各个词的TF-IDF值
    # min_df=5, max_features=10000
    tfidf_vec = TfidfVectorizer(max_features=10000) 
    tfidf_matrix = tfidf_vec.fit_transform(data['comment'].astype('U'))   
    # 划分数据集
    X_train,X_test,y_train,y_test = train_test_split(tfidf_matrix, data['rating'], test_size = 0.2, random_state = 1)#,stratify = y

    # 加载模型文件
    lgb_model = joblib.load('5_lgb_model.pkl')

    # 测试获取输入文本的情感
    text = '这店怎么这样了。第二次来吃，我买的套餐是5碟牛肉，最后只上了4碟，问老板娘，说已经改了只给4碟，没有这个套餐了。那我买这个券这个套餐，份量不给足我？ 你说改了就改了，我都不知情，那不你突然想加收就加收？那我买这个套餐写明有这些东西，那你一样也不能少，无论是什么时候买的，券都没有过期，难道我10年前买保险，10年后就不承认了？这等同于欺骗。以后不会再来，店铺没规律，没有诚信，只会做得越来越差，本来看着老板娘都是沙溪人就算了，免得在店铺里念叨。'
    get_sentiment(text, model = lgb_model)





'''
------------lightgbm调参明细--------------------


|   iter    |  target   | baggin... | featur... | lambda_l1 | lambda_l2 | max_depth | min_ch... | min_sp... | num_le... |
-------------------------------------------------------------------------------------------------------------------------
[1000]	cv_agg's auc: 0.918881 + 0.00107081
[2000]	cv_agg's auc: 0.923376 + 0.000964134
[3000]	cv_agg's auc: 0.924427 + 0.000955539
|  1        |  0.9246   |  0.9098   |  0.6722   |  3.014    |  1.635    |  6.69     |  34.07    |  0.04432  |  56.1     |
[1000]	cv_agg's auc: 0.917766 + 0.00103872
[2000]	cv_agg's auc: 0.921487 + 0.000962244
[3000]	cv_agg's auc: 0.92225 + 0.000942644
|  2        |  0.9223   |  0.9927   |  0.4068   |  3.959    |  1.587    |  7.266    |  46.65    |  0.008033 |  27.14    |
[1000]	cv_agg's auc: 0.919955 + 0.000993665
[2000]	cv_agg's auc: 0.922856 + 0.000989007
|  3        |  0.9232   |  0.804    |  0.7661   |  3.891    |  2.61     |  8.905    |  40.96    |  0.04669  |  52.1     |
[1000]	cv_agg's auc: 0.919916 + 0.00104383
[2000]	cv_agg's auc: 0.924986 + 0.00102221
[3000]	cv_agg's auc: 0.926312 + 0.000989509
|  4        |  0.9266   |  0.8237   |  0.6119   |  0.7168   |  2.834    |  7.082    |  23.66    |  0.02719  |  51.87    |
[1000]	cv_agg's auc: 0.919538 + 0.00105602
[2000]	cv_agg's auc: 0.923894 + 0.000991848
[3000]	cv_agg's auc: 0.924739 + 0.000994501
|  5        |  0.9248   |  0.8912   |  0.5547   |  0.09395  |  1.853    |  7.442    |  32.76    |  0.09443  |  48.55    |
[1000]	cv_agg's auc: 0.918384 + 0.00106539
[2000]	cv_agg's auc: 0.92474 + 0.00100561
[3000]	cv_agg's auc: 0.927346 + 0.000927246
[4000]	cv_agg's auc: 0.928655 + 0.00092156
[5000]	cv_agg's auc: 0.929355 + 0.000906554
[6000]	cv_agg's auc: 0.929767 + 0.000897905
[7000]	cv_agg's auc: 0.929996 + 0.000905485
|  6        |  0.9301   |  0.8963   |  0.1282   |  4.562    |  0.1954   |  5.561    |  5.283    |  0.002552 |  24.09    |
[1000]	cv_agg's auc: 0.917722 + 0.00107635
[2000]	cv_agg's auc: 0.924376 + 0.00103482
[3000]	cv_agg's auc: 0.927096 + 0.000994993
[4000]	cv_agg's auc: 0.928489 + 0.000964058
[5000]	cv_agg's auc: 0.929255 + 0.000949583
[6000]	cv_agg's auc: 0.929758 + 0.000968959
[7000]	cv_agg's auc: 0.929994 + 0.000963651
[8000]	cv_agg's auc: 0.930099 + 0.000979089
|  7        |  0.9301   |  0.9883   |  0.6073   |  4.948    |  2.69     |  5.943    |  5.322    |  0.09106  |  59.72    |



精度:0.859
召回:0.859
f1-score:0.859
accuracy_scores:0.859
AUC:0.932



'''

