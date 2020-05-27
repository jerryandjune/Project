# -*- coding: utf-8 -*-
"""
Created on Fri May 15 14:37:05 2020

@author: Jerry
"""

import jieba
import numpy as np
import pandas as pd
import time
from functools import wraps
import gc
import jieba.analyse
from collections import Counter
from flask import Flask,jsonify
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import pickle
import json
from collections import OrderedDict
from annoy import AnnoyIndex

class Keywords(object):
    '''基于textrank提取关键字'''
    def __init__(self):
        pass

    @staticmethod
    def get_keywords(sentence, topn = 20, weight = 0.5):#
        ''' 基于textrank算法提取关键字'''
        #词性限制集合为["ns", "n", "vn", "v", "nr"]，表示只能从词性为地名、名词、动名词、动词、人名这些词性的词中抽取关键词。
        allow_pos = ("ns", "n", "vn", "v", "nr")
        keywords = jieba.analyse.textrank(sentence, topK=topn, withWeight=True, allowPOS = allow_pos )
        #print(keywords)
        keywords_item = {}
        for item in keywords:
            # 取权重大于定义的weight
            if item[1]  > weight:
                keywords_item[item[0]] = item[1]    

        return keywords_item    



class Similarwords(object):
    '''基于word2vec提取相似词'''    
    model = None
    # 加载模型文件
    def __init__(self):
        if Similarwords.model is None:            
            Similarwords.model = KeyedVectors.load_word2vec_format('sgns.baidubaike.bigram-char',
                                                 binary=False, encoding="utf8",  unicode_errors='ignore')  

    @staticmethod
    def get_similar_words(text, topn = 20, weight = 0.5):#
        ''' 基于textrank算法提取关键字'''
        sim_words = Similarwords.model.most_similar(text,topn = topn)
        #print(keywords)
        sim_words_item = {}
        for item in sim_words:
            # 取权重大于定义的weight
            if item[1]  > weight:
                sim_words_item[item[0]] = item[1]    

        return sim_words_item


class Annoysimilarwords():
    '''基于annoy提取相似词'''    
    model = None
    word_index = {}
    reverse_word_index = {}
    # 加载模型文件
    def __init__(self):
        with open("word_index.pkl", 'rb') as fo:     # 读取pkl文件
            Annoysimilarwords.word_index = pickle.load(fo, encoding='bytes') 
        with open("reverse_word_index.pkl", 'rb') as fo:     # 读取pkl文件
            Annoysimilarwords.reverse_word_index = pickle.load(fo, encoding='bytes')
                    
        try:
            if Annoysimilarwords.model is None:            
                Annoysimilarwords.model = AnnoyIndex(300,metric='angular')
                Annoysimilarwords.model.load('bk_index_build200.index')
        except Exception as e:
            print(e)
       
    @staticmethod
    def get_similar_words(text, topn = 20, weight = 0.5):#
        ''' 基于annoy算法提取相似词'''        
        sim_words = Annoysimilarwords.model.get_nns_by_item(Annoysimilarwords.word_index[text], topn, include_distances=True)
        #print(keywords)
        sim_words_item = {}
        for item, j in zip(sim_words[0],sim_words[1] ):
            # 余弦相似度归一化的计算公式，
            weights = 0.5*(abs(1-j))+0.5
            # 取权重大于定义的weight
            if weights > weight:
                sim_words_item[Annoysimilarwords.reverse_word_index[item]] = weights  

        return sim_words_item    



if __name__ == '__main__':
  
    # 关键字提取
    sens = '包括本保险条款、保险单、投保单、与本合同有关的投保文件、保险凭证、合法有效的声明、批注、批单及其他您与我们共同认可的书面协议'
    ky = Keywords()
    keywords_result = Keywords.get_keywords(sentence = sens,topn = 20)
    print(keywords_result)
    # 基于词向量，加载模型时间约135秒
    #sw = Similarwords()
    #sim_words = sw.get_similar_words('美女', weight = 0.2)
    
    # 基于Annoy词向量搜索，超级快
    bk = Annoysimilarwords()
    bk_sim_words = bk.get_similar_words('美女', weight = 0.2) 
    print(bk_sim_words)

    kwlist = list(jieba.cut('缴费条款'))
    bk_sim_words = bk.get_similar_words(kwlist[0], weight = 0.2)
    bk_sim_words.update(bk.get_similar_words(kwlist[1], weight = 0.2))
    bk_sim_words = {}
    bk_sim_words = [bk_sim_words.update(bk.get_similar_words(i, weight = 0.2)) for i in kwlist]
    print(bk_sim_words)

    bk_sim_words = [i for i in kwlist]
    print(bk_sim_words)
            
# from gensim.models import KeyedVectors
# import json
# from collections import OrderedDict
# from annoy import AnnoyIndex

# import time
# import json
# import datetime
# import numpy as np
# import pickle



# start = time.time()
# tc_wv_model = KeyedVectors.load_word2vec_format('D:\BaiduNetdiskDownload\sgns.baidubaike.bigram-char\sgns.baidubaike.bigram-char',binary=False, encoding='utf8',  unicode_errors='ignore') 
# end = time.time()   
# print("Execution Time: ", end - start)
       
# word_index = OrderedDict()

# for counter, key in enumerate(tc_wv_model.vocab.keys()):
#     word_index[key] = counter


# with open('tc_word_index.json', 'w') as fp:
#     json.dump(word_index, fp)


# tc_index = AnnoyIndex(300)

# i = 0

# for key in tc_wv_model.vocab.keys():
#     v = tc_wv_model[key]
#     tc_index.add_item(i, v)
#     i += 1

# tc_index.build(200)

# tc_index.save(r'H:\tc_index_build200.index')

# tc_index.get_nns_by_item(word_index[u'保额'], 11,include_distances=True)
# reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 
# reverse_word_index
# u = AnnoyIndex(300)
# u.load(r'H:\tc_index_build200.index')
# result = u.get_nns_by_item(word_index[u'保额'], 11,include_distances=True)




# with open(r"H:\word_index.pkl", 'wb') as fo:     # 将数据写入pkl文件
#     pickle.dump(word_index, fo)


# with open(r"H:\word_index.pkl", 'rb') as fo:     # 读取pkl文件数据
#     word_index = pickle.load(fo, encoding='bytes')






# with open(r"H:\reverse_word_index.pkl", 'wb') as fo:     # 将数据写入pkl文件
#     pickle.dump(reverse_word_index, fo)


# with open(r"H:\reverse_word_index.pkl", 'rb') as fo:     # 读取pkl文件数据
#     reverse_word_index = pickle.load(fo, encoding='bytes')
# reverse_word_index[1]
# result[0]
# for item,j in zip(result[0],result[1]):
    
#     print(reverse_word_index[item],0.5*(abs(1-j))+0.5)


# tc_wv_model.most_similar('保额')


            
            