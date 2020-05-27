# -*- coding: utf-8 -*-
"""
Created on Sat May 16 22:23:03 2020

@author: Jerry
"""


from gensim.models import KeyedVectors
import json
from collections import OrderedDict
from annoy import AnnoyIndex

import time
import json
import datetime
import numpy as np
import pickle


# 保存字典文件为pkl
def save_dict(name,dic):
    with open("{}.pkl".format(name), 'wb') as fo:     # 将数据写入pkl文件
        pickle.dump(dic, fo)    
    
# 读取pkl文件
def load_dict(name, dic):
    with open("{}.pkl".format(name), 'rb') as fo:     # 读取pkl文件
        dic = pickle.load(fo, encoding='bytes')      
    return dic

# 使用gensim加载词向量
def load_model(path):
    model = KeyedVectors.load_word2vec_format(path, 
                                              binary=False, 
                                              encoding='utf8', 
                                              unicode_errors='ignore')
    return model


# 主函数: 构建向量索引文件，word-->ID 映射表,反向ID-->word映射表
def main():
    # 加载百度百科词向量
    bk_wv_model = load_model('H:/1-开课吧/05-项目/项目3/sgns.baidubaike.bigram-char')
    # 构建一份词汇ID映射表，并以json格式离线保存一份（这个方便以后离线直接加载annoy索引时使用）
    word_index = OrderedDict()
    for counter, key in enumerate(bk_wv_model.vocab.keys()):
        word_index[key] = counter
    
    with open('bk_word_index.json', 'w') as fp:
        json.dump(word_index, fp)
    # 保存词汇ID映射表，加速向量检索时需要用
    save_dict('word_index',word_index)
    
    # 开始基于百度百科词向量构建Annoy索引，词向量大概是65万条
    # 百度百科词向量的维度是300
    bk_index = AnnoyIndex(300)   
    i = 0    
    for key in bk_wv_model.vocab.keys():
        v = bk_wv_model[key]
        bk_index.add_item(i, v)
        i += 1
    # n_trees越大，构建的时间也越长，检索精度越高
    bk_index.build(200)
    # 保存向量索引文件，大概2.3G,注意：【【【保存路径不能有中文】】】
    bk_index.save('H:/bk_index_build200.index')    
    # 准备一个反向id==>word映射词表
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()]) 
    # 保存词汇反向id==>word映射词表，加速向量检索时需要用
    save_dict('reverse_word_index',reverse_word_index)
    
    
if __name__ == '__main__':
    main()



