# -*- coding: utf-8 -*-
"""
Created on Thu May 21 18:06:02 2020

@author: Jerry
"""

import numpy as np
import pandas as pd
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from tqdm import tqdm
import pickle
import keras.backend.tensorflow_backend as tb

# 加载tokenizer
def build_tokenizer(dict_path):
    tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
    return tokenizer

# 加载 bert 模型
def build_model(config_path,checkpoint_path):
    #tb._SYMBOLIC_SCOPE.value = False
    bert_model = build_transformer_model(config_path,checkpoint_path)
    return bert_model

# 生成句子向量特征
def extract_emb_feature(model,tokenizer,sentences,max_len,mask_if=False):
    #mask = generate_mask(sentences,max_len)
    token_ids_list = []
    segment_ids_list = []
    result = []
    #sentences = tqdm(sentences)
    for sen in sentences:
        token_ids, segment_ids = tokenizer.encode(sen,first_length = max_len)
        token_ids_list.append(token_ids)
        segment_ids_list.append(segment_ids)
        
    #print('Generating a sentence embedding')

    result = model.predict([np.array(token_ids_list), np.array(segment_ids_list)],verbose=0, batch_size=32)
    # if mask_if:
        # result = result * mask
    return np.mean(result,axis=1)

# 加载txt数据为list
def load_data(path):
    result=[]
    with open('{}'.format(path),'r',encoding = 'utf-8') as f:
    	for line in f:
    		result.append(line.strip('\n').split(',')[0])
    return result

# 保存文件为pkl
def save_pkl(name,dic):
    with open("{}.pkl".format(name), 'wb') as fo:     # 将数据写入pkl文件
        pickle.dump(dic, fo) 