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

# # 生成mask矩阵
# def generate_mask(sen_list,max_len):
#     len_list = [len(i) if len(i)<=max_len else max_len for i in sen_list]
#     array_mask = np.array([np.hstack((np.ones(j),np.zeros(max_len-j))) for j in len_list])
#     return np.expand_dims(array_mask,axis=2)

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


if __name__ == '__main__':
    # 设置bert文件路径
    dict_path = r"..\Flask\app\static\model04\search_database\chinese_L-12_H-768_A-12\vocab.txt"
    config_path = r"..\Flask\app\static\model04\search_database\chinese_L-12_H-768_A-12\bert_config.json"
    checkpoint_path = r"..\Flask\app\static\model04\search_database\chinese_L-12_H-768_A-12\bert_model.ckpt"
    # 加载tokenizer
    tokenizer = build_tokenizer(dict_path)
    # 加载bert模型
    model = build_model(config_path,checkpoint_path)
    
    '''问答数据 向量化处理'''
    # 加载question文本
    data = pd.read_csv(r'..\Flask\app\static\model04\项目4数据集.csv')
    finance_data = data.drop_duplicates('question')
    finance_data = finance_data[(finance_data['question'] != '') & (finance_data['answer'] != '')]
    finance_data_question_list = finance_data['question'].astype(str).to_list()
    finance_data_answer_dict = dict(zip(finance_data.index, finance_data.answer))
    # 保存answer
    save_pkl('finance_data_answer_dict',finance_data_answer_dict)
    # 开始抽取文本特征，获取句向量
    sentence_emb = extract_emb_feature(model, tokenizer, finance_data_question_list, max_len = 30)
    # 保存文本特征向量化的list为pkl
    save_pkl('sentence_emb',sentence_emb)
    # 把问题和问题对应的向量组成dict
    res = dict(zip(finance_data_question_list,sentence_emb))
    # 字典文件可以做向量检索使用
    save_pkl('sentence_emb_dict',res)


#v =   extract_emb_feature(model, tokenizer, ['吞卡证明啊'], max_len = 30)      
    
    
    
#v =   extract_emb_feature(model, tokenizer, ['手机为什么不能转钱？'], max_len = 30)  

'''
  0%|          | 0/328172 [00:00<?, ?it/s]

  1%|          | 3184/328172 [00:00<00:10, 31526.37it/s]

  2%|▏         | 6398/328172 [00:00<00:10, 31707.75it/s]

  3%|▎         | 9719/328172 [00:00<00:09, 32050.84it/s]

  4%|▍         | 13084/328172 [00:00<00:09, 32514.38it/s]
  .
  .
  .
  .
 95%|█████████▍| 310978/328172 [00:10<00:00, 26671.20it/s]

 96%|█████████▌| 313800/328172 [00:10<00:00, 26739.60it/s]

 96%|█████████▋| 316648/328172 [00:10<00:00, 27233.14it/s]

 97%|█████████▋| 319471/328172 [00:10<00:00, 27515.28it/s]

 98%|█████████▊| 322381/328172 [00:10<00:00, 27892.04it/s]

100%|██████████| 328172/328172 [00:11<00:00, 29450.29it/s]

Generating a sentence embedding

328172/328172 [==============================] - 6395s 19ms/step

'''