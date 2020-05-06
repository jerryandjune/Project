# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 11:25:27 2020

@author: Jerry
"""

import pandas as pd
import codecs, gc
import numpy as np
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.metrics import top_k_categorical_accuracy
from keras.layers import *
from keras.callbacks import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn import metrics
# 读取训练集和测试集
from sklearn.model_selection import train_test_split
from keras_bert import get_custom_objects
from keras.models import load_model
import json
import warnings
warnings.filterwarnings("ignore") 



# 初始参数设置
maxlen      = 128   # 设置序列长度为100，要保证序列长度不超过512
Batch_size  = 32    # 批量运行的个数
Epoch       = 2     # 迭代次数

def get_train_test_data():
    '''读取训练数据和测试数据 '''
    # train_df = pd.read_csv('/content/drive/My Drive/4-项目2/bert_train.csv').astype(str)
    # test_df = pd.read_csv('/content/drive/My Drive/4-项目2/bert_valid.csv').astype(str)
    train_df = pd.read_csv('/train_data/bert_train.csv').astype(str)
    test_df = pd.read_csv('/train_data/bert_valid.csv').astype(str)
    # 训练数据、测试数据和标签转化为模型输入格式
    DATA_LIST = []
    for data_row in train_df.iloc[:].itertuples():
        DATA_LIST.append((data_row.comment, to_categorical(data_row.rating, 2)))
    DATA_LIST = np.array(DATA_LIST)

    DATA_LIST_TEST = []
    for data_row in test_df.iloc[:].itertuples():
        DATA_LIST_TEST.append((data_row.comment, to_categorical(data_row.rating, 2)))
    DATA_LIST_TEST = np.array(DATA_LIST_TEST)

    data = DATA_LIST
    data_test = DATA_LIST_TEST

    X_train,X_valid = train_test_split(data,test_size=0.05,random_state = 0)
    return X_train, X_valid, data_test


def get_token_dict():
    """
    # 将词表中的字编号转换为字典
    :return: 返回自编码字典
    """
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict

# 重写tokenizer
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # 用[unused1]来表示空格类字符
            else:
                R.append('[UNK]')      # 不在列表的字符用[UNK]表示   UNK是unknown的意思
        return R

def seq_padding(X, padding=0):
    """
    :param X: 文本列表
    :param padding: 填充为0
    :return: 让每条文本的长度相同，用0填充
    """
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([ np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X])


class data_generator:
    """
    data_generator只是一种为了节约内存的数据方式
    """
    def __init__(self, data, batch_size=Batch_size, shuffle=True):
        """
        :param data: 训练的文本列表
        :param batch_size:  每次训练的个数
        :param shuffle: 文本是否打乱
        """
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))

            if self.shuffle:
                np.random.shuffle(idxs)

            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0][:maxlen]
                x1, x2 = tokenizer.encode(first=text)
                y = d[1]
                X1.append(x1)
                X2.append(x2)
                Y.append([y])
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y)
                    yield [X1, X2], Y[:, 0, :]
                    [X1, X2, Y] = [], [], []

def acc_top2(y_true, y_pred):
    """
    :param y_true: 真实值
    :param y_pred: 训练值
    :return: # 计算top-k正确率,当预测值的前k个值中存在目标类别即认为预测正确
    """
    return top_k_categorical_accuracy(y_true, y_pred, k=2)



def build_bert(nclass):
    """
    :param nclass: 文本分类种类
    :return: 构建的bert模型
    """
    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)  # 加载预训练模型

    for l in bert_model.layers:
        l.trainable = True
    #构建模型
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    p = Dense(nclass, activation='softmax')(x)

    model = Model([x1_in, x2_in], p)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),  # 用足够小的学习率
                  metrics=['accuracy', acc_top2])
    print(model.summary())
    return model


def train_model():
    """
    训练模型
    :return: 验证预测集，测试预测集，训练好的模型
    """
    # 搭建模型参数
    print('正在加载模型，请耐心等待....')
    model = build_bert(2)  # 二分类模型
    print('模型加载成功，开始训练....')
    early_stopping = EarlyStopping(monitor='val_acc', patience=3)  # 早停法，防止过拟合
    plateau = ReduceLROnPlateau(monitor="val_acc", verbose=1, mode='max', factor=0.5, patience=2)  # 当评价指标不在提升时，减少学习率
    checkpoint = ModelCheckpoint('model/bertkeras_model.h5', monitor='val_acc', verbose=2,
                                 save_best_only=True, mode='max', save_weights_only=True)  # 保存最好的模型
    # 获取数据并文本序列化
    X_train, X_valid, data_test = get_train_test_data()
    train_D = data_generator(X_train, shuffle=True)
    valid_D = data_generator(X_valid, shuffle=True)
    test_D = data_generator(data_test, shuffle=False)

    # 模型训练
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=Epoch,
        validation_data=valid_D.__iter__(),
        validation_steps=len(valid_D),
        callbacks=[early_stopping, plateau, checkpoint],
        )
    # 对验证集和测试集进行预测
    valid_D = data_generator(X_valid, shuffle=False)
    train_model_pred = model.predict_generator(valid_D.__iter__(), steps=len(valid_D), verbose=1)
    test_model_pred  = model.predict_generator(test_D.__iter__(), steps=len(test_D), verbose=1)
    # 将预测概率值转化为类别值
    train_pred = [np.argmax(x) for x in train_model_pred]
    test_pred = [np.argmax(x) for x in test_model_pred]
    y_true = [np.argmax(x) for x in X_valid[:, 1]]

    return train_pred, test_pred, y_true, model, data_test



def get_metrics_scores(y_true, y_pred, type ='metrics'):
    """
    :param y_true: 真实值
    :param y_pred: 预测值
    :param type: 预测种类
    :return: 评估指标
    """
    confusion_matrix_scores = metrics.confusion_matrix(y_true,y_pred)
    accuracy_scores = metrics.accuracy_score(y_true,y_pred)
    precision_scores = metrics.precision_score(y_true,y_pred,average = None)
    precision_score_macro_average = metrics.precision_score(y_true,y_pred,average = 'macro')
    precision_score_micro_average = metrics.recall_score(y_true,y_pred,average = 'micro')
    f1_scores = metrics.f1_score(y_true,y_pred,average = 'weighted')
    
    print(type,'...')
    print('混淆矩阵：', confusion_matrix_scores)
    print('准确率：', accuracy_scores)
    print('类别精度：', precision_scores) #不求平均
    print('宏平均精度：', precision_score_macro_average)
    print('微平均召回率:', precision_score_micro_average)
    print('加权平均F1得分:', f1_scores)
    return confusion_matrix_scores, accuracy_scores, precision_scores, precision_score_macro_average, precision_score_micro_average, f1_scores

def get_sentiment(txt):
    """ 获取文本情感
    :param txt: 输入的文本
    :return: 情感分析的结果，json格式
    """
    text = str(txt)
    DATA_text = []
    DATA_text.append((text, to_categorical(0, 2)))
    #DATA_text = np.array(DATA_text)
    text= data_generator(DATA_text, batch_size = 10, shuffle=False)
    test_model_pred = model.predict_generator(text.__iter__(), steps=len(text), verbose=0)
    #print('预测结果',test_model_pred)
    #print(np.argmax(test_model_pred)) 
    if test_model_pred[0][0] > test_model_pred[0][1]:
        sentiment_label = 0
        sentiment_classification = '负面情感'
    else:
        sentiment_label = 1
        sentiment_classification = '正面情感'
    negative_prob = str(test_model_pred[0][0])
    positive_prob = str(test_model_pred[0][1])
    result = {'text':txt,
              'sentiment_label':sentiment_label,
              'sentiment_classification':sentiment_classification,
              'negative_prob':negative_prob,
              'positive_prob':positive_prob}
    return json.dumps(result, ensure_ascii=False)



if __name__ == '__main__':
    # bert预训练模型路径设置
    # config_path = '/content/drive/My Drive/2-预训练模型/chinese_L-12_H-768_A-12/bert_config1.json'
    # checkpoint_path = '/content/drive/My Drive/2-预训练模型/chinese_L-12_H-768_A-12/bert_model.ckpt'
    # dict_path = '/content/drive/My Drive/2-预训练模型/chinese_L-12_H-768_A-12/vocab1.txt'
    config_path = 'chinese_L-12_H-768_A-12/bert_config.json'
    checkpoint_path = 'chinese_L-12_H-768_A-12/bert_model.ckpt'
    dict_path = 'chinese_L-12_H-768_A-12/vocab.txt'   
    # 获取新的tokenizer
    tokenizer = OurTokenizer(get_token_dict())
    # 训练和预测
    train_pred, test_pred, y_true, model, data_test = train_model()
    # 评估验证集
    get_metrics_scores(train_pred,y_true,type =' train metrics')
    # 评估测试集
    get_metrics_scores(test_pred,[np.argmax(x) for x in data_test[:, 1]],type =' test metrics')
    # 模型保存
    #model_path ='/content/drive/My Drive/3-模型保存/model/bertkeras_model.h5'
    model_path ='model/bertkeras_model.h5'
    model.save(model_path)


    # 模型加载
    custom_objects = get_custom_objects()
    my_objects = {'acc_top2': acc_top2}
    custom_objects.update(my_objects)
    model = load_model(model_path, custom_objects = custom_objects)


    # 单独评估一个本来分类
    text = '''
    肉类不新鲜，菜品比以前少，这样子就当吃麻辣烫一样，不过比麻辣烫还要差一点点。觉得都不新鲜的
    '''
    get_sentiment(text)


    #del model # 删除模型减少缓存
    gc.collect()  # 清理内存
    K.clear_session()  # clear_session就是清除一个session












'''
正在加载模型，请耐心等待....
Model: "model_6"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_3 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
input_4 (InputLayer)            (None, None)         0                                            
__________________________________________________________________________________________________
model_5 (Model)                 (None, None, 768)    101677056   input_3[0][0]                    
                                                                 input_4[0][0]                    
__________________________________________________________________________________________________
lambda_2 (Lambda)               (None, 768)          0           model_5[1][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 2)            1538        lambda_2[0][0]                   
==================================================================================================
Total params: 101,678,594
Trainable params: 101,678,594
Non-trainable params: 0
__________________________________________________________________________________________________
None
模型加载成功，开始训练....
Epoch 1/2
13004/13004 [==============================] - 6776s 521ms/step - loss: 0.2805 - accuracy: 0.8853 - acc_top2: 1.0000 - val_loss: 0.0562 - val_accuracy: 0.8969 - val_acc_top2: 1.0000
Epoch 2/2
13004/13004 [==============================] - 6760s 520ms/step - loss: 0.2368 - accuracy: 0.9053 - acc_top2: 1.0000 - val_loss: 0.3591 - val_accuracy: 0.8980 - val_acc_top2: 1.0000
685/685 [==============================] - 107s 156ms/step
63/63 [==============================] - 10s 153ms/step
 train metrics ...
混淆矩阵： [[9943 1083]
 [1150 9724]]
准确率： 0.8980365296803653
类别精度： [0.89633102 0.89978717]
宏平均精度： 0.898059097270533
微平均召回率: 0.8980365296803653
加权平均F1得分: 0.8980334099380205
 test metrics ...
混淆矩阵： [[883 103]
 [115 899]]
准确率： 0.891
类别精度： [0.88476954 0.89720559]
宏平均精度： 0.8909875639502558
微平均召回率: 0.891
加权平均F1得分: 0.8910052323348695

'''
