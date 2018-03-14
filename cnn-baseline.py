#!/usr/bin/env Python
# coding=utf-8
import pandas  as pd
import numpy as np
from  keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dropout, Dense, BatchNormalization, Activation, concatenate, GlobalMaxPool1D,GRU, Embedding,Bidirectional, Flatten,LSTM, BatchNormalization,Conv1D,MaxPooling1D
from keras.models import Model
from keras.layers import GlobalMaxPooling1D
from keras.layers import *
from keras.layers.convolutional import Convolution1D
from keras import optimizers
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras import backend as K
from keras import regularizers
import re
import jieba
from sklearn.model_selection import StratifiedKFold,KFold
import jieba.posseg
import jieba.analyse
import codecs
from keras.layers import Input, Concatenate

train = pd.read_csv('data/train_first.csv')
test = pd.read_csv('data/predict_first.csv')

max_features = 80000 ## 词汇量
maxlen = 150  ## 最大长度
embed_size = 200  # emb 长度

# 预处理：缺失值填充+结巴分词+去停用词 将词映射为ID

#train = train.drop_duplicates(subset='Discuss', keep='first',inplace=True)
#train['Discuss'] = train['Discuss'].drop_duplicates(keep='first', inplace=True)

def threshold(result):
    boolean = (result['Score'] >= 4.0) & (result['Score'] < 4.732)
    result['Score'].ix[boolean] = 4
    result['Score'].ix[(result['Score'] >= 4.732)] = 5.0
    result['Score'].ix[result['Score'] < 1] = 1.0
    return result

def threshold_score(result):
    for index, i in enumerate(result):
        if i >= 4.0 and i < 4.732:
            i = 4
        elif i >= 4.732:
            i = 5.0
        elif i < 1 :
            i = 1.0
        result[index] = i
    return result

def splitWord(query, stopwords):
    #query = re.sub(r'[a-zA-Z0-9]+','',query)
    wordList = jieba.cut(query)
    num = 0
    result = ''
    for word in wordList:
        word = word.rstrip()
        word = word.rstrip('"')
        if word not in stopwords:
            if num == 0:
                result = word
                num = 1
            else:
                result = result + ' ' + word
    return result.encode('utf-8')

def preprocess(data):
    stopwords = {}
    # for line in codecs.open('data/stop.txt','r','utf-8'):
    for line in codecs.open('data/stop.txt', 'r', 'gbk'):
        stopwords[line.rstrip()]=1
    data['doc'] = data['Discuss'].map(lambda x: splitWord(x, stopwords))
    return data

train.Discuss.fillna('_na_',inplace=True)
test.Discuss.fillna('_na_',inplace=True)
train = preprocess(train)
test = preprocess(test)

comment_text = np.hstack([train.doc.values])
tok_raw = Tokenizer(num_words=max_features)
tok_raw.fit_on_texts(comment_text)
train['Discuss_seq'] = tok_raw.texts_to_sequences(train.doc.values)
test['Discuss_seq'] = tok_raw.texts_to_sequences(test.doc.values)

def get_keras_data(dataset):
    X={
        'Discuss_seq':pad_sequences(dataset.Discuss_seq,maxlen=maxlen)
    }
    return X


def score(y_true, y_pred):
    return 1.0 / (1.0 + K.sqrt(K.mean(K.square(y_true - y_pred), axis=-1)))

def rmsel(true_label,pred):
    true_label = np.array(true_label)
    pred = np.array(pred)
    n = len(true_label)
    a = true_label - pred
    rmse = np.sqrt(np.sum(a * a)/n)
    b = 1/(1+rmse)
    return b


def cnn():
    # Inputs
    comment_seq = Input(shape=[maxlen], name='Discuss_seq')

    # Embeddings layers
    emb_comment = Embedding(max_features, embed_size)(comment_seq)

    # conv layers
    convs = []
    filter_sizes = [2, 3, 4, 5]
    for fsz in filter_sizes:
        l_conv = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_pool = MaxPooling1D(maxlen - fsz + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    out = Dropout(0.5)(merge)
    output = Dense(32, activation='relu')(out)

    output = Dense(units=1, activation='linear')(output)

    model = Model([comment_seq], output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", score])
    return model

# text-cnn 多filter_size
def text_cnn():
    # Inputs
    comment_seq = Input(shape=[maxlen], name='Discuss_seq')

    # Embeddings layers
    emb_comment = Embedding(max_features, embed_size)(comment_seq)

    # conv layers
    convs = []
    filter_sizes = [1, 2, 3, 4]
    for fsz in filter_sizes:
        l_conv1 = Conv1D(filters=100, kernel_size=fsz, activation='relu')(emb_comment)
        l_conv1 = BatchNormalization()(l_conv1)

        l_conv2 = Conv1D(filters=100, kernel_size=fsz, activation='relu')(l_conv1)
        l_conv2 = BatchNormalization()(l_conv2)

        l_pool = MaxPooling1D(maxlen - fsz + 1 - fsz +1)(l_conv2)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    #out = Dropout(0.5)(merge)
    out = merge
    output = Dense(100, activation='relu')(out)

    output = BatchNormalization()(output)

    output = Dense(units=1, activation='linear')(output)

    model = Model([comment_seq], output)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", score])
    return model


def cnn_rnn():
    comment_seq = Input(shape=[maxlen], name='Discuss_seq')

    # Embeddings layers
    main = Embedding(max_features, embed_size)(comment_seq)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Bidirectional(GRU(32))(main)
    main = Dense(units=1, activation='linear')(main)

    model = Model([comment_seq], main)
    adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss="mse", optimizer="adam", metrics=["mae", score])
    return model



X_train =get_keras_data(train)
X_test = get_keras_data(test)
y_train = train.Score.values

batch_size = 128
epochs = 20
early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=2)

# 训练
callbacks_list = [early_stopping]
model = cnn()
model.summary()
model.fit(X_train, y_train,
            validation_split=0.1,
            batch_size=batch_size,
            epochs=epochs,
            shuffle = True,
            callbacks=callbacks_list)


# 预测
preds = model.predict(X_test)
submission =pd.DataFrame(test.Id.values,columns=['Id'])
preds = threshold_score(preds)
submission['Score'] = preds
submission.to_csv('results/cnn-baseline.csv', index=None, header=None)


# def fast_cv(df):
#     df = df.reset_index(drop=True)
#     X = df['Discuss'].values
#     y = df['Score'].values
#     fast_pred = []
#     folds = list(KFold(n_splits=5, shuffle=True, random_state=2017).split(X, y))
#     for train_index, test_index in folds:
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
#
#         train_file = fasttext_data(X_train, y_train)
#
#         classifier = fasttext.supervised(train_file, 'model.model', lr=0.01, dim=128, label_prefix="__label__")
#         result = classifier.predict_proba(df.loc[test_index, 'Discuss'].tolist(), k=5)
#
#         pred = [[int(sco) * proba for sco, proba in result_i] for result_i in result]
#         pred = [sum(pred_i) for pred_i in pred]
#         print(rmsel(y_test, pred))
#
#         test_result = classifier.predict_proba(test_df['Discuss'].tolist(), k=5)
#         fast_predi = [[int(sco) * proba for sco, proba in result_i] for result_i in test_result]
#         fast_predi = [sum(pred_i) for pred_i in fast_predi]
#         fast_pred.append(fast_predi)
#
#     fast_pred = np.array(fast_pred)
#     fast_pred = np.mean(fast_pred, axis=0)
#     return fast_pred
