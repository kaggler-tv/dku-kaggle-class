# 데모

## 라이브러리 import 및 설정

%reload_ext autoreload
%autoreload 2
%matplotlib inline

!pip install -U pip

!pip install pandas

!pip install -U scikit-learn

!pip install -U tensorflow

!pip install -U tensorflow_hub

!pip install -U sentencepiece

import gc
from matplotlib import rcParams, pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import sys
sys.path.append('../src')
import tensorflow as tf
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.backend import clear_session
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Dropout
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.optimizers import Adam
import tensorflow_hub as hub
import tokenization
import warnings 
warnings.filterwarnings(action='ignore')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)
else:
    print('No GPU detected')

rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)
pd.set_option("display.precision", 4)
warnings.simplefilter('ignore')

## BERT Tokenizer 로드

http://nlp.stanford.edu/data/glove.6B.zip 를 다운받아 `data_dir`에 압축을 푼다.

data_dir = Path('../data/dacon-author-classification')
feature_dir = Path('../build/feature')
val_dir = Path('../build/val')
tst_dir = Path('../build/tst')
sub_dir = Path('../build/sub')
dirs = [feature_dir, val_dir, tst_dir, sub_dir]
for d in dirs:
    os.makedirs(d, exist_ok=True)

trn_file = data_dir / 'train.csv'
tst_file = data_dir / 'test_x.csv'
sample_file = data_dir / 'sample_submission.csv'
module_url = "https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1"

target_col = 'author'
n_fold = 5
n_class = 5
seed = 42

algo_name = 'bert'
max_len = 100
feature_name = f'n{max_len}'
model_name = f'{algo_name}_{feature_name}'

feature_file = feature_dir / f'{feature_name}.csv'
p_val_file = val_dir / f'{model_name}.val.csv'
p_tst_file = tst_dir / f'{model_name}.tst.csv'
sub_file = sub_dir / f'{model_name}.csv'

bert_layer = hub.KerasLayer(module_url, trainable=True)
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

## 학습데이터 로드

train = pd.read_csv(trn_file, index_col=0)
train.head()

test = pd.read_csv(tst_file, index_col=0)
test.head()

## Preprocessing

# https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub
def bert_encode(texts, tokenizer, max_len=max_len):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence)
        tokens += [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

trn = bert_encode(train.text.values, tokenizer, max_len=max_len)
tst = bert_encode(test.text.values, tokenizer, max_len=max_len)
y = train['author'].values
print(trn[0].shape, tst[0].shape, y.shape)

## Training

def get_model(bert_layer, max_len=max_len):
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    _, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    out = Dense(n_class, activation='sigmoid')(clf_output)
    
    model = Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

p_val = np.zeros((trn[0].shape[0], n_class))
p_tst = np.zeros((tst[0].shape[0], n_class))
for i, (i_trn, i_val) in enumerate(cv.split(trn[0], y), 1):
    print(f'training model for CV #{i}')
    es = EarlyStopping(monitor='val_loss', min_delta=0.001, patience=3,
                       verbose=1, mode='min', baseline=None, restore_best_weights=True)
    
    clf = get_model(bert_layer, max_len=max_len)
    if i == 1:
        print(clf.summary())
        
    clf.fit([x[i_trn] for x in trn], 
            to_categorical(y[i_trn]),
            validation_data=([x[i_val] for x in trn], to_categorical(y[i_val])),
            epochs=2,
            batch_size=16)
    p_val[i_val, :] = clf.predict([x[i_val] for x in trn])
    p_tst += clf.predict(tst) / n_fold
    
    del clf
    clear_session()
    gc.collect()

print(f'Accuracy (CV): {accuracy_score(y, np.argmax(p_val, axis=1)) * 100:8.4f}%')
print(f'Log Loss (CV): {log_loss(pd.get_dummies(y), p_val):8.4f}')

np.savetxt(p_val_file, p_val, fmt='%.6f', delimiter=',')
np.savetxt(p_tst_file, p_tst, fmt='%.6f', delimiter=',')

## 제출 파일 생성

sub = pd.read_csv(sample_file, index_col=0)
print(sub.shape)
sub.head()

sub[sub.columns] = p_tst
sub.head()

sub.to_csv(sub_file)


