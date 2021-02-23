# 결정트리 데모

## 라이브러리 import 및 설정

%reload_ext autoreload
%autoreload 2
%matplotlib inline

import graphviz
from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import warnings

rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)
pd.set_option("display.precision", 4)
warnings.simplefilter('ignore')

## 데이터 로드

data_dir = Path('../data/dacon-dku')
sub_dir = Path('../build/sub')

trn_file = data_dir / 'train.csv'
tst_file = data_dir / 'test.csv'
sample_file = data_dir / 'sample_submission.csv'

target_col = 'class'
seed = 42

algo_name = 'dt'
feature_name = 'j1'
model_name = f'{algo_name}_{feature_name}'

sub_file = sub_dir / f'{model_name}.csv'

trn = pd.read_csv(trn_file, index_col=0)
tst = pd.read_csv(tst_file, index_col=0)
y = trn[target_col]
trn.drop(target_col, axis=1, inplace=True)
print(y.shape, trn.shape, tst.shape)
trn.head()

## 결정트리 학습

clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
clf.fit(trn, y)

print(f'{accuracy_score(y, clf.predict(trn)) * 100:.4f}%')

## 결정트리 시각화

dot_data = export_graphviz(clf, out_file=None,
                           feature_names=trn.columns,
                           filled=True,
                           rounded=True,
                           special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

## 시험데이터 예측

sub = pd.read_csv(sample_file, index_col=0)
print(sub.shape)
sub.head()

sub[target_col] = clf.predict(tst)
sub.head()

sub[target_col].value_counts()

## 제출파일 저장

sub.to_csv(sub_file)

