# Pandas 데모 - Explarotary Data Analysis

## 라이브러리 import 및 설정

%reload_ext autoreload
%autoreload 2
%matplotlib inline

from matplotlib import pyplot as plt
from matplotlib import rcParams
import numpy as np
from pathlib import Path
import pandas as pd
import seaborn as sns
import warnings

rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)
pd.set_option("display.precision", 4)
warnings.simplefilter('ignore')

## 데이터 다운로드

데이터는 [Dacon 단국대 소/중 데이터 분석 AI 경진대회 웹사이트](https://www.dacon.io/competitions/official/235638/data/)에서 다운로드 받아 `../input` 폴더에 저장.

!ls -alF ../input/

data_dir = Path('../input/')
trn_file = data_dir / 'train.csv'
tst_file = data_dir / 'test.csv'
feature_file = data_dir / 'feature.csv'
seed = 42

## EDA

### 학습데이터 로드

trn = pd.read_csv(trn_file, index_col=0)
print(trn.shape)
trn.head()

trn.tail()

### 데이터 개요

trn.describe()

trn.dtypes

### 종속변수 분포

trn['class'].value_counts().sort_index()

trn['class'].hist()

### 독립변수 분포

trn['i'].hist(bins=100)

np.arange(0, 1, .01)[:10]

trn['i'].quantile(np.arange(0, 1, .01))

trn.loc[trn['i'] < 0]

trn['dered_i'].quantile(np.arange(0, 1, .01))

trn.loc[trn['dered_i'] < 0]

### 시각화

trn_sample = trn.sample(n=10000, random_state=seed)

sns.pairplot(data=trn_sample, vars=['u', 'g', 'r', 'i', 'z'], hue='class', size=5)

sns.pairplot(data=trn_sample, vars=['dered_u', 'dered_g', 'dered_r', 'dered_i', 'dered_z'], hue='class', size=5)

sns.pairplot(data=trn_sample, vars=['airmass_u', 'airmass_g', 'airmass_r', 'airmass_i', 'airmass_z'], hue='class', size=5)

sns.pairplot(data=trn_sample, vars=['u', 'dered_u', 'airmass_u'], hue='class', size=5)

sns.pairplot(data=trn_sample, vars=['redshift', 'nObserve', 'nDetect'], hue='class', size=5)

trn.groupby('class').mean()

trn.groupby('class').mean().T.plot(kind='barh')

### 시험 데이터 로드

tst = pd.read_csv(tst_file, index_col=0)
print(tst.shape)
tst.head()

### 학습/시험 데이터 결합

df = pd.concat([trn, tst], axis=0)
print(df.shape)
df.tail()

df.fillna(-1, inplace=True)
df.tail()

### 피쳐 변환

df['nObserve'].hist(bins=30)

df['nObserve'] = df['nObserve'].apply(np.log1p)

df['nObserve'].hist(bins=30)

### 피쳐 생성

df['d_dered_u'] = df['dered_u'] - df['u']
df['d_dered_g'] = df['dered_g'] - df['g']
df['d_dered_r'] = df['dered_r'] - df['r']
df['d_dered_i'] = df['dered_i'] - df['i']
df['d_dered_z'] = df['dered_z'] - df['z']
df['d_dered_rg'] = df['dered_r'] - df['dered_g']
df['d_dered_ig'] = df['dered_i'] - df['dered_g']
df['d_dered_zg'] = df['dered_z'] - df['dered_g']
df['d_dered_ri'] = df['dered_r'] - df['dered_i']
df['d_dered_rz'] = df['dered_r'] - df['dered_z']
df['d_dered_iz'] = df['dered_i'] - df['dered_z']
df['d_obs_det'] = df['nObserve'] - df['nDetect']
print(df.shape)
df.head()

### 피쳐 삭제

df.corr().style.background_gradient()

df.drop(['airmass_z', 'airmass_i', 'airmass_r', 'airmass_g', 'u', 'g', 'r', 'i', 'nDetect', 'd_dered_rg', 'd_dered_ri'], 
        axis=1, inplace=True)
print(df.shape)
df.head()

### 새로운 학습 데이터 파일 저장

df.to_csv(feature_file)

!ls -alF ../input/

feature = pd.read_csv(feature_file, index_col=0)
print(feature.shape)
feature.head()

feature.loc[feature['class'] != -1].corr().style.background_gradient()

