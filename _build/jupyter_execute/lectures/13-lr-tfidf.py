# 데모

## 라이브러리 import 및 설정

%reload_ext autoreload
%autoreload 2
%matplotlib inline

from matplotlib import pyplot as plt
from matplotlib import rcParams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
import seaborn as sns
import warnings

rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)
pd.set_option("display.precision", 4)
warnings.simplefilter('ignore')

## 학습데이터 로드

data_dir = Path('../data/dacon-author-classification')
feature_dir = Path('../build/feature')
val_dir = Path('../build/val')
tst_dir = Path('../build/tst')
sub_dir = Path('../build/sub')

trn_file = data_dir / 'train.csv'
tst_file = data_dir / 'test_x.csv'
sample_file = data_dir / 'sample_submission.csv'

target_col = 'author'
n_fold = 5
n_class = 5
seed = 42

algo_name = 'lr'
feature_name = 'tfidf'
model_name = f'{algo_name}_{feature_name}'

feature_file = feature_dir / f'{feature_name}.csv'
p_val_file = val_dir / f'{model_name}.val.csv'
p_tst_file = tst_dir / f'{model_name}.tst.csv'
sub_file = sub_dir / f'{model_name}.csv'

trn = pd.read_csv(trn_file, index_col=0)
print(trn.shape)
trn.head()

tst = pd.read_csv(tst_file, index_col=0)
print(tst.shape)
tst.head()

## NLTK 예시

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from nltk.stem.snowball import SnowballStemmer

s = trn.text[4]
print(s)

tokens = word_tokenize(s)
print(tokens)

lemmatizer = WordNetLemmatizer()
[lemmatizer.lemmatize(t) for t in tokens]

stemmer = SnowballStemmer("english")
[stemmer.stem(t) for t in tokens]

## Bag-of-Words 피처 생성

vec = CountVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('english'), ngram_range=(1, 2), min_df=100)
X_cnt = vec.fit_transform(trn['text'])
print(X_cnt.shape)

X_cnt[0, :50].todense()

vec = TfidfVectorizer(tokenizer=word_tokenize, stop_words=stopwords.words('english'), ngram_range=(1, 3), min_df=50)
X = vec.fit_transform(trn['text'])
X_tst = vec.transform(tst['text'])
print(X.shape, X_tst.shape)

X[0, :50].todense()

## 로지스틱회귀 모델 학습

cv = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

y = trn.author.values
y.shape

p = np.zeros((X.shape[0], n_class))
p_tst = np.zeros((X_tst.shape[0], n_class))
for i_cv, (i_trn, i_val) in enumerate(cv.split(X, y), 1):
    clf = LogisticRegression()
    clf.fit(X[i_trn], y[i_trn])
    p[i_val, :] = clf.predict_proba(X[i_val])
    p_tst += clf.predict_proba(X_tst) / n_class

print(f'Accuracy (CV): {accuracy_score(y, np.argmax(p, axis=1)) * 100:8.4f}%')
print(f'Log Loss (CV): {log_loss(pd.get_dummies(y), p):8.4f}')

np.savetxt(p_val_file, p, fmt='%.6f', delimiter=',')
np.savetxt(p_tst_file, p_tst, fmt='%.6f', delimiter=',')

## 제출 파일 생성

sub = pd.read_csv(sample_file, index_col=0)
print(sub.shape)
sub.head()

sub[sub.columns] = p_tst
sub.head()

sub.to_csv(sub_file)