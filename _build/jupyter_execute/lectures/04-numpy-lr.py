# 선형회귀 데모

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
from tqdm.notebook import tqdm
import warnings

rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')
pd.set_option('max_columns', 100)
pd.set_option("display.precision", 4)
warnings.simplefilter('ignore')

## 학습데이터 로드

data_dir = Path('../input/')
trn_file = data_dir / 'train.csv'
seed = 42

trn = pd.read_csv(trn_file, index_col=0)
print(trn.shape)
trn.head()

sns.pairplot(trn.sample(n=10000), vars=['u', 'dered_u'], hue='class', size=5)

X = trn[trn['class'] != 0][['u']].values
y = trn[trn['class'] != 0]['dered_u'].values
print(X.shape, y.shape)

## RMSE 손실함수 정의

rmse = lambda y, p: np.sqrt(np.mean((y - p) ** 2))

## NumPy를 이용한 선형회귀

alpha = .001
theta0 = .1
theta1 = .1
epoch = 4

x = X.flatten()
print(x.shape)

rmses = []
for i in tqdm(range(epoch)):
    for x_i, y_i in zip(x, y):
        p_i =  theta0 + theta1 * x_i
        theta0 = theta0 + alpha * (y_i - p_i)
        theta1 = theta1 + alpha * (y_i - p_i) * x_i
    rmses.append(rmse(y, theta0 + theta1 * x))

theta1, theta0

rmses

plt.plot(rmses)

plt.figure(figsize=(8, 8))
plt.scatter(X, y, alpha=.1)
plt.plot(X, theta0 + theta1 * x, color='darkorange', linewidth=.5)
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('u')
plt.ylabel('dered_u')

## scikit-learn을 이용한 선형회귀

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X, y)
p = lr.predict(X)

rmse(y, p)

lr.coef_, lr.intercept_

plt.figure(figsize=(8, 8))
plt.scatter(X, y, alpha=.1)
plt.plot(X, p, color='darkorange', linewidth=.5)
plt.xlim(0, 40)
plt.ylim(0, 40)
plt.xlabel('u')
plt.ylabel('dered_u')

## scikit-learn을 이용한 로지스틱회귀

X = trn[trn['class'] != 0].drop('class', axis=1)
y = trn[trn['class'] != 0]['class'].values - 1

np.unique(y)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

lr = LogisticRegression()
lr.fit(X, y)
p = lr.predict_proba(X)[:, 1]

lr.coef_, lr.intercept_

log_loss(y, p)

roc_auc_score(y, p)

from kaggler.metrics import plot_roc_curve, plot_pr_curve

plt.figure(figsize=(16, 16))
plt.subplot(1, 2, 1)
plot_roc_curve(y, p);
plt.title('AUC Curve')
plt.subplot(1, 2, 2)
plot_pr_curve(y, p);
plt.title('Precision-Recall Curve')

