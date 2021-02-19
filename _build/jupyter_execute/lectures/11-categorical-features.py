# 범주형변수 가공 데모

## 라이브러리 import 및 설정

%reload_ext autoreload
%autoreload 2
%matplotlib inline

import kaggler
from lightgbm import LGBMRegressor
from matplotlib import rcParams, pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import hstack
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from warnings import simplefilter

rcParams['figure.figsize'] = (16, 8)
plt.style.use('fivethirtyeight')
simplefilter('ignore')

## 학습데이터 로드

학습데이터는 데이콘 [영화 관객수 예측 모델 개발](https://dacon.io/competitions/open/235536/data/) 페이지에서 다운로드하여 `../data/movies/` 폴더에 저장해 둔다.

이 데이터는 영화의 장르, 개봉일, 상영시간 등과 같은 데이터로 영화 총 관객수 (`box_off_num`)를 예측하는 데이터이다.

data_dir = Path('../data/movies/')
trn_file = data_dir / 'movies_train.csv'
seed = 42
target_col = 'box_off_num'

df = pd.read_csv(trn_file, index_col=0)
print(df.shape)
df.head()

## EDA (Exploratory Data Analysis)

df.info()

* 독립변수를 포함 총 6개의 수치형변수가 있다. 그 중 `dir_prev_bfnum`은 결측값이 많음을 확인할 수 있다.
* `distributor`, `genre`, `screening_rat`, `director`의 총 4개의 범주형변수가 있다.
* `release_time`은 시계열변수이지만 문자열 (`object`)으로 인식하고 있다.

df['release_time'] = pd.to_datetime(df['release_time'])
print(df['release_time'].dtype)

df.fillna(0, inplace=True)
df.info()

`release_time`을 시계열타입 (`datetime`)으로 변환하고 결측값을 0으로 대체하였다.

num_cols = [x for x in df.columns if df[x].dtype in [np.int64, np.float64] and x != target_col]
cat_cols = ['distributor', 'genre', 'screening_rat', 'director']
print(f'    numeric ({len(num_cols)}):\t{num_cols}')
print(f'categorical ({len(cat_cols)}):\t{cat_cols}')

### 범주형변수 EDA

print(cat_cols)

pd.DataFrame(df['distributor'].value_counts())

pd.DataFrame(df['genre'].value_counts())

pd.DataFrame(df['screening_rat'].value_counts())

pd.DataFrame(df['director'].value_counts())

## 수치형/시계열변수 가공

수치형 독립변수 중 멱변환 분포를 따르는 변수에도 `np.log1p()` 변환을 적용하였다.

df[['dir_prev_bfnum', 'dir_prev_num', 'num_staff', 'num_actor']] = df[['dir_prev_bfnum', 'dir_prev_num', 'num_staff', 'num_actor']].apply(np.log1p)
df[num_cols].describe()

df['year'] = df['release_time'].dt.year
df['month'] = df['release_time'].dt.month
df.head()

num_cols += ['year', 'month']
print(num_cols)

features = num_cols + cat_cols
print(features)

## 범주형변수 가공

rmse = lambda y, p: np.sqrt(mean_squared_error(y, p))
rmsle = lambda y, p: np.sqrt(mean_squared_error(np.log1p(y), np.log1p(p)))

### Ordinal Encoding

from sklearn.preprocessing import OrdinalEncoder
df_cat = df.copy()
oe = OrdinalEncoder()
df_cat[cat_cols] = oe.fit_transform(df[cat_cols])
df_cat[cat_cols].head()

trn, tst = train_test_split(df_cat, test_size=.2, random_state=seed)
clf = LGBMRegressor(random_state=seed)
clf.fit(trn[features], np.log1p(trn[target_col]))
p = np.expm1(clf.predict(tst[features]))
print(f' RMSE:\t{rmse(tst[target_col], p):12.2f}')
print(f'RMSLE:\t{rmsle(tst[target_col], p):12.2f}')

### Label Encoding with Grouping

from kaggler.preprocessing import LabelEncoder
df_cat = df.copy()
le = LabelEncoder(min_obs=2)
df_cat[cat_cols] = le.fit_transform(df[cat_cols])
df_cat[cat_cols].head()

trn, tst = train_test_split(df_cat, test_size=.2, random_state=seed)
clf = LGBMRegressor(random_state=seed)
clf.fit(trn[features], np.log1p(trn[target_col]))
p = np.expm1(clf.predict(tst[features]))
print(f' RMSE:\t{rmse(tst[target_col], p):12.2f}')
print(f'RMSLE:\t{rmsle(tst[target_col], p):12.2f}')

### One-Hot-Encoding

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
X = hstack((df[num_cols],
            ohe.fit_transform(df[cat_cols])))
print(X.shape)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, df[target_col], test_size=.2, random_state=seed)
clf = LGBMRegressor(random_state=seed)
clf.fit(X_trn, np.log1p(y_trn))
p = np.expm1(clf.predict(X_tst))
print(f' RMSE:\t{rmse(tst[target_col], p):12.2f}')
print(f'RMSLE:\t{rmsle(tst[target_col], p):12.2f}')

### One-Hot-Encoding with Grouping

from kaggler.preprocessing import OneHotEncoder
ohe = OneHotEncoder(min_obs=2)
X = hstack((df[num_cols],
            ohe.fit_transform(df[cat_cols])))
print(X.shape)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, df[target_col], test_size=.2, random_state=seed)
clf = LGBMRegressor(random_state=seed)
clf.fit(X_trn, np.log1p(y_trn))
p = np.expm1(clf.predict(X_tst))
print(f' RMSE:\t{rmse(tst[target_col], p):12.2f}')
print(f'RMSLE:\t{rmsle(tst[target_col], p):12.2f}')

### Target Encoding without Cross-Validation

from kaggler.preprocessing import TargetEncoder
trn, tst = train_test_split(df, test_size=.2, random_state=seed)
te = TargetEncoder(cv=None)
trn[cat_cols] = te.fit_transform(trn[cat_cols], trn[target_col])
tst[cat_cols] = te.transform(tst[cat_cols])
trn[cat_cols].head()

clf = LGBMRegressor(random_state=seed)
clf.fit(trn[features], np.log1p(trn[target_col]))
p = np.expm1(clf.predict(tst[features]))
print(f' RMSE:\t{rmse(tst[target_col], p):12.2f}')
print(f'RMSLE:\t{rmsle(tst[target_col], p):12.2f}')

### Target Encoding with Cross-Validation

trn, tst = train_test_split(df, test_size=.2, random_state=seed)
te = TargetEncoder()
trn[cat_cols] = te.fit_transform(trn[cat_cols], trn[target_col])
tst[cat_cols] = te.transform(tst[cat_cols])
trn[cat_cols].head()

clf = LGBMRegressor(random_state=seed)
clf.fit(trn[features], np.log1p(trn[target_col]))
p = np.expm1(clf.predict(tst[features]))
print(f' RMSE:\t{rmse(tst[target_col], p):12.2f}')
print(f'RMSLE:\t{rmsle(tst[target_col], p):12.2f}')

### Frequency Encoding

from kaggler.preprocessing import FrequencyEncoder
df_cat = df.copy()
fe = FrequencyEncoder()
df_cat[cat_cols] = fe.fit_transform(df[cat_cols])
df_cat[cat_cols].head()

trn, tst = train_test_split(df_cat, test_size=.2, random_state=seed)
clf = LGBMRegressor(random_state=seed)
clf.fit(trn[features], np.log1p(trn[target_col]))
p = np.expm1(clf.predict(tst[features]))
print(f' RMSE:\t{rmse(tst[target_col], p):12.2f}')
print(f'RMSLE:\t{rmsle(tst[target_col], p):12.2f}')

### Hash Encoding

from sklearn.feature_extraction import FeatureHasher
fh = FeatureHasher(n_features=128, input_type='string')
X = hstack([df[num_cols]] + [fh.fit_transform(df[col]) for col in cat_cols])
print(X.shape)

X_trn, X_tst, y_trn, y_tst = train_test_split(X, df[target_col], test_size=.2, random_state=seed)
clf = LGBMRegressor(random_state=seed)
clf.fit(X_trn, np.log1p(y_trn))
p = np.expm1(clf.predict(X_tst))
print(f' RMSE:\t{rmse(tst[target_col], p):12.2f}')
print(f'RMSLE:\t{rmsle(tst[target_col], p):12.2f}')

