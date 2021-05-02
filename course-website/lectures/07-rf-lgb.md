# 7. Random Forest, GBM

## 7.1 Random Forest 

Random Forest는 2010년 초반까지 가장 많이 사용된 결정트리 기반 앙상블 알고리즘입니다. 앙상블 알고리즘은 기본 모델을 여러개 사용해 예측을 할 때 사용하는 방법입니다. Random Forest는 결정트리 알고리즘을 여러개 활용해 구축한 알고리즘입니다. 

1995년에 벨랩에 Tin Kam Ho 박사가 Random Decision Forests라는 논문을 발표하게 됩니다. 이 당시 발표한 내용은 여러개의 결정트리를 랜덤하게 고른 피쳐로 학습한 후 조합하는 방법이였습니다. 

그리고 2001년에 UC Berkeley의 Leo Breiman 교수가 Tin Kam Ho가 제시한 아이디어에 랜덤 샘플링을 추가해서 Random Forests라는 논문을 발표하게 됩니다. 해당 논문에서는 여러개의 결정트리를 랜덤하게 추출한 피쳐 뿐만 아니라 샘플링도 랜덤하게 해서 학습 후 조합을 실시합니다. 이 방법을 bagging이라고 합니다. 

결정트리의 단점으로 과적합이 쉽게 일어나고 variance가 높은 것으로 알려져 있는데 random forest는 variance를 효과적으로 감소시켜준다는 장점이 있습니다. 

```{note}
머신러닝 모델의 성능 또는 에러에 대해 평가할 때 bias와 variance를 주로 언급합니다. Bias가 높다라는 것은 예측값이 원래 값과 차이가 많이 나는 경우를 의미하며 variance가 높다는 것은 모델을 여러번 학습해서 예측할 때 나온 값의 범위가 높은 경우를 의미합니다. 
```

- 그림 

Random forest를 도식화하면 위와 같습니다. 위와 같이 각각의 결정 트리를 다른 피쳐와 다른 샘플로 학습을 합니다. 예를 들어 Tree 1은 첫번째 부터 천번째 샘플의 두번째, 세번째, 네번째 피쳐로 학습을 시키고 Tree 2는 천번째 부터 2천번째 샘플의 첫번째, 두번째, 네번째 피쳐로 학습하고 Tree 3는 2천번째 부터 3천번째의 첫번째, 두번째, 세번째 피쳐로 학습을 시킵니다. 그리고 나서 각 Tree가 예측한 결과를 평균을 내어 최종 예측값을 산출합니다. 그림 6.X에서 처럼 확률 값을 평균 내어 최종 산출 값으로 결정하는 방법을 soft-voting이라고 합니다. Leo Breiman이 발표한 논문에서는 각각의 Tree가 범주를 예측해서 가장 많이 나온 범주를 최종 예측값으로 결정하는 hard-voting방법을 소개했었습니다. 

```{note}
그림 7.X에서는 이진분류를 예시로 들었지만 multi-class 문제에서도 적용이 가능합니다. 
```

Random forest를 구현하는 코드는 아래와 같습니다. 

- 그림

학습 시 사용하는 주요 인자들에 대한 설망은 아래와 같습니다. 

인자 | 설명 |
---------|----------|
 n_estimators | 사용할 결정트리의 개수 | 
 min_sample_leaf | 가장 마지막 노드에 존재해야 하는 샘플의 최소 개수 | 
 max_features | 결정트리 학습 시 사용할 피쳐 개수, auto일 시 원래 피쳐 개수에 루트(root)를 씌운 값 사용 | 
 max_samples | 샘플할 데이터의 개수 | 
 random_state | 재현성을 위한 시드값 | 
 n_jobs | 학습시 사용할 쓰레드의 개수, -1일 경우 사용가능한 모든 쓰레드 사용해 학습 | 
- 표 7.X

주요 인자를 설정해서 Random forest 객체를 생성한 뒤 학습과 예측하는 과정은 scikit-learn에서 제공하는 다른 알고리즘과 마찬가지로 `fit()`과 `predict()`을 사용합니다. 실습 파일을 통해 random forest를 학습을 실습해보시길 바랍니다. 

결정트리 모델은 하나의 모델만 존재하기 때문에 시각화를 통해 어떤 피쳐가 중요하게 작용했는지 확인 가능합니다. 하지만 random forest나 gradient boosting model 같은 경우 여러개의 결정트리를 사용하기 때문에 모든 결정트리를 시각화해 확인하기 제한됩니다. 그렇기 때문에 변수별로 각 결정트리의 손실함수를 최소화 하는데 얼마나 중요하게 작용했는지를 계산해서 시각화 하는 방법이 주로 사용됩니다. 

## 7.2 Gradient Boosting Machine (GBM)

Gradient Boosting Machine은 정형 데이터에서 가장 좋은 성능을 내는 결정트리 기반 앙상블 알고리즘입니다.

```{note}
정형 데이터에서는 GBM 알고리즘이 우세를 보이며 비정형 데이터에는 딥러닝 알고리즘이 우세를 보입니다.
```

GBM은 random forest보다 일찍 등장했습니다. 마찬가지로 ramdom forest를 제안한 UC Berkeley의 Leo Breiman 교수가 1997년에 boosting을 이용한 손실함수 최적화 아이디어를 제안했습니다. 해당 논문에 기반해서 1999년에 Stanford의 Jerome Friedman 교수가 Gradient Boosting Machine 알고리즘을 발표합니다. 

GBM은 결정트리를 순차적으로 학습을 합니다. 첫번째 결정트리를 학습시키고, 해당 트리의 오차를 줄이는 방향으로 두번째 결정트리를 학습시킵니다. 이런식으로 N번째 트리까지 학습시키는 방법이 boosting 학습방법입니다. 이러한 boosting 방법은 개별 결정트리의 bias를 효과적으로 감소시켜 성능을 향상시킵니다.

```{note}
Element of Statistical Learning 10장에 이론적인 내용이 상세히 서술돼있습니다. 
```

- 그림

위 그림은 GBM의 학습 과정을 도식화한 것입니다. 첫번째 결정트리의 오차인 `r1`이 두번째 트리의 종속변수가 되며, 두번째 트리의 오차인 `r2`가 세번째 트리의 종속변수가 되는 방식으로 학습합니다. 

## 7.3 XGBoost vs LightGBM

GBM을 구현할 때는 XGBoost 또는 LightGBM 라이브러리를 주로 활용합니다. XGBoost는 현재 Carnegie Mellon University에 있는 Tianqi Chen 교수가 2011년에 박사학위 과정 때 만든 라이브러리 입니다. LightGBM은 Microsoft Asia에 있는 팀이 개발한 라이브러리 입니다. LightGBM은 2017년에 발표가 되면서 빠르게 확산이 된 라이브러리입니다. 

해당 라이브러리 들은 출시가 될 당시 다른 구현 방법에 비해 빨랐기 때문에 주목을 받았습니다. XGBoost는 출시될 당시 scikit-learn에서 제공하는 GBM과 R에서 제공하는 gbm 라이브러리에 비해 적게는 2~3배, 많게는 10배까지 빨랐습니다. 단순히 빠를 뿐만 아니라 성능도 다른 GBM 라이브러리 보다 좋았기 때문에 많은 호응을 받았습니다. 2017년에 LightGBM이 등장하면서 XGBoost보다 빠르게 작동했고 성능도 조금 더 높았습니다. 그래서 2017년도 부터는 캐글의 상위 랭커들이 LightGBM을 많이 사용하기도 했습니다. 

2020년에 실시한 설문조사에 의하면 XGBoost를 가장 많이 사용하며 그 다음으로 LightGBM 라이브러리를 많이 사용합니다. 또한 벤치마킹 테스트 결과를 확인해보면 cpu를 사용할 때 데이터의 개수가 10만개 정도일 때는 xgboost가 빠르지만 데이터 개수가 1백만개가 넘어가면 lightgbm가 더 빠른것을 확인할 수 있습니다. 속도 뿐만 아니라 lightgbm이 auc 성능 또한 가장 높은 것을 확인할 수 있습니다. 

- 그림

- 그림

- 그림

XGBoost 라이브러리가 사용은 많이 되지만 대회에서는 LightGBM이 성능이 잘 나오는 경우가 많으므로 이번 장에서는 LightGBM 실습을 진행해보도록 하겠습니다. 

LightGBM 구현 코드는 아래와 같습니다. 

- 그림

LightGBM의 주요 인자는 아래와 같습니다. 

인자 | 설명 |
---------|----------|
 objective | 학습의 목적 (binary, multiclass, 등) | 
 n_estimators | 사용할 결정트리의 개수 | 
 num_leaves | 사용할 리프 노드의 개수 | 
 learning_rate | boosting 학습시 기존 에러에 적용할 가중치 | 
 min_child_sample | 리프 노드에 존재해야 하는 최소한의 샘플 수 | 
 subsample | 결정트리 학습 시 사용할 샘플의 개수/비율 | 
 subsample_freq | subsampling을 진행할 단위 횟수 | 
 colsample_bytree | 결정트리 학습 시 사용할 피쳐의 개수/비율 | 
 random_state | 재현성을 위한 시드값 | 
 n_jobs | 학습시 사용할 쓰레드의 개수, -1일 경우 사용가능한 모든 쓰레드 사용해 학습 |
- 표 7.X

LightGBM에서 제공하는 `fit()`함수에는 학습데이터의 종속변수와 독립변수 뿐만 아니라 검증데이터의 종속변수와 독립변수 또한 입력할 수 있는 `eval_set`인자가 존재합니다. `eval_set`인자와 더불어 `eval_metric`을 명시하면 학습 도중에 검증데이터셋에 대한 평가 지표를 같이 보여줍니다. 또한 `early_stopping_rounds`인자는 특정 조건이 만족하면 학습을 조기 중단 시켜주는 역할을 합니다. 예를 들어 10이라고 입력을 하면 10번째마다 `eval_metric`이 향상되는지 확인을 합니다. 향상이 안된다면 학습을 조기 종료합니다. `n_estimators`에 신경쓸 필요 없이 `early_stopping_rounds`에 의해서 학습이 조기 종료되기 때문에 상당히 유용한 인자입니다. 

```{tip}
learning_rate에 따라 early_stopping_rounds 조정이 가능합니다. 0.01 정도의 낮은 learning_rate인 경우 eval_metric이 수렴 시 monotonic하게 줄어들 기 때문에 early_stopping_rounds는 10정도로 설정해도 무관합니다. 

early_stopping_rounds를 크게 주는 경우는 learning_rate값이 높을 때 입니다. learning_rate가 높으면 eval_metric가 진동(oscillate)을 하며 수렴하기 때문에 early_stopping_rounds를 넉넉하게 줄 필요가 있습니다. 
```

다음 실습 단계에서 random forest와 lightgbm 알고리즘을 직접 실행해보면서 익혀보시길 바랍니다. 

## 7.4 참고자료

- [Scikit-learn 튜토리얼 페이지](https://scikit-learn.org/stable/tutorial/)
    - [Random Forests](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
    - [Gradient Boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting)
- [LightGBM 홈페이지](https://lightgbm.readthedocs.io/en/latest/)
    - [LightGBM Parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
    - [LightGBM Scikit-Learn API](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier)
- [XGBoost 홈페이지](https://xgboost.readthedocs.io/en/latest/index.html)
- ESL 2판 ([pdf](https://web.stanford.edu/~hastie/ElemStatLearn/printings/ESLII_print12_toc.pdf))
    - Chapter 10: Boosting and Additive Trees
    - Chapter 15: Random Forest

Random forest와 GBM의 이론적인 배경이 궁금하시다면 The Elements of Statistical Learning (ESL)의 10장과 15장을 참고하시길 바랍니다. 