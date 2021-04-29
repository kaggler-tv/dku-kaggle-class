# 8. Cross-Validation, Stacking

## 8.1 Cross-Validation

구축한 모델의 성능 평가시 사용할 수 있는 주요 방법은 4가지가 있습니다. 각 방법에 대한 설명은 아래와 같습니다. 

방법 | 설명 |
---------|----------|
 Hold-out Validation | 학습데이터의 일부를 검증셋으로 분류 후 나머지만 학습에 사용하고 검증셋은 모델 성능 검증에 사용 | 
 N-Fold CV | 학습데이터를 N개의 폴드로 나눈 후 각각의 폴드에 대해 한 폴드를 검증셋으로 나머지는 학습셋으로 사용해 총 N번 학습하고 N개의 검증셋을 모아 모델 성능 검증에 사용 | 
 Stratified N-Fold CV | N-Fold CV에서 각각의 폴드에 종속변수의 분포가 동일하게 폴드를 나누는 방식, 분류학습에서 종속변수의 범주의 분포가 균일하지 않을 때 사용 | 
 Leave-One-Out (LOO) CV | 샘플의 개수를 N으로 사용한 N-Fold CV. 샘플의 개수가 적을 때 (50 미만) 사용 | 
 - 표 8.X 

Hold-out Validation은 데이터가 아주 많은 경우에는 충분합니다. 하지만 데이터과학 대회에서는 조금이라도 성능을 높이기 위해 N-Fold CV를 사용해 모든 학습데이터를 학습과 검증에 사용하는 것을 선호합니다. 

Hold-out Validation을 도식화한 그림과 코드 예제는 아래와 같습니다. 

- 그림

- 그림

Scikit-learn에서 `train_test_split()`함수를 통해 적용 가능하며 해당 함수는 나누고자 하는 독립변수와 종속변수, 그리고 검증셋의 크기를 나타내는 `test_size`와 재현성을 위한 `random_state`를 인자로 받습니다. 

N-Fold CV를 도식화한 그림과 코드 예제는 아래와 같습니다. N-Fold CV는 K-Fold CV라고도 불립니다. 

- 그림

- 그림

Scikit-learn에서 `KFold()`함수를 통해 적용 가능합니다. `KFold()`는 나누고자 하는 폴드의 개수인 `n_splits`를 인자로 명시해주어야 합니다. `shuffle`인자는 데이터 분리 시 섞을지 여부를 결정하며 `random_state`는 데이터를 섞을 시 재현성을 고정시키는 인자입니다. KFold 객체를 생성한 뒤 for문을 활용해 KFold 객체가 생성하는 인덱스를 활용해 학습데이터셋과 검증데이터셋으로 구분하여 모델을 학습하고 검증합니다. 

KFold를 활용해 각각의 검증셋에 대해 예측한 값을 cross-validation prediction 또는 out-of-fold prediction이라고 하는데 해당 값을 stacking시 활용하게 됩니다. 다음 절에서 관련 내용을 자세히 살펴보겠습니다. 

이번 실습에서는 lightgbm, logistic regression, 그리고 random forest 각각에 대해 cross-validation을 적용한 파일이 3개가 있습니다. 실습을 통해 cross-validaion 개념을 익혀보시길 바랍니다. 

## 8.2 Stacking

Stacking은 여러가지 모델을 조합해서 하나의 예측값을 얻는 앙상블 방식 중의 하나입니다. 또한 캐글에서 주로 사용되는 앙상블 기법 이기도 합니다. 

Stacking은 1992년에 Los Alamos National Laboratory의 David H. Wolpert가 Stacked Generalization라는 논문으로 발표를 했습니다. 주용 내용은 여러개 모델의 예측값을 입력으로 사용해 다른 모델을 학습하는데, 이 때 N-Fold CV를 함께 사용해 과적합을 피하는 내용입니다. 

데이터가 방대한 경우 Hold-out Validation과 함께 사용 가능하며 이런 경우 Blending이라고도 표현합니다. Blending 방식은 Netflix Grand Prize 대회에서 우승자들이 사용한 방법이기도 합니다. 

- 그림

위 그림은 stacking을 도식화한 그림입니다. 위 예시에서는 5-Fold CV를 통해 학습데이터셋에 대한 CV Prediction을 산출합니다. 만약 random forest, lightgbm, 그리고 logistic regression으로 5-Fold CV를 3번 진행한다면 각 모델 별로 하나씩 CV Prediction이 산출되어 총 3개의 CV Prediction값을 갖게 됩니다. 이렇게 얻은 3개의 CV Prediction을 독립변수로 사용해서 다음 단계인 Stage 1 Ensemble에서 모델을 학습합니다. Stage 1 단계에서도 N-Fold CV를 활용해 CV Prediction을 얻게 됩니다. 

시험데이터셋에 대해서는 초기에 사용한 random forest, lightgbm, 그리고 logistic regression을 활용해 얻은 예측값을 Stage 1 Ensemble에서 학습한 모델의 입력값으로 넣어 최종 예측을 하게 됩니다. 

- 그림

위 그림은 stacking을 적용한 예시입니다. Single model을 생성하는 단계에서 총 64개의 모델을 구축합니다. 이 때 모델은 GBM, NN, FM, LR등 다양하게 사용한 것을 볼 수 있습니다. 각각의 single model에 예측한 총 64개의 예측값을 Stage 1 Ensemble의 입력값으로 활용합니다. Stage 1에서는 총 15개의 모델이 구축됐으며 마찬가지로 15개의 예측값이 생성됩니다. 15개의 예측값을 Stage 2의 입력값으로 주어 2개의 모델을 구축합니다. Stage 3에서는 이전 단계에서 활용한 모든 모델의 예측값을 입력값으로 주어 최종 모델 1개를 학습합니다. 

위와 같이 계속해서 stage를 쌓아갈 수 있습니다. 하지만 성능 향상에 가장 많은 도움이 되는 것은 Stage 1 Ensemble입니다. Stage가 쌓여갈 수록 성능은 점점 더 적게 향상됩니다. 

```{tip}
일반적으로 데이터과학 대회에서는 stacking을 stage 1 또는 stage 2까지만 사용합니다.
```

모델 성능 향상을 이루고자 할 때 일반적으로 stacking은 가장 마지막에 고려하는 기법입니다. 가장 먼저 고려해야 할 것은 새로운 피쳐(독립변수)를 구축하는 것이며 그 다음으로 개별 모델의 하이퍼파라미터 튜닝을 고려합니다. 마지막 단계에서 stacking을 통해 모델의 성능을 끌어올리는 것이 일반적으로 사용하는 방법입니다. 

```{note}
동계 스포츠인 쇼트트랙에서 결승선에 들어올 때 앞다리를 내밀어 조금이나마 기록을 향상시키는데, stacking도 이와 같다고 볼 수 있습니다. 성능을 극단적으로 향상시키기 위해 사용되기 보다는 마지막 단계에서 최종 순위를 다툴 때 사용하는 방법입니다. 
```

Stacking 실습 파일은 총 2개가 있으며 하나는 lightgbm을 활용한 예제이고 하나는 logistic regression을 활용한 예제입니다. 3개의 N-Fold CV 실습 파일에서 생성한 예측값을 불러와서 stacking을 진행하도록 설계돼 있습니다. 

## 8.3 참고자료

- [Scikit-learn Cross-Validation 튜토리얼](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Kaggle Ensembling Guide](http://mlwave.com/kaggle-ensembling-guide/)
- [머신러닝 마스터 클래스](https://www.upaper.net/jeongyoonlee/1136706)
    - 12장. 앙상블과 스태킹
