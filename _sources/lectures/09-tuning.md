# 9. 모델 튜닝

이번 장에서는 모델 튜닝 또는 하이퍼파라미터 최적화를 진행할 때 사용가능한 라이브러리에 대해 알아보겠습니다. 

- 그림 9-X

위 그림은 모델 튜닝 싸이클을 나타내고 있습니다. 먼저 초기 파라미터를 설정 후 모델을 학습하고 검증합니다. 검증 결과에 따라 파라미터를 변경하고 다시 학습 및 검증을 해서 최선의 결과에 도달할 때 까지 반복을 합니다. 검증을 할 때는 8장에서 확인했듯이 Hold-out Validation 또는 N-Fold Cross-Validation을 사용합니다. 

최선의 결과를 찾기 위해선 검증결과를 확인하고 파라미터를 변경하는 과정을 반복해야 합니다. 선형회귀 알고리즘 처럼 파라미터 개수가 적고 이진값으로 인자를 받는 경우 직접 파라미터를 변경해볼 수 있습니다. 

- 그림 9-X Scikit-learn 선형 회귀 알고리즘 파라미터

하지만 lightgbm 처럼 파라미터가 많고 정수형 또는 실수형을 인자로 받게되면 직접 파라미터를 변경하면서 검증하기가 어렵습니다. 이런 경우 하이퍼파라미터 최적화를 용이하게 해주는 도구 활용이 필요합니다. 

- 그림 9-X LightGBM 알고리즘 파라미터

Scikit-learn에서 대표적으로 제공하는 하이퍼 파리미터 최적화 툴로 GridSearchCV와 RandomizedSearchCV가 있습니다. GridSearch는 검색하고자 하는 피처 값들의 모든 조합을 사용하여 모델을 학습/겁증 후 최고의 모델을 선택합니다. 아래 에시 코드에서 처럼 `param_grid`에 탐색하고자 하는 조합을 모두 입력을 해서 검증을 실시합니다. 

- 그림

RandomizedSearch는 검색하고자하는 피쳐 값들의 조합 중 N개의 조합을 임의로 선택해서 모델을 학습/검증 후 최고의 모델을 선택합니다. 이 때 피쳐 값은 범위 또는 분포로도 지정할 수 있습니다. 아래 그림처럼 `l1_ratio`와 `alpha`를 각각 균등분포와 로그균등분포를 적용하면 해당 분포로 부터 임의의 값을 추출해서 모델을 학습하고 검증을 합니다. 

- 그림

Scikit-learn 공식 튜토리얼 페이지에 추가 설명을 확인할 수 있습니다. 

앞서 살펴본 GridSearch나 RandomizedSearch는 가능한 모든 조합을 사용하거나 그 중 일부를 랜덤하게 사용합니다. 다음 절에서 살펴볼 Hyperopt와 Optuna는 베이지안 방식을 활용해 사용하고자 하는 하이퍼파라미터 조합을 탐색합니다. 

## 9.1 Hyperopt

Hyperopt 라이브러리는 가장 널리 사용되는 모델 튜닝 라이브러리 중 하나입니다. 2013년에 Harvard에 James Bergstra와 MIT의 Dan Yamins가 Hyperopt를 [ICML](http://proceedings.mlr.press/v28/bergstra13.pdf)학회와 [SciPy](https://conference.scipy.org/proceedings/scipy2013/pdfs/bergstra_hyperopt.pdf)에서 발표를 했습니다. Hyperopt는 Tree-structured Parzen Estimator(TPE) 알고리즘을 사용해 베이지안 최적화를 수행합니다. 출시 당시에는 scipy 라이브러리를 활용한 최적화 작업을 주로 사용했었는데 Hyperopt가 scipy에서 제공하는 `scipy.optimize.minimize()` API와 사용 방법이 유사해서 많은 관심을 끌었습니다. 

실습 파일에서 Hyperopt를 적용해 lightGBM의 하이퍼파라미터를 최적화해보겠습니다. 

```{tip}
데이터가 많을 경우 하이퍼파라미터 최적화 시에 N-Fold CV를 사용하면 시간이 오래 걸립니다. Hold-out Validation이나 전체 데이터셋에서 샘플링을 한 데이터를 사용해서 최적화 하면 탐색 시간을 줄일 수 있습니다. 
```

```{tip}
모델 튜닝은 마지막 단계에서 성능을 끌어올릴 때 사용하시길 바랍니다. 무작정 좋은 하이퍼파라미터를 찾을려고 하는 것 보다 일반적으로 적합한 피쳐를 생성하는 것이 점수 상승에 효과적입니다. 
```

## 9.2 Optuna

Optuna는 2019년 일본의 Preferred Networks사의 연구진들이 [KDD](https://arxiv.org/pdf/1907.10902.pdf)에서 발표를 한 라이브러리입니다. 최근에 각광을 받고 있는 모델 튜닝 라이브러리입니다. Optuna는 기존 하이퍼파라미터 최적화 라이브러리들의 장점들을 결합했으며 단순한 API를 제공합니다. 

Optuna의 `integration`모듈 내에 있는 알고리즘들은 일반적으로 탐색 범위로 사용하는 하이퍼파라미터들을 사용해 최적화를 실시합니다. 그래서 사용자가 직접 범위를 명시하지 않아도 내부적으로 모델 튜닝을 실시합니다. 다만 세부적인 사항은 사용자가 지정할 수 없다는 단점이 존재합니다.  

```{tip}
처음 모델 튜닝을 실시해서 적정한 탐색 범위 설정이 어렵다면 optuna의 integration모듈 사용을 추천드립니다. 적절한 범위를 직접 명시해 빠른 시간내 탐색을 마칠려면 hyperopt 또는 optuna에서 제공하는 다른 API를 활용해 직접 범위를 설정해서 모델 튜닝을 하시길 바랍니다. 
```

## 9.3 참고자료

- [Scikit-learn Hyperparameter Tuning 튜토리얼](https://scikit-learn.org/stable/modules/grid_search.html)
- [Hyperopt 웹사이트](http://hyperopt.github.io/hyperopt/)
- [Optuna 웹사이트](https://optuna.org/)
- [머신러닝 마스터 클래스](https://www.upaper.net/jeongyoonlee/1136706)
    - 9장. 하이퍼파라미터 최적화
