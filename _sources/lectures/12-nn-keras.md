# 12. 신경망, Keras

이번 장에서는 신경망(neural network) 모델과 신경망 모델 구축시 사용하는 Keras 프레임워크에 대해 살펴보겠습니다. 

## 12.1 로지스틱회귀

신경망 모델에 대해 설명하기 전에 [4장](04-numpy.md)에서 배운 로지스틱회귀에 대해 다시 살펴보겠습니다. 로지스틱회귀는 종속변수가 범주형변수일 때 사용하는 알고리즘입니다. 종속변수와 독립변수의 관계를 선형 수식으로 나타낸 뒤 로지스틱 함수를 적용해서 0과 1 사이의 값으로 변환합니다. 이를 도식화 하면 아래 그림 처럼 나타낼 수 있습니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch12-img01.jpg?raw=true)
- 그림 12-1 로지스틱회귀 모형의 도식화([출처](https://docs.google.com/presentation/d/1D7_8FNJjpzStB-T63vv2aY17YyJJr4k5iqStfZJ2m3k/edit?usp=sharing))

`x1`, `x2`, `x3` 값을 각각 `theta1`, `theta2`, `theta3`에 곱한 후 모두 더한 뒤 `g`함수에 입력하는 것으로 도식화 할 수 있습니다. 로지스틱회귀에서는 `g`함수가 로지스틱함수가 되겠습니다. 

## 12.2 신경망

신경망은 그림 12-1에 있는 입력충(input layer)과 출력층(output layer) 사이에 은닉층(hidden layer)이 추가된 것입니다. 그래서 은닉층이 하나도 없으면 선형회귀모델 또는 로지스틱회귀 모델이 되는 것이며 은닉층이 1개라도 존재한다면 신경망 모델이 됩니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch12-img02.jpg?raw=true)
- 그림 12-2 신경망 모델([출처](https://docs.google.com/presentation/d/1D7_8FNJjpzStB-T63vv2aY17YyJJr4k5iqStfZJ2m3k/edit?usp=sharing))

은닉층에는 여러개의 뉴런이 포함됩니다. 또한 은닉층 별로 다른 개수의 뉴런을 가질 수 있습니다. 각 뉴런에서는 출력 값에 특정 함수를 적용해서 다음 층으로 보내는데 이를 활성화 함수(activation function)라고 합니다. 주로 사용되는 활성화 함수로는 sigmoind, tanh, relu가 있습니다. 

신경망 모델이 예측하는 방식은 먼저 입력층에 입력값이 들어오면 각각의 입력값에 가중치(weight)를 곱해서 다음 은닉층에 보냅니다. 그리고 은닉층은 입력 받은 값에 활성화 함수를 적용한 값을 산출합니다. 이 과정을 출력층에 이르기 까지 반복을 해서 최종 출력값을 산출합니다. 

신경망 모델이 학습하는 방법은 경사하강법과 chain rule을 적용한 backpropagation을 통해 가중치를 학습합니다. 

이번 장에서는 신경망 모델을 사용하기 위한 기본적인 개념 위주로 학습할 예정입니다. 

## 12.3 신경망 - 특수 레이어

앞서 설명 드린 은닉층 외에도 특수한 레이어들이 존재합니다. 아래 표에 정리돼있습니다.

레이어 | 설명 
---------|----------
 Dropout | 학습시 랜덤하게 일정 비율의 뉴런을 drop 
 Batch/Layer Normalization | 레이어의 입력값을 표준화 
 Max/Average Pooling | 입력값 중 Max/Average만 선택 
 Embedding | Label/Ordinal 인코딩 된 정수를 실수 벡터로 변화나 
 Convolution | 1D/2D Convolution 
 Recurrent | LSTM, GRU 
- 표 12-1 특수 레이어

## 12.4 Keras

Keras는 주요 오픈소스 신경망 프레임워크 중 하나입니다. 구글의 Francois Chollet이 2015년에 발표했습니다. Keras는 Tensorflow/Theano/CNTK/PlaidML의 high-level API를 제공하지만 Tensorflow를 backend로 가장 많이 사용합니다. Keras 외의 주요 프레임워크는 아래 그림에 나와 있습니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch12-img03.jpg?raw=true)
- 그림 12-3 주요 신경망 프레임워크([출처](https://docs.google.com/presentation/d/1D7_8FNJjpzStB-T63vv2aY17YyJJr4k5iqStfZJ2m3k/edit?usp=sharing))

PyTorch와 Tensorflow는 각각 Facebook과 Google에서 개발을 주도하는 프레임워크 이며 PyTorch Lightning과 Keras는 각각 PyTorch와 Tensorflow를 더 사용하기 쉽게 만들어준 프레임워크입니다. Fast.ai 또한 PyTorch를 쉽게 사용할 수 있게 만들어준 라이브러리입니다. Mxnet은 XGBoost를 개발한 Tianqi Chen이 개발했으며 지금은 Amazon에서 후원을 하고 있는 프로젝트입니다. 

```{tip} 
이미지, 음성, 자연어 등의 비정형 데이터를 사용한 데이터과학 대회에서는 신경망 알고리즘이 성능의 우세를 보이며 정형데이터를 사용한 대회에서는 트리 기반의 lightgbm 또는 xgboost가 우세를 보입니다. 
```

### 12.4.1 Keras - Sequential API

Keras는 sequential API와 functional API를 제공합니다. Sequential API는 쉽게 사용할 수 있으며 복잡한 모델 구축을 하고자 할 때는 functional api를 주로 사용합니다. 

Sequential API의 특징은 입력층에서 출력층까지 차례로 레이어를 추가합니다. 그림 12-4처럼 `add()`를 통해 하나씩 레이어를 쌓아갑니다. `compile()`단계에서는 손실함수와 학습시 사용할 `optimizer`를 사용합니다. 모델 학습시에는 `fit()`함수를 사용합니다. 이 때 종속변수가 multi-class일 경우 one-hot encoding을 한 후에 `fit()`에 입력합니다. Keras에서 제공하는 `to_categorical()`함수를 통해 종속변수에 one-hot encoding 적용이 가능합니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch12-img04.jpg?raw=true)
- 그림 12-4 Sequential API 사용 예제([출처](https://docs.google.com/presentation/d/1D7_8FNJjpzStB-T63vv2aY17YyJJr4k5iqStfZJ2m3k/edit?usp=sharing))

자세한 코드는 [실습 파일](12-nn-cv.ipynb)에서 확인할 수 있습니다. 

### 12.4.2 Keras - Functional API

앞서 sequential API는 순차적으로만 레이어 추가가 가능합니다. 하지만 때에 따라선 아래와 같이 레이어를 복잡하고 다양하게 구성할 수도 있습니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch12-img05.jpg?raw=true)
- 그림 12-5 복잡한 신경망 모델([출처](https://docs.google.com/presentation/d/1D7_8FNJjpzStB-T63vv2aY17YyJJr4k5iqStfZJ2m3k/edit?usp=sharing))

위와 같은 복잡한 신경망 모델 구축을 위해선 functional API를 통해 구현 가능합니다. 이번 예시에서는 12.4.1절에서 구축한 신경망 모델을 functional API에서는 어떻게 구축하는지 살펴보겠습니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch12-img06.jpg?raw=true)
- 그림 12-6 Functional API 사용 예제([출처](https://docs.google.com/presentation/d/1D7_8FNJjpzStB-T63vv2aY17YyJJr4k5iqStfZJ2m3k/edit?usp=sharing))

Functional API에서는 Input레이어를 직접 명시를 해줘야 합니다. 그리고 나서 은닉층 `x`는 입력층인 `inputs`를 입력 받도록 설계합니다. 마찬가지로 출력층 `outputs`도 `x`를 입력 받도록 정의합니다. 그리고 나서 `compile()`을 적용하고 `fit()`을 통해 학습하는 것은 sequential API와 같습니다. 

자세한 코드는 [실습 파일](12-nn-cv.ipynb)에서 확인할 수 있습니다. 

### 12.4.3 Keras - Callbacks

Callbacks은 `fit()`함수에 추가할 수 있는 기능입니다. Keras 사용시 유용한 callbacks은 아래와 같습니다. 

레이어 | 설명 
---------|----------
 EarlyStopping | 검증셋 성능이 향상되지 않으면 학습 중단 
 ReduceLROnPlateau | 검증셋 성능이 향상되지 않으면 learning rate 감서 
 LearningRateScheduler | Learning rate을 커스텀하게 변경 
 ModelCheckpoint | 모델 저장 
 TensorBoard | 학습 과정 시각화 
- 표 12-2 주요 callbacks

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch12-img07.jpg?raw=true)
- 그림 12-7 Callbacks 사용 예제([출처](https://docs.google.com/presentation/d/1D7_8FNJjpzStB-T63vv2aY17YyJJr4k5iqStfZJ2m3k/edit?usp=sharing))

그림 12-7과 같이 `es`와 `rlr`에 각각 EarlyStopping과 ReduceLROnPlateau callback을 저장하고 `fit()`의 `callbacks` 파라미터에 전달함으로써 구현 가능합니다. 

## 12.5 참고자료

- [Stanford CS229 Lecture 8: Neural Networks](http://cs229.stanford.edu/notes2020fall/notes2020fall/deep_learning_notes.pdf)
- [Deep Learning with Python](https://amzn.to/2TJcmz3)
    - Keras 창시자가 집필한 책
- [Keras 공식 홈페이지 문서 (한글)](https://keras.io/ko/)
