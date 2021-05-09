# 14. Embeddings

Embedding은 단어를 정수로 label/ordinal encoding 한 후 실수 벡터로 변환하는 방법입니다. `dog`라는 단어를 `2`라는 정수로 label encoding 한 후 `[0.25, 1.00]`의 실수 벡터로 변환하는 과정을 예시로 들 수 있습니다. Embedding은 신경망 모델을 통해 학습해서 얻게 됩니다. [13.3.3절](13-text-features.md)에서 embedding시 사용하는 신경망 모델 구조의 예시를 확인할 수 있습니다. 

## 14.1 Embedings in Keras

이번 절에서는 Keras에서 제공하는 embedding layer를 살펴보겠습니다. Keras의 `layers.Embedding()`함수를 사용해서 embedding layer를 적용할 수 있습니다. 문자열을 입력으로 받는 레이어 다음에 embedding layer을 쌓게 되면 embedding layer 출력값이 해당 문자열에 대한 embedding 값이 됩니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch14-img01.jpg?raw=true)
- 그림 14-1 layers.Embedding 사용 예제([출처](https://docs.google.com/presentation/d/1ultSXr-_wihsh4Cu-6y5TQIoFIorV5-LyxCP6fBbR6w/edit?usp=sharing))

## 14.2 Pretrained Embeddings

Embedding layer를 사용해 직접 embedding 값을 산출할 수도 있지만 이미 학습된 embedding 값을 사용할 수도 있습니다. Pretrained embedding 값은 Wikipedia, Twitter, Common Crawl 등의 대규모 문서 데이터로 미리 학습한 embedding 값을 뜻합니다. 해당 값들은 오픈 소스로 공개돼있습니다. 구글에서 공개한 [Word2Vec](https://code.google.com/archive/p/word2vec/), 페이스북에서 공개한 [fastText](https://fasttext.cc/), 그리고 스탠포드에서 공개한 [GloVe](https://nlp.stanford.edu/pubs/glove.pdf)가 대표적인 pretrained embedding입니다. Pretrained embedding값은 대규모의 데이터로 학습을 했기 때문에 단어간 관계와 의미를 비교적 잘 나타냅니다. 

Word2Vec은 CBOW 또는 Skip-gram 구조를 활용해 embedding 값을 학습합니다. fastText는 문자(character) 단위로 묶어서 하나의 단어로 취급한 뒤 embedding을 실시합니다. GloVe는 단어 빈도를 기반으로 행렬 분해(matrix factorization)을 적용해 embedding값을 산출합니다. 

fastText는 상당히 빨리 처리되는 알고리즘입니다. 하지만 실제 적용시 모델 성능은 Word2Vec과 GloVe와 비교했을 때 상대적으로 낮게 나오는 편입니다. Word2Vec과 GloVe는 모델 학습시 사용하면 성능은 유사하게 나오지만 GloVe가 더 간결한 구조를 지닙니다. 그래서 최근에는 GloVe를 많이 사용하는 추세입니다. 

Pretrained embeddings 값을 사용하기 위해선 [Word2Vec](https://code.google.com/archive/p/word2vec/), [fastText](https://fasttext.cc/), [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) 각 웹사이트에서 제공하는 embedding 값을 다운로드 받아야 합니다. Embedding 값이 저장된 `txt`파일을 파이썬 딕셔너리 형태로 읽어 온 뒤(그림 14-2) 훈련/시험데이터에 있는 각 단어의 embedding 값이 저장된 embedding_matrix를 생성해줍니다(그림 14-3). 그리고 나서 keras의 embedding layer를 생성할 때 embedding_matrix를 `weights`파라미터에 입력하고 `trainable`파라미터를 `False`로 설정하면(그림 14-4) embedding layer의 weights를 학습하지 않게 됩니다. 이렇게 설정을 마치면 단어가 embedding layer를 거칠 때 pretrained embeddings값으로 변환되어 출력됩니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch14-img02.jpg?raw=true)
- 그림 14-2 GloVe embedding 값 불러오기([출처](https://docs.google.com/presentation/d/1ultSXr-_wihsh4Cu-6y5TQIoFIorV5-LyxCP6fBbR6w/edit?usp=sharing))

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch14-img03.jpg?raw=true)
- 그림 14-3 embedding_matrix 생성([출처](https://docs.google.com/presentation/d/1ultSXr-_wihsh4Cu-6y5TQIoFIorV5-LyxCP6fBbR6w/edit?usp=sharing))

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch14-img04.jpg?raw=true)
- 그림 14-4 embedding_layer 생성([출처](https://docs.google.com/presentation/d/1ultSXr-_wihsh4Cu-6y5TQIoFIorV5-LyxCP6fBbR6w/edit?usp=sharing))

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch14-img05.jpg?raw=true)
- 그림 14-5 embedding_layer를 사용한 신경망 모델([출처](https://docs.google.com/presentation/d/1ultSXr-_wihsh4Cu-6y5TQIoFIorV5-LyxCP6fBbR6w/edit?usp=sharing))

그림 14-4 에서 구축된 embedding layer를 사용해 그림 14-5 처럼 모델을 구축하면 모델 학습시 embedding layer를 제외한 나머지 층의 가중치만 학습을 하게 됩니다. 

## 14.3 RNN

Recurrent Neural Network(RNN)은 시퀀스 데이터를 다루는 신경망의 한 종류입니다. 시퀀스 데이터는 순서가 존재하는 데이터를 의미합니다. 텍스트 데이터와 시계열 데이터가 시퀀스 데이터의 예로 볼 수 있습니다. 

RNN의 종류로 [LSTM](https://www.bioinf.jku.at/publications/older/2604.pdf)과 [GRU](https://arxiv.org/abs/1406.1078)가 있습니다. LSTM은 1990년대에 개발된 구조이며 그 당시 시퀀스 데이터에 압도적인 성능을 보여 각광을 받았습니다. GRU는 LSTM을 조금 더 단순화한 구조이며 뉴욕대의 조경현 교수에 의해 2014년에 공개됐습니다. 

RNN은 입력 변수를 3차원 형태로 받습니다. 각 차원은 샘플 개수, 피쳐 개수, 그리고 시퀀스 길이를 뜻합니다. 

RNN은 메모리를 통해 이전 입력값의 정보를 기억해뒀다가 다음 입력값을 처리할 때 같이 사용해서 처리합니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch14-img06.jpg?raw=true)
- 그림 14-6 RNN 구조([출처](https://docs.google.com/presentation/d/1ultSXr-_wihsh4Cu-6y5TQIoFIorV5-LyxCP6fBbR6w/edit?usp=sharing))

그림 14-6는 RNN을 도식화한 그림입니다. `v` 벡터에 이전 단계의 정보를 저장해서 다음 시퀀스를 처리시 함께 사용합니다. 

## 14.3 RNN in Keras

Keras에서 제공하는 `LSTM` 또는 `GRU` 레이어를 사용해 RNN을 구현할 수 있습니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch14-img07.jpg?raw=true)
- 그림 14-7 LSTM 사용 예제([출처](https://docs.google.com/presentation/d/1ultSXr-_wihsh4Cu-6y5TQIoFIorV5-LyxCP6fBbR6w/edit?usp=sharing))

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch14-img08.jpg?raw=true)
- 그림 14-8 GRU 사용 예제([출처](https://docs.google.com/presentation/d/1ultSXr-_wihsh4Cu-6y5TQIoFIorV5-LyxCP6fBbR6w/edit?usp=sharing))

RNN 레어어를 여러개 쌓기 위해선 그림 14-8처럼 `return_sequences`파라미터를 `True`로 설정해서 3차원 형태의 값이 반환되도록 설정해야 합니다. 

## 14.4 참고자료

- [Keras - Working with RNNs Tutorial](https://keras.io/guides/working_with_rnns/)
- [Keras - Pre-trained Word Embedding Tutorial](https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html)
- [머신러닝 마스터 클래스](https://www.upaper.net/jeongyoonlee/1136706)
    - 11장. 자연어처리
