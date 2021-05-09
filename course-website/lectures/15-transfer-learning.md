# 15. 전이학습 (Transfer Learning)

이번 장에서는 전이학습에 대해 살펴보겠습니다. 전이학습은 자연어처리 대회에서 자주 사용되는 기법입니다. 

최근 NLP 모델들의 복잡도가 증가함에 따라 학습 시간 및 학습시 요구되는 장비의 사양이 증가했습니다. 2018년에 구글에서 발표한 [BERT](https://arxiv.org/abs/1810.04805)는 3억 4천만개의 변수를 학습해야 합니다. 구글은 Nvidia V100 32GB GPU를 8개를 가지고 2주간 BERT를 학습했습니다. V100은 서버용 GPU이며 개당 가격은 대략 $13,061 입니다. GPU 하나당 약 1500만원이라고 볼 수 있는데 이것을 8개를 사용해 2주간 학습을 해야 합니다. 2020년에 OpenAI에서 발표한 [GPT-3](https://arxiv.org/abs/2005.14165v2) 모델은 1,750억개의 변수를 학습해야 합니다. 학습 비용만 약 460만 달러 정도가 소요 된다고 추정됩니다. 모델 하나를 학습하는데 약 5억원이 필요한 것입니다. 그렇기 때문에 일반인 뿐만 아니라 웬만한 기업에서도 BERT나 GPT-3를 처음부터 학습하기에는 굉장히 많은 비용과 시간이 필요합니다. 

그래서 BERT와 GPT-3처럼 복잡하지만 성능이 좋은 모델을 새로운 데이터에 적용하기 위해 나온 개념이 전이학습입니다. 전이학습은 이미 학습된 모델을 가지고 와서 새로운 데이터에 학습시키는 방법입니다. 처음 부터 학습을 해도 되지 않기 때문에 비용과 시간이 절약되는 방법입니다. 

전이학습에는 Freeze와 Fine-tune 방법이 존재합니다. Freeze는 선학습된 모델은 고정하고 레이어를 추가해서 추가된 레이어만 학습하는 방법입니다. 이 경우 freeze된 모델의 역할은 주어진 입력값을 고정된 출력값으로 산출하는 encoding 역할을 하게 되며 학습은 추가된 레이어에서 진행됩니다. Fine-tune은 선학습된 모델과 추가한 레이어 모두 새로운 데이터로 학습을 하는 방법입니다. 

[실습 코드](15-bert.ipynb)에서 BERT를 fine-tuning하는 과정을 확인할 수 있습니다.

## 15.1 선학습 NLP 모델 라이브러리 

전이학습을 하기 위해선 선학습된 NLP 모델이 필요합니다. 선학습된 NLP 모델들을 제공하는 라이브러리로 Transfomers와 Tensorflow Hub 라이브러리가 존재합니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch15-img01.jpg?raw=true)
- 그림 15-1 Transformers 로고([출처](https://docs.google.com/presentation/d/10iXh7a15zEU2Zuvs-od3rdc688CKSQfjP11-9on7RM4/edit?usp=sharing))

Transfomers에서는 최신 NLP 모델들을 제공합니다. 오픈소스로 공개된 32개의 공식 모델과 더불어 커뮤니티에서 개발한 모델들 또한 제공합니다. Transformers는 PyTorch와 Tensorflow/Keras 2.0과 호환되지만 대부분의 예제 코드는 PyTorch로 구현돼 있습니다. 

Tensorflow Hub은 Tensorflow/Keras 2.0에서 사용가능한 선학습 NLP 모델을 제공합니다. 이번 실습 파일에서는 Tensorflow Hub을 사용해 전이학습을 구현했습니다. 

이번 [실습 파일]((15-bert.ipynb))외에도 커뮤니티에서 제공되는 [Transformers를 사용한 예제](https://towardsdatascience.com/working-with-hugging-face-transformers-and-tf-2-0-89bf35e3555a)와 [Tensorflow Hub을 사용한 예제](https://www.kaggle.com/xhlulu/disaster-nlp-keras-bert-using-tfhub)를 참고하실 수 있습니다. 

## 15.2 참고자료

자연어 처리 분야가 최근들어 빠르게 발전하고 있기 때문에 입문자가 관련 연구를 따라가기 어려울 수 있습니다. 아래 참고 자료를 통해 최신 개념을 익혀 보시길 바랍니다. 

- [위클리 NLP](https://jiho-ml.com/tag/weekly-nlp/) by 박지호님
    - 박지호님은 현재 Google에서 근무하고 계시며 매주 발행하는 위클리 NLP에서 중요한 개념과 최신 연구 동향을 소개하고 있습니다. 
    - [Week 25 - NLP의 옵티머스 프라임, Transformer 등장!](https://jiho-ml.com/weekly-nlp-25/)
    - [Week 27 - 전에 배운걸 잘 써먹어야 산다 Transfer Learning](https://jiho-ml.com/weekly-nlp-27/)
    - [Week 28 - BERT만 잘 써먹어도 최고가 될 수 있다?](https://jiho-ml.com/weekly-nlp-28/)
- [NLP in Korean – Anything about NLP in Korean](https://nlpinkorean.github.io/)
    - 해당 블로그는 미국에서 유학중인 유학생분께서 NLP 주요 개념을 잘 설명한 영문 자료를 저자 동의하에 한국어로 번역해 공개한 블로그입니다. 
    - [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](https://nlpinkorean.github.io/illustrated-bert/)
    - [The Illustrated Transformer](https://nlpinkorean.github.io/illustrated-transformer/)
    - [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://nlpinkorean.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)
- [머신러닝 마스터 클래스](https://www.upaper.net/jeongyoonlee/1136706)
    - 11장. 자연어처리
- [책 한 권 값으로 O'Reilly 책 다 보기](https://hack-jam.tistory.com/31?fbclid=IwAR2EPnLEk56kLEd1rnZxtB5_LNn5JM6ENHYBmVNsYJO2BC-Mk616_NnTSi4)
    - O'Reilly 출판사는 데이터과학에 대한 여러 책을 보유하고 있습니다. 