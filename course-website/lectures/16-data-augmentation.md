# 16. Data Augmentation

Data augmentation은 기존 데이터에 일정 변환을 가해서 학습/예측을 위한 데이터 양을 늘리는 기법입니다. 일반적으로 이미지 데이터에 사용하는 방법입니다. 예를 들어 그림 16-1 처럼 앵무새 사진이 있을 때 사진을 오른쪽으로 뒤집거나 확대하거나 필터를 추가해서 여러 장의 이미지를 생성하는 방법을 data augmentation이라고 합니다. 

그림 16-1

이미지를 회전하거나 확대 하거나 축소하는 것 외에도 여러개의 이미지를 결합하는 방식의 data augmentation이 존재합니다. Mixup은 서로 다른 이미지를 투명도를 조정후 겹쳐서 하나의 이미지를 생성하는 것이며 Cutout은 이미지의 일정 부분을 삭제하는 방법입니다. CutMix는 서로 다른 이미지의 일부를 자른 뒤 합쳐서 하나의 이미지를 생성하는 방법입니다. 

- 그림 16-2 Mixup, Cutout, CutMix 예시

모델 학습시 데이터의 양이 부족할 때 data augmentation을 통해 데이터의 양을 부풀려서 학습이 더 잘되게 할 수 있습니다. 뿐만 아니라 예측 시에도 data augmentation을 적용해서 여러 개의 이미지에 대해 예측한 후 평균을 구해서 최종 예측값을 산출 할 수 있습니다. 예측 시에 data augmentation을 적용하는 과정을 Test time augmentation(TTA)라고 합니다. TTA는 컴퓨터비전 데이터과학 대회에서 일반적으로 사용하는 방법입니다. 

## 16.1 Image Augmentation

이미지 데이터를 augmentation 할 때 사용가능한 라이브러리를 살펴보겠습니다. [Scikit-image](https://scikit-image.org/)는 scikit-learn 개발진이 개발한 라이브러리이며 사용하기가 쉽습니다. Numpy 행렬로 저장된 이미지를 변환하는 기능을 제공합니다. [Albumentations](https://github.com/albumentations-team/albumentations/)은 캐글 사용자들이 개발한 라이브러리입니다. 빠른 성능을 보이며 다양한 augmentation 기법들을 지원합니다. 또한 pipeline API를 제공하기 때문에 여러개의 augmentation 기법을 모아 한번에 적용 가능케 합니다. [keras.ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)는 Keras에서 제공하는 함수이며 간단한 augmentation 기법을 제공합니다. 캐글 사용자가 정리한 각종 이미지 augmentation 라이브러리 [비교글](https://www.kaggle.com/parulpandey/overview-of-popular-image-augmentation-packages)에서 각 라이브러리에 대한 추가 정보를 살펴볼 수 있습니다. 

## 16.2 Text Augmentation

Text augmentation은 문자열 데이터에 augmentation을 적용하는 방법입니다. Back translation 방법, Easy data augmentation 방법, 그리고 sentence-level transformation 방법이 존재합니다. 

Back translation은 원문 텍스트를 다른 언어로 번역하고 다시 원문으로 번역해서 얻은 데이터를 추가로 사용하는 방법입니다. 예를 들어 영문 데이터를 불어로 번역한 후 다시 영문으로 번역해 추가 데이터로 사용하는 것입니다. 번역을 여러번 거치면서 같은 의미지만 달라진 부분이 있을 수 있기 때문에 이와 같은 방법을 사용해 데이터의 양을 부풀립니다. 

Easy Data Augmentation은 synonym replacement, random insertion, random swap, 그리고 random deletion 방법이 존재합니다. Synonym replacement는 유사어로 단어를 변환하는 방법입니다. 

- 그림 16-X 원문

- 그림 16-X Synonym Replacement 적용 예시

Random Insertion은 유사어를 문장내 랜덤한 위치에 삽입하는 방법입니다. 문장이 문법적으로 올바르지 않게 변하지만 머신러닝 모델의 성능을 높여주기도 합니다. 

- 그림 16-X Random Insertion 적용 예시

Random Swap은 무작위로 단어의 위치를 바꾸는 방법 입니다. 

- 그림 16-X Random Swap 적용 예시

Random Swap은 무작위로 단어를 삭제하는 방법입니다. 

- 그림 16-X Random Deletion 적용 예시

Sentence-level transformation은 문장 단위로 실시하는 augmentation 기법입니다. Shuffle Sentences와 Exclude Duplicate방법이 존재합니다. Shuffle Sentences는 문단 내의 문장들의 순서를 바꾸는 방법입니다. Exclude Duplicate는 문단 내의 중복 문장을 제거하는 방법입니다. 

이 외에도 character-level transformation 방법이 있습니다. Character-level transformation은 비슷하게 생긴 문자로 대체해서 데이터를 증강하는 방법입니다. 얘를 들어 `E`를 `3`으로 바꾸거나 `A`를 `4`로 바꿔서 증강하는 방법입니다. 

텍스트 augmentation 라이브러리는 이미지 augmentation 라이브러리 처럼 많지는 않습니다. 가장 최근의 나온 라이브러리로 NLPAug 라이브러리가 있습니다. WordNet, word2vec, fasttext, BERT 등의 embedding 방법을 통해 유사어를 찾아 데이터를 증강해줍니다. [예제 코드](https://github.com/makcedward/nlpaug/blob/master/example/textual_language_augmenter.ipynb)를 통해 사용법을 자세히 살펴볼 수 있습니다. 

## 16.3 참고자료

- [Overview of popular Image Augmentation packages](https://www.kaggle.com/parulpandey/overview-of-popular-image-augmentation-packages)
- [Data Augmentation in NLP: Best Practices From a Kaggle Master](https://neptune.ai/blog/data-augmentation-nlp)
- [Tweet Sentiment Extraction Notebooks](https://www.kaggle.com/c/tweet-sentiment-extraction/notebooks)
- [Real or Not? NLP with Disaster Tweets Notebooks](https://www.kaggle.com/c/nlp-getting-started/notebooks)
    - [NLP Augmentation_BERT](https://www.kaggle.com/nandhuelan/nlp-augmentation-bert)
