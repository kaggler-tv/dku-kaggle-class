# 19. 후처리 및 기타 팁

19장에서는 후처리(post-processing) 방법 및 데이터과학 대회 참여시 사용가능한 기타 팁들에 대해 살펴보겠습니다. 

## 19.1 후처리

후처리는 모델이 예측한 값을 변경해서 시험 데이터셋에서의 성능을 향상시키는 단계입니다. 아래 표는 후처리 방법의 다양한 예시와 더불어 사용된 사례를 정리한 표입니다. 

후처리 방법 | 적용 사례 |
---------|----------|
 예측값의 분포를 시험셋의 종속변수 분포에 가깝게 변경 | [LANL Earthquake 10th](https://www.kaggle.com/c/LANL-Earthquake-Prediction/discussion/94466) | 
 예측값이 확률인 경우, 0과 1에 근사한 값을 0이나 1로 반올림 | [Google QUEST 11th](https://www.kaggle.com/c/google-quest-challenge/discussion/129839) | 
 예측값이 확률이고 메트릭이 logloss인 경우, 0과 1에 근사한 값을 clip | [Google QUEST 11th](https://www.kaggle.com/c/google-quest-challenge/discussion/129839) | 
 리더보드 점수가 향상되는 방향으로 예측값을 push | [Jigsaw 1st](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160986) | 
 하나의 샘플에서 여러 예측값이 발생한 경우, 하나의 값으로 결합 | [IEEE-CIS 6th](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111247) | 
- 표 19-1 후처리 방법과 적용 사례

첫번째 방법은 예측값의 분포를 시험셋의 종속변수 분포에 가깝게 변경하는 방법입니다. 시험셋의 종속변수는 주어지지 않기 때문이 일반적으로 알 수가 없습니다. 그래서 제출 파일에 하나의 상수 값을 입력 후 제출을 해서 리더보드에 반환되는 점수를 바탕으로 시험셋의 분포를 유추를 한 뒤 분포 변환을 적용합니다. Public 리더보드와 Private 리더보드의 분포가 다른 경우 Public 리더보드에 과적합되는 위험이 존재하지만 두 개의 리더보드의 분포가 유사한 경우 좋은 성능을 기대할 수 있습니다. 

두번째 방법은 예측값이 확률인 경우 0과 1에 근사한 값을 0과 1로 반올림하는 방법입니다. 

하지만 평가 지표가 logloss인 경우 두번째 방법은 오히려 안 좋은 성능을 보일 수 있습니다. 이런 경우 0과 1에 근사한 값을 특정 값으로 clip을 할 수 있습니다. 

네번째 방법은 리더보드 점수가 향상되는 방향으로 예측값을 push하는 방법입니다. 첫번째 제출값, 두번째 제출값, 그리고 세번째 제출값을 제출할 때 마다 리더보드 점수가 상승했다면 각 제출값 사이의 차이를 구해서 어느 방향으로 예측값이 움직여야 점수가 상승하는지 파악할 수 있습니다. 이렇게 알게된 방향으로 예측값을 조정할 수 있습니다. 

다섯번째 방법은 하나의 샘플에서 여러 예측값이 발생한 경우 하나의 값으로 결합하는 방법입니다. 이 방법은 IEEE-CIS 대회 데이터에 특화된 방법으로 소개가 됐었습니다. 

## 19.2 기타 팁

아래는 데이터과학 대회 참여 시 적용가능한 추가 팁들을 정리한 표입니다. 

팁 | 설명 | 적용 사례 |
---------|----------|----------|
 Pseudo-labeling | 시험셋 샘플 중 확실한 샘플을 학습셋에 추가 | [Instant Gratification - cdeotte](https://www.kaggle.com/cdeotte/pseudo-labeling-qda-0-969) |
 Adversarial Validation | 시험셋과 비슷한 피쳐분포를 가지는 샘플을 검증셋으로 선택 | [M5 - tunguz](https://www.kaggle.com/tunguz/m5-adversarial-validation) |
 Unsupervised Learning | Autoencoder를 사용하여 latent 피쳐를 추출하여 사용 | [shivamb’s tutorial](https://www.kaggle.com/shivamb/semi-supervised-classification-using-autoencoders), [Porto Seguro 1st](https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44629) |
- 표 19-2 기타 팁과 적용 사례

## 19.3 참고자료

데이터과핟 대회 수상자들의 솔루션을 살펴보면 다양한 팁들을 얻을 수 있습니다. 아래에는 각 분야별 주요 대회 우승 솔루션이 정리돼있습니다.

- Computer Vision: 
    - SIIM-ISIC Melanoma Classification ([1st](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412), [2nd](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175324), [3rd](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175633))
    - Google Landmark Recognition 2020 ([1st](https://www.kaggle.com/c/landmark-recognition-2020/discussion/187821), [2nd](https://www.kaggle.com/c/landmark-recognition-2020/discussion/188299), [3rd](https://www.kaggle.com/c/landmark-recognition-2020/discussion/187757))
    - Google Landmark Retrieval 2020 ([1st](https://www.kaggle.com/c/landmark-retrieval-2020/discussion/176037), [2nd](https://www.kaggle.com/c/landmark-retrieval-2020/discussion/177078))

- NLP: 
    - Tweet Sentiment Extraction ([1st](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159477), [2nd](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159310), [3rd)](https://www.kaggle.com/c/tweet-sentiment-extraction/discussion/159910)
    - Jigsaw Multilingual Toxic Comment Classification ([1st](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160862), [3rd](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160964), [4th](https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification/discussion/160980))

- Tabular:
    - IEEE-CIS Fraud Detection (1st [#1](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111284)/[#2](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111308), 2nd [#1](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111321)/[#2](https://www.kaggle.com/c/ieee-fraud-detection/discussion/111554))
    - Santander Customer Transaction Prediction ([1st](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/89003), [2nd](https://www.kaggle.com/c/santander-customer-transaction-prediction/discussion/88939))

- [Chai Time Data Science Podcast](https://anchor.fm/chaitimedatascience)
    - 유명한 데이터 사이언티스트들을 초대해서 그들의 팁을 공유하는 팟캐스트입니다. 
- [Heads or Tails - Kaggle Hidden Gems on Twitter](https://twitter.com/heads0rtai1s)
    - heads0rtails라는 캐글 사용자가 선별해서 공개하는 노트북 시리즈 입니다. heads0rtails은 EDA 노트북을 잘 만드는 것으로 유명한 캐글 사용자입니다. 
