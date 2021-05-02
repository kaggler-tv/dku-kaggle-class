# 18. Image Classification

18장에서는 이미지 분류 대회 접근법에 대해 살펴보겠습니다. 일반적으로 이미지 분류 대회를 접근할 때는 EDA/Preprocessing, Image Augmentation, Image Modeling, Metadata Modeling, Ensembling 순으로 진행합니다. 

## 18.1 EDA / Preprocessing

이미지 데이터가 JPG/PNG로 처럼 RGB 이미지 포맷으로 저장돼있을 경우에는 [OpenCV](https://opencv.org/)나 [Pillow](https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#using-the-image-class) 라이브러리를 사용해 불러와서 처리합니다. 

X-ray, MRI 처럼 의료용 이미지일 경우 [DiCOM](https://en.wikipedia.org/wiki/DICOM) 형태로 제공됩니다. DiCOM은 의료 이미지 데이터를 저장하는 표준 포멧입니다. DiCOM은 환자의 성별, 이미지 저장시 사용한 기기 등 metadata를 포함합니다. DiCOM을 불러올 때는 [Pydicom](https://github.com/pydicom/pydicom) 라이브러리를 사용합니다. 

OpenCV, Pillow, Pydicom을 사용해 불러온 이미지에 `matplotlib.pyplot`에서 제공하는 `imshow`를 사용해서 시각화 할 수 있습니다. 이미지 시각화를 통해 이미지의 사이즈, 밝기, 색감등을 확인해서 추후 전처리가 필요한지 결정을 내립니다. 

이미지 사아즈가 동일하지 않다면 resize를 통해 사이즈를 맞춰주고 밝기나 색감이 다르다면 gray scale로 통일하거나 normalization을 적용해주기도 합니다. 또한 이미지 내의 주요 부분만 추출하기 위해 cropping을 적용하기도 합니다. Cropping을 할 때는 중요 부분을 같이 잘라내지 않도록 주의가 필요합니다. 

## 18.2 Image Augmentation

Image augmentation은 16장에서 다뤄본 것 처럼 이미지를 회전하거나 뒤집거나 확대/축소 등을 사용해서 데이터의 양을 부풀리는 과정입니다. 또한 Mixup, Cutout, CutMix등의 방법도 존재하며 세부 사항은 16장을 참고바랍니다. 

```{tip}
Image augmentation을 cross validation과 함께 사용할 경우 augmented된 이미지는 원본 이미지와 같은 fold에 속해야 합니다. 모델 학습에 사용한 데이터가 검증 데이터셋에도 존재한다면 모델의 성능은 상대적으로 좋게 나올 것이므로 올바른 모델 검증이 진행되지 않기 때문입니다. 
```

## 18.3 Image Modeling

이미지 데이터를 처리하는 모델을 구축할 때는 직접 CNN 구조를 디자인하거나 또는 17장에서 살펴본 pre-trained 모델을 사용할 수도 있습니다. 데이터과학 대회에서는 이미 방대한 데이터셋에 대해 검증된 pre-trained 모델을 주로 사용합니다. 

Pre-trained 모델을 사용하는 방법도 크게 3가지로 구분됩니다. 첫번째는 Pre-trained 모델을 있는 그대로 사용해서 직접 예측하는 방법입니다. 두번째는 pre-trained 모델의 변수를 고정 시킨 후 새 레이어를 추가 후 학습 및 예측을 하는 방법입니다. 세번째는 pre-trained 모델의 변수도 함께 학습 후 예측하는 방법입니다. 일반적으로 세번째 방법을 fine-tuning이라고 부르며 대회에서 최종 모델로 많이 사용하는 방법입니다. 

Pre-trained 모델을 베이스라인 모델로 사용할 때 일반적으로 Inception-v3, ResNet-50, 그리고 EfficentNet-B0을 사용합니다. 베이스라인 모델은 대회 초기에 여러가지 실험을 적용해볼 수 있도록 기준이 되는 모델입니다. 베이스라인 모델에 다양한 이미지 augmentation 및 전처리를 적용해 어떤 augmentation과 전처리 방법이 성능을 높이는지 실험을 해볼 수 있습니다. 그렇게 해서 선정된 augmentation과 전처리 기법으로 데이터를 처리한 후 최종 모델로 EfficientNet-B3 또는 B3 이상을 사용해 성능을 더 올릴 수 있습니다. 

## 18.4 Metadata Modeling

18.1절에서 언급한 것 처럼 의료용 데이터 포멧인 DiCOM 같은 경우 이미지에 대한 metadata가 함께 제공될 수 있습니다. 이러한 metadata를 이미지 데이터와 함께 사용 시 최종 예측 성능을 향상 시킬 수 있습니다. Metadata를 결합하는 방법은 두 가지가 있습니다. 첫번째는 metadata만을 가지고 lightgbm, xgboost등의 모델을 학습한 후 이미지 모델과 앙상블하는 방법입니다. 두번째는 CNN 모델을 구성해두고 sub-network로 metadata를 처리할 수 있는 신경망을 구축한 뒤 CNN 모델과 sub-network의 결과물을 합친 후 dense 레이어에 전달해서 최종 결과물을 산출하는 방법입니다. 

2019년과 2020년에 열린 의료용 이미지 대회인 Melanoma Classification 대회의 1위 솔루션이 모두 metadata를 함께 사용한 방법이였습니다. 2019년도와 2020년도 모두 CNN 모델의 결과물과 sub-network에서 metadata를 처리해 나온 결과물을 합친 뒤 추가 레이어에 전달해서 최종 예측을 했습니다.

- 그림 18- [Melanoma Classification 2019 1st Place](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154683) 

- 그림 18- [Melanoma Classification 2020 1st Place](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412) 

## 18.5 Ensembling

이미지 대회에서 앙상블을 적용할 때도 여러가지 방법이 존재합니다. 첫번째는 서로 다른 pre-trained CNN 모델을 앙상블 하는 방법입니다. ResNet, EfficientNet, Inception-v3 등의 모델을 학습 후 결과물을 앙상블하는 경우입니다. 두번째 방법은 동일한 CNN 모델에 입력 이미지 사이즈를 다르게 해서 여러개의 모델을 생성 후 앙상블하는 방법입니다. 마지막 방법은 GBDT 처럼 CNN과는 다른 종류의 알고리즘을 CNN과 함께 앙상블 하는 방법입니다. 

## 18.6 Tips

이미지 데이터과학 대회 참여 시 도움 되는 팁들에 대해서 살펴보겠습니다. 

- GPU/TPU는 가능하면 사용하는 것이 좋습니다. CNN 모델을 빠르게 학습할 수 있기 때문입니다. 

- Mixed precision을 사용해 모델 학습을 권장드립니다. Mixed precision은 모델 가중치의 소수점 자리수를 필요에 따라 16자리 또는 32자리로 사용하는 방법입니다. Mixed precision을 통해 모델 학습 시간을 줄일 수 있습니다.

- Learning rate scheduler 사용을 권장드립니다. 모델 학습이 진행됨에 따라 learning rate를 다르게 가져가는 방법이며 실전에서 모델 성능 향상에 효과가 있다고 알려져 있습니다. 

- Test-time augmentation(TTA) 또한 모델 성능 향상에 도움되기 때문에 사용을 권장드립니다. 

## 18.7 참고자료

- [Computer Vision Code Examples](https://keras.io/examples/vision/) (Keras)
- [Petals to the Metal | Kaggle](https://www.kaggle.com/c/tpu-getting-started)
    - [Pretrained CNN Epic Fight ⚔️ | Kaggle](https://www.kaggle.com/servietsky/pretrained-cnn-epic-fight)
    - [Rotation Augmentation GPU/TPU - [0.96+] | Kaggle](https://www.kaggle.com/cdeotte/rotation-augmentation-gpu-tpu-0-96)
- [Melanoma Classification 2019 1st Place ](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/154683)
- [Melanoma Classification 2020 1st Place](https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/175412)
- [머신러닝 마스터 클래스](https://www.upaper.net/jeongyoonlee/1136706) (PyTorch)
    - 10장. 이미지 모델링
