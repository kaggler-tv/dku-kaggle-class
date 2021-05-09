# 10. Pipeline, Make

10장에서는 Pipeline과 Pipeline 생성 툴인 Make에 대해 살펴보겠습니다. 

## 10.1 Pipeline

데이터과학 대회를 참여하다보면 다양한 피쳐를 생성해야되고 다양한 모델을 시도해봐야 하며 하이퍼파라미터 최적화도 진행해야 되고 마지막에는 앙상블도 사용하게 됩니다. 이처럼 모델 구축 과정이 점점 복잡하게 됩니다. 

우승권에 들게 되면 코드를 제출해야 하는데 이 때 코드가 결과물을 어느 정도 재현이 가능해야 합니다. 하지만 대회 기간 동안 구축한 코드가 너무 복잡하다면 종종 결과물을 재현할 수 없는 경우가 발생합니다. 그렇기 때문에 재현을 나중에 가능케 하기 위해 pipeline을 정리해서 구축할 필요가 있습니다. 또한 개발하는 도중에도 어떤 요소에서 문제가 생겼는지 빨리 찾아내고 고칠 수 있도록 pipeline을 만들고 사용할 필요가 있습니다. 

좋은 pipeline은 첫번째로 **version control**이 지원돼야 합니다. 새로운 코드를 추가 했을 때 오류가 발생했을 시 이전 버전으로 돌아가면 문제가 해결 되도록 지원된다면 관리가 용이합니다. 두번째는 pipeline 구축시 **modular design**을 할 필요가 있습니다. 여러가지 module들이 독립적으로 돌아갈 수 있어야 하며 하나의 module에서 오류가 발생해도 다른 module에는 영향을 주지 않아야 합니다. 세번째로 **dependency check**이 지원돼야 합니다. 예를 들어 앙상블을 하기 위해선 먼저 학습된 개별 모델이 필요합니다. 이처럼 특정 기능이 수행될 때 필요한 요소가 없다면 없는 요소를 탐지하고 알려줄 수 있는 기능을 dependency check라고 합니다. 네번째로 **automated execution**이 필요합니다. dependency check에서 탐지된 파일들을 자동으로 실행시켜서 전체 pipeline이 정상적으로 작동하게 해주는 기능입니다. 

일반적으로 version control은 Github을 통해 사용하며 modular design은 각각의 module을 `.py`파일로 스크립트화 해서 관리합니다. Depedency check하고 automated execution은 pipeline management tool을 사용해서 구현합니다. 

## 10.2 Pipeline Tool

가장 많이 사용되는 pipeline tool로는 [Airflow](https://airflow.apache.org/), [Luigi](https://github.com/spotify/luigi), 그리고 [Make](https://en.wikipedia.org/wiki/Make_(software)#Makefile)가 있습니다. Airflow와 Luigi는 기업에서 많이 사용하며 여러 개의 서버가 서로 통신을 하면서 pipeline을 관리해줍니다. Airflow는 2014년에 Airbnb에서 오픈소스로 공개했으며 Luigi는 2012년에 Spotify가 오픈소스로 공개했습니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch10-img01.jpg?raw=true)
- 그림 10-1 Airflow와 Luigi 로고([출처](https://docs.google.com/presentation/d/1aQztmEeidVqLjnGhu6LKcWo_VeOGacpRaaY3_gQ9CcA/edit?usp=sharing))

개인이 사용하는 pipeline tool로는 Make가 자주 사용됩니다. 1976년에 Bell Lab의 Stuard Feldman이 개발 했으며 macOS와 Linux에는[GNU Make](https://www.gnu.org/software/make/)가 기본으로 탑재돼있습니다.

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch10-img02.jpg?raw=true)
- 그림 10-2 Stuard Feldman([출처](https://docs.google.com/presentation/d/1aQztmEeidVqLjnGhu6LKcWo_VeOGacpRaaY3_gQ9CcA/edit?usp=sharing))

## 10.3 Make

Make는 [영상 자료](https://youtu.be/5dBnsQJAkAw?t=1596)와 함께 학습하시길 권장드립니다 .