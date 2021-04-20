# 2. Setup
데이터과학 전반적으로 여러 가지 환경 설정 하는 것이 필요하며, 대회 참가시에는 더군다나 빨리 데이터를 분석해 인사이트를 뽑아내고 모델을 구축하는 것이 중요하므로 어떤 컴퓨터에서나 빠르게 본인의 환경 구축을 하는 것이 상당히 중요합니다. 그래서 2장에서는 데이터과학 대회 참여를 위한 환경 설정에 대해 알아보겠습니다. Github 설정 방법, 파이썬 환경 설정 방법, 그리고 터미널 사용 방법을 배워보겠습니다. 

```{note}
터미널 사용에는 호불호가 존재합니다. 허나 터미널 에 익숙해지면 개발 과정을 빠르게 해준다는 장점이 있기 때문에 이번 장에서 같이 소개해 볼 예정입니다. 
```

## 2.1 Github 소개

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img01.jpg?raw=true)
- 그림 2-1 Github 로고([출처](https://docs.google.com/presentation/d/1GyEc1zvn-4NsSliYf-pidnFT-9MBwePl4Gq1qn2saT4/edit#slide=id.p))

Github은 전 세계적으로 가장 많이 사용되는 코드 버전 관리 및 코드 협업 웹 플랫폼입니다. 4천만 명이 넘는 유저를 보유하고 있고, 1억개 이상의 코드 리포지토리를 보유하고 있습니다. 1억개가 넘는 코드 리포지토리 중에 약 2천 8백만개가 오픈 소스로 공개가 되어 있습니다. 그렇기 때문에 누구나 Github 웹사이트에 가면 2천 8백만개 이상의 프로그램들에 대한 코드를 다운로드 받아서 사용할 수 있습니다. 그만큼 전 세계적으로 가장 크고 많은 유저들이 사용하는 공개 코드 버전 관리 및 협업 웹 플랫폼입니다 

2018년에는 마이크로소프트가 무려 75억 달러, 한화로는 약 8조원 정도가 넘는 돈으로 Github을 인수했습니다. Github은 오픈 소스 플랫폼이므로 수익이 많이 나는 회사는 아니었지만 마이크로소프트는 4천만 명이 넘는 개발자와 1억개 이상의 코드 리포지토리에 8조 원이 넘는 가치를 부여한 것이라고 볼 수 있습니다. 그만큼 Github은 대단한 임팩트를 가진 회사입니다. 

그래서 아직 Github을 사용 안해보셨다면 이번 기회를 통해 사용법을 습득해보시길 바랍니다. 데이터과학 대회 뿐만 아니라 모든 프로젝트의 코드를 Github을 통해 관리한다면, 커리어와 학업에도 큰 도움이 될 것입니다. 

### 2.1.1 Github 용어
이번 절에서는 Github에서 사용하는 용어들에 대해 배워보겠습니다. 가장 먼저 배울 용어는 **Git**입니다. Git은 코드 버전 관리 프로그램의 한 종류입니다. 예전에는 [Subversion](https://ko.wikipedia.org/wiki/%EC%84%9C%EB%B8%8C%EB%B2%84%EC%A0%84)과 [CVS](https://ko.wikipedia.org/wiki/CVS)등의 다른 종류의 버전 관리 프로그램들도 사용했으나 지금은 Git이 가장 널리 사용되는 프로그램입니다. 

여러 사람이 팀으로 협업을 할 때 코드 관리가 상당히 어려울 수 있습니다. 예를 들어 팀 리포트를 작성할 때 `A조_리포트_Final.docx`라고 저장하고 다음 사람이 수정을 하면 `A조_리포트_Final2.docx` 또는 `A조_리포트_Final_Final.docx`과 같은 방식으로 파일명을 수정해가면서 작업하신 경험이 있으실거라고 생각됩니다. 소프트웨어나 코드 작성시 이렇게 한다면 관리가 안되기 때문에 **Git**과 같은 버전 관리 프로그램을 사용하는게 일반적입니다. 다른 기업이나 단체에서도 Github 플랫폼을 쓰지 않더라도 **Git**은 사용하는 경우가 대부분이기 때문에 반드시 짚고 넘어가야할 프로그램입니다. 

그 다음 용어는 **리포지토리**(repository)입니다. 이것은 코드 저장소를 뜻합니다. 그리고 **브랜치**(branch)는 각각 다른 코드 버전의 이름을 나타내는 용어입니다. 그리고 마스터/메인(master/main)은 가장 기본이 되는 디폴트 브랜치 이름을 뜻합니다. 과거에는 마스터라고 불렸는데, Github에서는 최근에 메인으로 이름을 변경했습니다. Github외에 다른 환경에 가면 마스터라고도 칭합니다.

**커밋**(commit)은 코드를 수정 후, 수정본을 제출하는 과정을 칭합니다. 커밋한 코드를 다른 사람에게 검토를 신청하는 과정을 **Pull Request** 또는 줄여서 **PR**이라고 부릅니다. Pull Request한 코드 변경 사항이 검토자로 부터 승인이 난 후 메인 브랜치에 반영을 하는 과정을 **머지**(merge)라고 합니다. 

앞서 언급한 7개의 용어만 아셔도 Github에서 코드를 관리하고 공유하는데 큰 어려움은 없을 겁니다. 이외에도 세부적으로 들어가면 더 많은 용어가 있는데, 직접 Github을 사용하시면서 하나씩 알아가는 것을 권장드립니다. 2.1.2절 부터 2.1.6절까지는 새로운 리포를 생성하고 코드 변경사항을 커밋하고, PR을 보내고, PR이 승인 난 후 머지하는 과정을 설명드릴 예정입니다. 책을 통해 전체 과정을 숙지한 후에 직접 실습해보시길 권장드립니다. 

### 2.1.2 새 리포지토리 생성
먼저 새로운 리포지토리를 생성하는 과정을 알아보겠습니다. Github에 로그인 하시면 우상단에 `+`모양의 버튼이 있습니다(그림 2-2). 해당 버튼을 누른 뒤 `New Repository`버튼을 누르면 그림 2-3과 같은 화면이 나옵니다. 그리고 나서 가장 먼저 리포지토리 이름을 입력해줍니다. 본 예시에서는 `hello-world`라고 입력했습니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img02.png?raw=true)
- 그림 2-2 로그인 후 우상단 `+` 버튼 클릭

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img03.jpg?raw=true)
- 그림 2-3 새 리포지토리 생성 화면([출처](https://docs.google.com/presentation/d/1GyEc1zvn-4NsSliYf-pidnFT-9MBwePl4Gq1qn2saT4/edit#slide=id.p))

그리고 나서 리포지토리가 어떤 리포지토리인지 설명하는 내용을 `Description`란에 적으면 됩니다. 그 후 해당 리포지토리를 공개(Public)할 것인지 비공개(Private)할 것인지 여부를 정해주면 됩니다. 

공개/비공개 여부를 선정한 후에는 `README`라는 텍스트 파일을 생성할 지 여부를 결정해줍니다. 마찬가지로 `.gitignore`파일과 `license`파일을 생성할 지 여부도 결정 한 후 녹색 `Create repository`버튼을 누르면 새로운 리포지토리가 생성됩니다. 

```{admonition} README, .gitignore, license 파일의 역할
:class: note

README - 리포지토리를 탐색하기전 알아두어야 할 주요사항 명시

.gitignore - git 추적을 하지 않을 파일 명시

license - 리포지토리(코드) 사용에 대한 라이센스
```

### 2.1.3 새 브랜치 생성
새롭게 생성한 리포지토리에 들어가면 그림 2-4에 나와 있듯이 `branch: master`라는 버튼이 있습니다. 해당 버튼을 클릭하면 새로운 브랜치 명을 입력할 수 있는 창이 뜹니다. 브랜치 명을 `readme-edits`라고 입력하고 `Create branch`버튼을 클릭하면 새로운 브랜치가 생성됩니다.

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img04.gif?raw=true)
- 그림 2-4 새 브랜치 생성([출처](https://docs.google.com/presentation/d/1GyEc1zvn-4NsSliYf-pidnFT-9MBwePl4Gq1qn2saT4/edit#slide=id.p))

일반적으로 코드를 수정할 때는 마스터/메인 브랜치에서 작업하지 않고, 새로운 브랜치를 만든 후에 해당 브랜치에서 코드를 변경하고 커밋 후 PR을 통해 검토를 받고, 승인이 났을 때 변경사항을 마스터/메인 브랜치로 머지하는 과정을 거치게 됩니다. 이렇게 하는 이유는 여러 명이서 같은 코드를 동시에 수정할 시에 충돌이 발생할 수 있기 때문에, 충돌을 방지하기 위해서 위와 같은 방법을 사용합니다. 

### 2.1.4 코드 변경 후 커밋
앞서 생성한 `readme-edits`라는 브랜치에서 `README.md`파일을 수정해보고, 수정사항을 커밋해보겠습니다. 텍스트 파일 같은 경우에는 Github 웹사이트에서 직접 수정할 수 있습니다. `README.md`파일을 선택 후 우측 상단에 있는 연필 아이콘을 클릭하면(그림 2-5) 텍스트 파일을 직접 웹상에서 수정할 수 있습니다(그림 2-6). 수정을 한 후에는 어떤 변경 사항을 반영했는지 메시지를 남길 수 있습니다. 마지막으로 `Commit changes`라고 되어 있는 녹색 버튼을 클릭하면 변경 사항이 해당 브랜치에 커밋됩니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img05.png?raw=true)
- 그림 2-5 수정화면으로 넘어가는 연필 아이콘

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img06.jpg?raw=true)
- 그림 2-6 파일 수정 화면([출처](https://docs.google.com/presentation/d/1GyEc1zvn-4NsSliYf-pidnFT-9MBwePl4Gq1qn2saT4/edit#slide=id.p))

### 2.1.5 Pull Request 제출
앞서 제출한 커밋에 대한 코드 리뷰를 받기 위해선 Pull Request를 생성해야 합니다. 리포지토리 상단에 있는 `Pull requests` 탭을 클릭 후 `New pull request`라는 녹색 버튼을 클릭해서 새로운 PR을 생성할 수 있습니다(그림 2-7). 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img07.gif?raw=true)
- 그림 2-7 PR 생성 버튼

`New pull request`를 선택하면 어떤 브랜치로 PR을 생성할 지 선택할 수가 있습니다. 2.1.3절에서 생성한 `readme-edits`라는 브랜치를 선택하면, 어느 파일의 어느 부분이 변경 됐는지 확인할 수 있습니다. 그리고 나서 `Create pull request` 버튼을 클릭하면 PR이 생성됩니다(그림 2-8). 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img08.JPG?raw=true)
- 그림 2-8 PR 생성 과정

### 2.1.6 Pull Request 머지(merge)
PR이 생성 되면 리뷰어(검토자)를 선정할 수 있습니다. 혼자서 개인 프로젝트를 진행중이라면 PR 생성후 머지를 할 때 리뷰어 선정이 필요 없습니다. 하지만 팀으로 협업을 할 시에는 내가 작성한 코드에 대한 리뷰를 다른 팀원에게 신청할 수 있습니다. PR 페이지 우측 상단에 리뷰어를 선택할 수 있는 옵션이 있으며, 다른 팀 멤버로 리뷰어를 선정하면 해당 리뷰어가 코드를 검토하고 승인할 수 있습니다. 승인하는 과정을 Approval이라고 하며, 승인을 하게 되면 PR을 메인 브랜치로 머지할 수 있게 됩니다. 

머지를 할 때에는 그림 2-9에 나와 있는 `Merge pull request` 녹색 버튼을 눌러주면 됩니다. 버튼을 클릭하면 머지할 것인지 최종 확인을 한번 더 하게 됩니다. `Confirm merge`라는 버튼이 뜨는데, 해당 버튼을 누르면 PR이 메인 브랜치로 반영됩니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img09.JPG?raw=true)
- 그림 2-9 PR 세부 화면

메인 브랜치로 반영 후에는 변경사항을 반영하기 위해 생성해둔 브랜치는 더이상 필요없기 때문에 `Delete branch`버튼을 클릭해서(그림 2-10) 기존에 생성해둔 브랜치를 삭제하고 메인 브랜치만 유지하고 관리하는 것을 권장드립니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img10.JPG?raw=true)
- 그림 2-10 Delete branch 버튼

지금까지 다룬 내용을 정리하자면 메인 브랜치가 존재하고, 메인 브랜치의 코드는 직접 수정하는 것이 아니고 새로운 브랜치를 통해서 수정을 해줍니다. 새로운 브랜치를 생성해서 변경하고자 하는 파일들을 변경하고, 변경사항을 커밋한 후, 제출된 변경사항을 PR을 통해 검토하고, 이상이 없으면은 메인 브랜치로 머지합니다. 변경사항이 메인 브랜치에 반영된 후에는 새로 생성해둔 브랜치를 삭제함으로써 전체 코드 변경 과정이 완료됩니다. 

위 과정을 직접 실습해보는 것을 권장드립니다. 

```{tip}
1. 코드 변경 사항은 한 줄을 수정하더라도 새로운 브랜치를 생성해서 PR 후 머지하는 것을 권장드립니다. 해당 과정이 번거롭다고 해서 여러 수정사항을 모았다가 한번에 PR하면 코드 검토 시 오류 사항을 탐지하기 어렵기 때문입니다. 

2. Github 데스크톱 프로그램을 다운로드 받으셔서 로컬 컴퓨터에서 작업하는 것도 가능합니다. 숙련된 분들은 터미널에서도 모든 작업이 가능합니다. 
```

## 2.2 Python 환경 설정
다음으로는 파이썬 환경 설정에 대해 배워보겠습니다. 

### 2.2.1 Python 설치
파이썬 설치를 할 때 권장드리는 방법은 [Anaconda](https://www.anaconda.com/products/individual) 또는 Anaconda의 미니 버전인 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)라는 파이썬 배포판을 사용해 설치를 하는 것입니다. 파이썬 배포판을 활용해 설치를 하면 가상환경 구축이 쉽고, 각종 머신러닝 라이브러리를 설치하는 것도 쉽고, Jupyter Notebook/Lab과 같은 IDE도 포함되어 있기 때문에 파이썬으로 데이터과학을 하기가 편리합니다. 또한 Miniconda는 관리자 권한 없이도 설치가 가능합니다. 그러므로 학교/회사 서버 같이 본인이 관리자 권한이 없는 경우에도 Miniconda를 사용하면은 관리자 권한이 없어도 설치가 가능합니다. 

```{note}
아나콘다 설치 가이드
- 영문/공식 홈페이지: https://docs.anaconda.com/anaconda/install/
-  한글/윈도우즈: https://gracefulprograming.tistory.com/124
```
### 2.2.2 Anaconda/Conda 실행

| 운영체제 | Anaconda Navigator(입문자용) | Conda(숙련자용) |
|-----|-----|:--------|
|윈도우즈|시작메뉴에서 Anaconda Navigator 선택|시작메뉴에서 Anaconda Prompt 선택|
|맥OS|Spotlight 검색창에서 Navigator 검색|터미널에서 conda 명령어 선택|
|리눅스/윈도우즈 WSL||터미널에서 conda 명령어 선택|
- 표 2-1 Anaconda Navigator와 Conda 비교표

표 2-1은 Anaconda를 실행하는 방법을 정리한 표입니다. Anaconda를 실행하는 방법은 두 가지가 있습니다. 첫번째 방법은 Anaconda Navigator 프로그램을 이용해서 실행할 수도 있고, 두번째 방법은 Conda 터미널 프로그램을 활용해 실행하는 것입니다. 

입문자분들께는 GUI가 구축되어 있어 마우스 클릭으로 손쉽게 상호작용할 수 있는 Anaconda Navigator를 사용하는 것을 추천드립니다. 윈도우즈에서는 시작메뉴, 또는 검색창에 Anaconda Navigator를 검색해서 실행 시킬 수 있습니다. 맥OS에서는 Spotlight 검색창에서 Navigator를 검색해서 실행시킬 수 있습니다. 

윈도우즈에서 Conda를 사용하고자 하시면 시작메뉴에서 Anaconda Prompt를 선택해서 실행시킬 수 있으며, 맥OS/리눅스/윈도우즈 WSL에서는 터미널에서 conda 명령어를 통해 실행시킬 수 있습니다.

### 2.2.3 파이썬 가상환경 생성/실행

파이썬으로 데이터 분석을 하거나 개발할 때는 가상환경 사용을 권장드립니다. 가상환경 없이 파이썬으로 새로운 라이브러리나 프로그램을 설치하거나 삭제하게 되면 운영체제에 있는 프로그램에 직접적인 영향을 줄 수 있습니다. 반면 가상환경을 구축한 뒤에 해당 환경내에서 프로그램을 설치하거나 삭제하면 운영체제에 있는 프로그램에 영향을 주지 않습니다. 그러므로 가상환경 사용을 추천드립니다. 

파이썬에는 파이썬2와 파이썬3이 존재합니다. 2020년 기준으로 제가 추천드리는 파이썬 버전은 3.7 또는 3.8 버전입니다. 3.9가 최신이긴 하지만 아직까지 데이터과학 프로그램들과 완벽하게 호환되지 않습니다. 그래서 최신 버전보다 한단계 낮은 3.7이나 3.8을 추천드립니다. 

Anaconda Navigator에서 가상환경을 생성하기 위해선 `Environments` 탭을 클릭 후 하단에 있는 `Create` 버튼을 클릭합니다(그림 2-11). 팝업 창이 뜨면 가상환경 이름 입력과 파이썬 버전을 선택을 완료한 후에 `Create` 녹색 버튼을 누르면 가상환경이 생성됩니다(그림 2-12). 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img11.png?raw=true)

- 그림 2-11 Anaconda Navigator내 Create 버튼

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img12.JPG?raw=true)

- 그림 2-12 가상환경 생성 화면([출처](https://docs.google.com/presentation/d/1GyEc1zvn-4NsSliYf-pidnFT-9MBwePl4Gq1qn2saT4/edit#slide=id.p))

터미널에서는 `conda create -n [가상환경 이름] python=[파이썬 버전]`을 입력하면 새로운 가상환경이 생성됩니다. 예를 들어 아래 코드를 입력하면 파이썬 3.7 버전이 설치되어 있는 `py37`이라는 가상환경이 생성됩니다. 

```
conda create -n py37 python=3.7
```
터미널에서 가상환경을 실행시키기 위해서는 `conda activate [가상환경 이름]` 명령어를 사용하면 됩니다. 아래 코드는 위에서 생성한 `py37` 가상환경을 실행시키는 코드입니다. 

```
conda activate py37
```

### 2.2.4 Jupyter Notebook 실행

가상환경 실행 후에는 해당 환경 위에서 파이썬 프로그램을 실행하고 개발을 해야 합니다. 파이썬으로 코딩을 할 때 많이 사용하는 도구 중 하나인 Jupyter Notebook을 실행하기 위해선 Anaconda Navigator에서는 메인 화면에 선택할 수 있는 여러 프로그램 중 하나로 제공이 됩니다. 그러므로 Jupyter Notebook 아이콘 하단에 있는 `Launch` 버튼을 누르면 해당 가상환경에서 Jupyter Notebook을 실행할 수 있습니다. Anaconda Navigator에서 주의할 점은 Jupyter Notebook을 실행하기 전에 상단에 있는 선택창에서 실행하고자 하는 가상환경을 먼저 선택을 해주셔야 합니다. 기본 값으로 `base`라는 이름의 가상환경으로 설정이 되어 있습니다. 해당 값을 원하는 가상환경으로 변경 후 Jupyter Notebook을 실행하시길 권장드립니다(그림 2-13). 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img13.png?raw=true)
- 그림 2-13 Anaconda Navigator 화면

터미널에서는 가상환경을 `conda activate [가상환경 이름]`으로 실행시킨 후 `jupyter notebook` 명령어를 통해 Jupyter Notebook을 실행시킬 수 있습니다. 

### 2.2.5 VS Code / Vim

파이썬으로 개발을 할 때 추천드리는 코드 에디터는 VS Code와 Vim입니다. VS Code는 마이크로소프트에서 나온 오픈 소스 에디터이며 입문자나 전문가 모두에게 추천드립니다. VS Code나 나오기 전에는 Sublime, PyCharm과 같은 에디터를 활용했는데 VS Code가 출시된 후에는 대부분 VS Code를 활용하는 쪽으로 추세가 변했습니다. 윈도우즈/맥OS/리눅스 모두에서 사용가능하며 무료입니다. 

![](https://user-images.githubusercontent.com/1487073/58344409-70473b80-7e0a-11e9-8570-b2efc6f8fa44.png)
- 그림 2-14 VS Code 화면([출처](https://github.com/microsoft/vscode))

터미널 사용에 자신 있으신 분들께는 Vim을 추천드립니다. 터미널 상에서 개발을 하게 되면 개발 속도가 빨라지기 때문에 여러가지 장점이 있습니다. 단점은 터미널에서 사용하는 프로그램을 배우는데 시간이 많이 소요되는 것입니다. 하지만 한번 익숙해지면 개발 속도가 훨씬 더 향상되는 장점이 있습니다. 

![](https://upload.wikimedia.org/wikipedia/commons/thumb/8/8c/Vim-%28logiciel%29-console.png/300px-Vim-%28logiciel%29-console.png)
- 그림 2-15 Vim 화면([출처](https://en.wikipedia.org/wiki/Vim_(text_editor)))

### 2.2.6 캐글 노트북 / Google Colaboratory

지금까지 로컬 컴퓨터에 파이썬을 설치하고 환경 설정해서 활용하는 방법을 안내드렸습니다. 만약 로컬 컴퓨터에 환경 설정을 하고 개발할 여건이 안된다거나 본인 컴퓨터 외의 컴퓨터에서 개발을 하고 싶다면 캐글 노트북 또는 Google Colaboratory(Colab; 코랩)을 활용해 온라인 상에서 개발이 가능합니다. 

캐글 웹사이트([https://www.kaggle.com/](https://www.kaggle.com/))에 접속 후 회원가입을 하고 로그인을 하시면 캐글에서 제공하는 노트북 환경을 사용할 수 있습니다. 캐글 노트북을 사용하면 온라인에서 간단한 파이썬 개발이 가능합니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img16.JPG?raw=true)
- 그림 2-16 캐글 노트북 화면([출처](https://docs.google.com/presentation/d/1GyEc1zvn-4NsSliYf-pidnFT-9MBwePl4Gq1qn2saT4/edit#slide=id.p))

Google Colab ([https://colab.research.google.com/](https://colab.research.google.com/))은 구글에서 제공하는 주피터 노트북 환경이며 Github과 Google Drive와 연동이 가능합니다. Github에 있는 코드를 가지고 오거나 또는 Google Drive에 있는 파일을 가지고 오는 등의 유용한 기능들을 제공하기 때문에 Google Colab을 사용하는 것 또한 온라인 상에서 개발 할 수 있는 방법 중 하나입니다.  

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img17.JPG?raw=true)
- 그림 2-17 Google Colab 화면

## 2.3 터미널

터미널은 개인 취향에 따라 선호도가 갈리는 프로그램입니다. 그럼에도 불구하고 익숙해지면 개발 속도를 향상시켜준다는 장점이 있기 때문에 사용하게 됩니다. [유튜브](https://youtu.be/861NAO5-XJo)에 접속하셔서 터미널을 사용하는 demo 영상을 확인할 수 있습니다. 본 강의에서는 터미널 관련 내용들을 간략하게 소개하도록 하겠습니다. 

맥OS와 리눅스에서는 Terminal 프로그램을 활용해 사용하면 됩니다. 윈도우즈 10이상에서는 Windows Subsystem for Linux(WSL) 설정을 통해 터미널을 사용하시길 권장드립니다. WSL을 사용하면 리눅스에서 사용하는 모든 프로그램을 윈도우즈에서 사용이 가능합니다. 만약 WSL 사용이 제한된다면, [Cygwin](https://www.cygwin.com/)처럼 윈도우즈에서 지원하는 Bash 프로그램을 사용하시길 바랍니다. 

### 2.3.1 Bash 명령어

터미널에서 주로 사용하는 명령어는 아래 표에서 확인할 수 있습니다.

| 명령어 | 설명                                      | 명령어   | 설명                             | 명령어 | 설명                      |
| ------ | ----------------------------------------- | -------- | -------------------------------- | ------ | ------------------------- |
| mkdir  | 새 폴더 생성                              | *        | 와일드 카드,  임이의 이름        | ls     | 파일 목록 출력            |
| rmdir  | 빈 폴더 삭제                              | man      | 명령어 매뉴얼                    | cd     | 폴더 간 이동              |
| touch  | 파일 타임스탬프 업데이트,<br/>빈 파일 생성 | /pattern | man 페이지 안에서 검색           | head   | 파일 앞 부분 출력         |
| cp A B | 파일 A를 B로 복사                         | n        | man 페이지 안에서 다음 패턴 검색 | tail   | 파일 뒷 부분 출력         |
| mv AB  | 파일 A를 B로 이동                         | --help   | 특정 명령어 도움말               | wc     | 문자/열 수 출력           |
| rm     | 파일 삭제                                 | cat      | 파일 내용 출력                   | grep   | 패턴과 매치되는 라인 출력 |
- 표 2-2 Bash 주요 명령어 

### 2.3.2 파이프와 리다이렉션

터미널의 주요기능으로는 파이프와 리다이렉션 기능이 있습니다. 파이프(Pipe)는 여러가지 명령어를 연결 시켜 실행할 수 있는 기능입니다. 파이프는 `|`기호로 나타냅니다. 예를 들어 그림 2-18에 있는 `ls | grep mse`명령어는 먼저 `ls`를 실행시켜 해당 폴더내 있는 모든 파일명을 반환하고, 반환된 파일명들을 대상으로 `grep mse`명령어가 작동해 `mse`문자가 들어간 파일명을 찾게 됩니다. 그래서 최종적으로 `mse231`파일명이 반환됩니다. 본 예시에서는 2개의 명령어만 연결했지만, 3개 이상의 명령어도 파이프를 통해 연결할 수 있습니다. 

![](https://github.com/kaggler-tv/dku-kaggle-class/blob/master/course-website/imgs/ch02-img18.JPG?raw=true)
- 그림 2-18 파이프 예제([출처](https://docs.google.com/presentation/d/1GyEc1zvn-4NsSliYf-pidnFT-9MBwePl4Gq1qn2saT4/edit#slide=id.p))

리다이렉션(Redirection)은 각종 명령어들의 출력물을 파일로 저장 하거나 다른 프로그램의 입력으로 제공할 수 있게 해주는 기능입니다. 예를 들어 앞 예시에서 확인한 `ls | grep mse`명령어의 결과물을 파일로 저장하고 싶을 때 `> list.txt`를 명령어 뒤에 추가해주면 `list.txt`파일에 `ls | grep mse`의 결과물인 `mse231`이 저장됩니다. 그림 2.X는 redirection을 활용해 현재 경로 내에 있는 모든 폴더명과 파일명을 `list.txt`파일에 저장하는 예시입니다. `ls > list.txt`를 통해 구현 가능합니다. 

### 2.3.3 유용한 프로그램 

터미널에서 사용 가능한 다양한 프로그램들에 대해 알아보도록 하겠습니다. 첫번째로 **tmux**는 원격으로 접속 시 세션을 유지해주는 기능과 화면 분할 기능을 제공합니다. [강의 영상](https://youtu.be/_jhmJQfRkpQ?t=2940)에서 화면 분할 기능 영상 demo를 확인할 수 있습니다. 

두번째로 **Vim**은 터미널에서 사용 가능한 텍스트 에디터이며 [2.2.5절](#vs-code-vim)에서 관련 내용을 확인할 수 있습니다. 

세번째로 **ssh**는 원격 접속을 할 때 사용하는 프로그램입니다. 원격 접속 뿐만 아니라 터널링 기능도 제공합니다. 원격 서버와 로컬 컴퓨터간의 터널을 뚫어서 원격 서버에서 표출되는 UI를 로컬 컴퓨터에서 확인할 수 있게 하는 기능입니다. 

네번째로 **scp**는 서버에 파일 전송을 할 때 사용하는 프로그램입니다. 

다섯번째로 **wget/curl**은 웹사이트에서 파일을 다운로드 받고자 할 때 사용하는 프로그램입니다. 

여섯번째로 **awk**은 텍스트 파일 처리시 사용합니다. 예를 들어 csv파일 내에 컬럼 개수를 확인할려고 할 때 터미널 내에서 **awk** 프로그램을 통해 확인할 수 있습니다. 

일곱번째로 **sed**는 텍스트 파일을 검색하거나 수정할 때 사용합니다. 예를 들어 특정 파일 내에서 `mse`단어를 `auc`로 모두 변경해야 할 때 **sed**를 통해 적용 가능합니다. 

터미널은 처음 배울 때 시간이 꽤 걸립니다. 하지만 하나하나씩 배우시다 보면 점차 손에 익숙해질 것이며 추후 데이터과학 대회에 참여할 때나 개발을 할 때 코딩 효율을 높여줄 것입니다. 

## 2.4 참고자료

아래에 첨부한 자료는 2장과 더불어 함께 공부하면 좋은 자료입니다. 

1. [Github 공식 가이드](https://guides.github.com/activities/hello-world/)
2. [Anaconda 공식 가이드](https://docs.anaconda.com/anaconda/install/)
3. [The Missing Semester of Your CS Education at MIT](https://missing.csail.mit.edu/)
    - [Course overview + the shell](https://missing.csail.mit.edu/2020/course-shell/)
    - [Shell Tools and Scripting](https://missing.csail.mit.edu/2020/shell-tools/)
    - [Editors (Vim)](https://missing.csail.mit.edu/2020/editors/)
    - [Data Wrangling](https://missing.csail.mit.edu/2020/data-wrangling/)
    - [Version Control (Git)](https://missing.csail.mit.edu/2020/version-control/)

3번은 MIT에서 2020년도에 진행한 수업이며 일반적으로 컴퓨터 공학 수업에서 빠진 내용들을 다루는 수업입니다. 예를 들어 Git과 Vim사용 방법들을 다룹니다. 첨부한 링크에 들어가시면 2장에서 소개한 내용들과 더불어 보다 깊이있는 내용들을 학습할 수 있습니다.