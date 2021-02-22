---
layout: post
title:  "[ML]Bagging(+RandomForest)"
date:   2021-02-17 15:47:29 +0900
categories: ML
---



## 1. Bagging

### 1.1 Bagging이란?

[이 전 포스팅](https://choi-jun9803.github.io/project/2021/02/14/project-DACON-LG-AI-%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C-%ED%9B%84%EA%B8%B0-%EB%B0%8F-%EB%B3%B5%EA%B8%B0(LightGBM%EC%9D%84-%EC%A4%91%EC%8B%AC%EC%9C%BC%EB%A1%9C).html)에서 ```Boosting```, 그 중에서도 ```LightGBM```을 다뤘었다. 설명 도중에 Bagging과 Boosting의 차이점에서 대충 설명하긴 했지만 Bagging이 정확히 무엇인지에 대해서와, 모델링에서 사용했던 ```RandomForest```에 대해서는 언급을 거의하지 않았기 때문에 이번 포스팅에서는 ```Bagging```과 ```RandomForest```에 대해서 설명을 할 것이다.   
&nbsp;

먼저, ```Bagging```은 **Bootstrap aggregating**의 약자로 반복적으로 샘플을 랜덤으로 복원 추출하여 N개를 만든 후, N개의 모델을 학습시키고 각각의 예측값들을 평균(회귀 문제의 경우)을 내거나 투표(분류 문제의 경우)를 통해 예측하는 모델이다.

![Bagging_algorithm](https://user-images.githubusercontent.com/64791442/108373527-563ac700-7243-11eb-8677-637e44eb86de.jpg)

위 사진은 앙상블을 평균으로 처리한 것으로 봐서 분류가 아닌 regressor의 경우이다. 생각보다 간단하죠?



### 1.2 Bagging의 수학적 원리(간단 정리)

그렇다면 ```Bagging```은 어떤 원리로 기존 방법보다 더 나은 결과를 가져오는 것일까?    

옛날에 이를 이해하기 위해서 인터넷 서칭을 해봤었는데 몇몇 인터넷 블로그에서는 데이터가 **unstable**하면 성능이 잘 안 나온다고 잘못 설명해놓고 있었다. 처음엔 말이 달라서 이해가 안되자 대충 넘어갔는데 나중에 좀 더 공부하면서 논문을 읽다보니 반대로 데이터가 **unstable**할수록 성능이 잘 나온다고 설명되어 있었다.

&nbsp;

그렇다면 도대체 어떤 원리로? 간단하게 설명하자면 **Jensen's inequality**를 이용해서 MSE를 줄일 수 있음을 증명할 수 있다.

[참고 논문 - Breiman l - Bagging Predictors, 1996](https://link.springer.com/article/10.1023/A:1018054314350)

~~타이포라로 수식쓰는건 너무 귀찮으니 논문에 필기한걸 캡쳐하겠다 ㅎㅎ~~

![Bagging](https://user-images.githubusercontent.com/64791442/108373168-eaf0f500-7242-11eb-8a59-024b69f03906.jpg)

###### (L같은건 bootstrap했을 때의 샘플 하나라고 보면 된다.)

```Bagging```이 아닌 ```bootstrap```한 것들을 학습한 모델들의 값에 대한 ```error term```의 평균을 표현하면 식 4.1이 나온다. 그런데 이 식에서 ```Jensen's inequality```를 이용해서 마지막 요소의 ```Expectation```의 위치를 바꾼다면?? 이때 식에서의 f(x)가 ```convex```인 x^2이니 식 4.2가 성립한다. (왼쪽 필기 참조)   
&nbsp;

이렇게 ```bootstrap```한 샘플들의 분류기를 앙상블(평균처리)해서 모델을 작업하는 이유가 수학적으로 증명되었다. 그렇다면 어떤 데이터에서 ```Bagging```이 잘 통할까?? 첨부한 사진 맨 아랫 줄을 보면 (E(F))^2과 E(F^2)의 차이의 수준이 성능을 판가름한다고 써있다. 이것이 바로 앞서 말했던 ```unstable```한 데이터일수록 ```Bagging```의 성능이 증가하는 이유다. 

&nbsp;

![Bagging의 수학적 이유](https://user-images.githubusercontent.com/64791442/108666023-8de29100-7519-11eb-8881-ddddd226e899.jpg)

논문에서 수학적으로 설명되어 있는 부분을 내 방식으로 풀어봤다. 정리하면 data가 ```unstable```해야지 ```Bagging```이 아닌 것과 ```Bagging```의 차이가 커지므로 이 알고리즘의 의미가 있는 것이다.

&nbsp;

## 2. RandomForest

### 2.1 RandomForest란?

```RandomForest```, 진짜 데이터 사이언스 쪽에 관심 있는 사람이라면 이 이름을 한 번 정도는 들어봤을 것이다. 그런데 이것이 정확히 무엇인지 잘 모르면서 사용하는 사람이 가끔 있는 것 같다. 학회 운영진을 하면서 면접을 봤는데 한 분이 자신의 프로젝트를 설명하면서 이 기법을 사용했더니 성능이 얼만큼 늘었다~ 이런 식으로 말한 적이 있었다. 그래서 ```RandomForest```가 어떤 것인지 간단하게 설명해달라고 요청했더니 잘 모른다고 답했다... 기법을 사용해보기만 한 것은 의미가 없다. 그건 그 날 처음 사용해 본 사람도 알 수 있는 일이다. Ctrl+C Ctrl+V만 하면 되니까 ㅎㅎ&nbsp;

그렇다면 ```RandomForest```는 어떤 것이고, 어떤 특징을 가지고 있을까?



일단 ```RandomForest```은 이름에서 알 수 있듯이(?) ```tree```기반 모델인데 ```Bagging```과 아주 큰 차이점이 있다. ```Bagging```은 데이터를 랜덤하게 복원추출함으로써 사용한다. ```feature```는 전부 사용한다. 그러나 ```RandomForest```는 데이터는 전부 사용하나 ```feature```의 개수를 랜덤하게 추출해서 사용한다. (주로 추출 변수 개수는 전체 변수 개수의 제곱근을 많이 사용한다.) &nbsp;

즉, 랜덤한 변수들만 골라서 ```tree```모델을 여러 개 만들어서 앙상블하는 방법인 것이다.

이렇게 조금 다른 방법들로 인해서 하이퍼파라미터의 종류도 조금 다르다.

### 2.2 RandomForest의 Hyperparameter

- **n_estimators** : 트리의 개수, default는 100, 많을수록 좋은 성능이 나오는 것은 아니다.
- **max_features** : 데이터의 feature를 참조할 개수, default는 auto
- **max_depth** : 트리의 깊이
- **min_sample_leaf** : 리프노드가 되기 위한 최소한의 샘플 데이터 수
- **min_samples_split** : 노드를 분할하기 위한 최소한의 데이터 수

[참조](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

