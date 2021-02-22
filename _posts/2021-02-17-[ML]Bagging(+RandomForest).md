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

