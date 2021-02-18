---
layout: post
title:  "[ML]Bagging(+RandomForest)"
date:   2021-02-17 15:47:29 +0900
categories: ML
---



## 1. Bagging

### 1.1 Bagging이란?

[이 전 포스팅](https://choi-jun9803.github.io/project/2021/02/14/project-DACON-LG-AI-%EA%B2%BD%EC%A7%84%EB%8C%80%ED%9A%8C-%ED%9B%84%EA%B8%B0-%EB%B0%8F-%EB%B3%B5%EA%B8%B0(LightGBM%EC%9D%84-%EC%A4%91%EC%8B%AC%EC%9C%BC%EB%A1%9C).html)에서 ```Boosting```, 그 중에서도 ```LightGBM```을 다뤘었다. 설명 도중에 Bagging과 Boosting의 차이점에서 대충 설명하긴 했지만 Bagging이 정확히 무엇인지에 대해서와, 모델링에서 사용했던 ```RandomForest```에 대해서는 언급을 거의하지 않았기 때문에 이번 포스팅에서는 ```Bagging```과 ```RandomForest```에 대해서 설명을 할 것이다.



먼저, ```Bagging```은 **Bootstrap aggregating**의 약자로 반복적으로 샘플을 복원 추출하여 N개를 만든 후, N개의 모델을 학습시키고 각각의 예측값들을 평균을 내거나 투표(voting)를 통해 예측하는 모델이다.



***

### 1.2 Bagging의 수학적 원리(간단 정리)

그렇다면 ```Bagging```은 어떤 원리로 기존 방법보다 더 나은 결과를 가져오는 것일까? 

옛날에 이를 이해하기 위해서 인터넷 서칭을 해봤었는데 몇몇 인터넷 블로그에서는 데이터가 **unstable**하면 성능이 잘 안 나온다고 잘못 설명해놓고 있었다. 처음엔 말이 달라서 이해가 안되자 대충 넘어갔는데 나중에 좀 더 공부하면서 논문을 읽다보니 반대로 데이터가 **unstable**할수록 성능이 잘 나온다고 설명되어 있었다.



***

그렇다면 도대체 어떤 원리로? 간단하게 설명하자면 **Jensen's inequality**를 이용해서 MSE를 줄일 수 있음을 증명할 수 있다.

[참고 논문 - Breiman l - Bagging Predictors, 1996](https://link.springer.com/article/10.1023/A:1018054314350)

~~타이포라로 수식쓰는건 너무 귀찮으니 논문에 필기한걸 캡쳐하겠다 ㅎㅎ~~

