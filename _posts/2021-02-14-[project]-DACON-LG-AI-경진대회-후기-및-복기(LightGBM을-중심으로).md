---
layout: post
title:  "[project]DACON LG AI 경진대회 후기 및 복기(LightGBM을 중심으로)"
date:   2021-02-14 16:27:29 +0900
categories: project
---

2021년 2월 3일에 끝난 DACON LG AI 시스템 품질 변화로 인한 사용자 불편 예지 AI 경진대회[DACON LG AI](https://dacon.io/competitions/official/235687/overview/ )에 참가했었습니다.

처음 출전하는 대회라서 많은 시행착오도 겪고 뻘짓(?)도 많이 했는데 나름 배운 것도 많고 얻어가는게 많아서 이렇게 블로그에 포스팅하면서 글로 정리해보려 합니다. 😀



먼저 대회에 대한 설명을 해보자면 "비식별화 된 시스템 기록(로그 및 수치 데이터)을 분석하여 시스템 품질 변화로 사용자에게 불편을 야기하는 요인을 진단"하는 대회로, 퀄리티 로그 데이터와 에러 로그 데이터를 이용하여 고객들이 불편을 제기할지 안 할지 예측을 하는 예측 모델을 만드는 것이다.



이 대회에서 나는 **LightGBM, Randomforest, LogisticRegression**을 이용하여 모델 3개를 만들고 각각의 예측 값들을 stacking하여 다시 LightGBM 모델에 학습시켰고 이때 하이퍼 파라미터를 조정하기 위해 **Bayesian optimaiztion**을 이용하여 최적의 하이퍼 파라미터 값들을 찾았다. (이 글에서는 LightGBM을 중심으로 설명하기 때문에 이론에 대한 설명은 LightGBM만 할 것이다. 추후 포스팅에 bayesian optimaization에 대해 업로드 할 예정)



## 1. 아쉬웠던 점

이 대회에서는 처음치고는 나쁘지 않은 성적을 거뒀는데 ~~(리더보드 첫 페이지에는 있었다 ㅎㅎ)~~ 그래도 입상을 하지 못해서 좀 아쉬웠다. 만약 다음에 공모전을 나가게 된다거나 나 스스로의 성장을 위해서 피드백을 해보려 한다.

   

아쉬웠던 점:



1. 전처리를 온전하게 하지 못했다.
2. 시간 부족으로 모델링에 많은 시간을 투자하지 못했다. 

먼저 첫 번째 아쉬웠던 점은 전처리의 문제이다. 대회 컨셉 자체가 비식별화라서 그런지 데이터에 대한 설명 자체가 없었다. 



예를 들어 ```err_code```같은 경우 ```string``` 값들이 1/5 정도 됐다. 그런데 문자들이 의미를 알 수 없는 문자들이 (예를 들어 'U-NA21416' <- 실제로 이 값이 있었다는건 아니고 예시이다.) 있어서 데이터를 충분히 해석할 수 없었다. 



또한, 로그 데이터의 특성인지는 모르겠지만 값들 중 데이터 타입이 ```int```,```float```인 것들도 양적비교가 가능한 것이 아닌, 하나의 코드로 적용되는 것이기 때문에 전부 카테고리화해서 변수를 적용했어야 했다. 그러다 보니 ```train set```과 ```test set```에 다른 값들이 존재하게 되면 중복되는 값들만 사용할 수 있었다. 만약 해당 데이터에 대한 도메인 지식이나 전처리에 시간을 좀 더 들였다면 위와 같은 데이터와 결측치 전부 활용할 수 있었을거 같은데 조금 아쉬웠다.



두 번째 아쉬웠던 점은 모델링에 많은 시간을 투자하지 못했다는 점이다. 아마 해당 대회 참가자들도 데이터에 대한 지식이 없었을텐데 왜 점수가 높은 사람은 높았느냐? 하고 묻는다면 전처리 문제도 있지만 가장 먼저 떠오르는건 모델링에 투입한 시간이 다르기 때문에 결과가 달라지지 않았을까 하는 생각이다. ~~그래도 1등이랑 0.02점 정도 밖에 차이 안 났다 ㅎ~~ 



## 2. 모델링

내가 사용한 모델링은 ```LightGBM```이었다. ```LightGBM```이 빠르고 정확도가 높은 알고리즘이기에 후진 스펙의 노트북으로 참여한 내 입장에서는 최적의 알고리즘이었다. 물론 ```RandomForest```,```LogisticRegression```도 ```stacking```을 사용할 때 썼지만 ```stacking```해서 얻은 데이터를 최종적으로 fitting한 모델은 ```LightGBM```이었으므로 본 포스팅에서는 ```LightGBM```을 중심으로 적으려 한다.  

```LightGBM```에 대해서 설명하기 전에 먼저 ```boosting``` 알고리즘에 대해서 설명을 해야 한다. 

### 2.1 Boosting

```boosting``` 알고리즘은 순차적인 방법으로 다수의 분류기를 생성하는 기법이다. 이전 분류기의 학습 결과를 반영하지 않는 ```bagging```과 달리 ```boosting```은 이전 분류기의 학습 결과를 토대로 다음 분류기의 학습 데이터의 샘플 가중치를 조정해 학습하는 방법이다. 

![Bosting](https://user-images.githubusercontent.com/64791442/107853649-8d336600-6e5a-11eb-9a72-bff9e27a210c.png)

이렇게 만든 여러개의 분류기들을 앙상블해서 예측을 하는 모델인 것이다. 

>  "그냥 마지막에 나온 분류기가 가장 적절한 분류기일테니까 그 분류기만 쓰면 되지 않나? "

하는 생각이 있을 수도 있지만, 

하나의 강력한 분류기가 아닌 'weak'한 분류기 여러개를 앙상블하는 것이 더 효율적이라는 생각에 기반하여 생성한 모든 분류기들로 ```fitting```을 진행한다. 

***

### 2.2 LightGBM

그 중에서도 우리가 사용한 모델은 ```LightGBM```인데 ```GradientBoosting```의 한 종류로 ```tree```기반 학습 알고리즘이다.

```GradientBoosting```은 쉽게 말해서 이전 분류기의 ```residual``` 에 ```fitting```하는 개념으로, 목적함수```loss function```을 미분하여 함수값을 최소로 하는 가중치를 ```gradient```로 찾아 학습시킨 분류기에서 발생한```residual```을 다음 분류기에서 다시 ```fitting``` 하면서 반복하는 것이다. 분류기 1을 통해 예측하고 남은 잔차를 분류기 2을 통해 예측하고, 또 거기서 나온 잔차를 다음 분류기가 예측하고,,,,



이런 ```GradientBoosting```에는 여러 가지 기법들이 있는데 그 중 ```LightGBM```은 ```GradientBoosting``` 에 ```tree``` 기반으로 적용하는 기법이다. 

![LightGBM_leaf_wise](https://user-images.githubusercontent.com/64791442/107749601-35173980-6d5e-11eb-8201-d28c87d604f9.png)

```GradienBoosting```을 ```tree```기반으로 하는 다른 알고리즘과의 차이점은 **leaf-wise**기반이라는 점이다. leaf-wise기반은 최대 손실 값(```loss```)을 가지는 리프 노드를 계속해서 분할하여 ```loss```를 줄여 가는 방식이다. 다른 ```tree```기반 ```Boosting``` 알고리즘은 ```Overfitting```을 방지하기 위해서 ```tree```를 대칭으로 만들면서 분류하는 **level-wise**기반이다. 

![other_Boosint_level_wise](https://user-images.githubusercontent.com/64791442/107749747-67c13200-6d5e-11eb-9f15-e5053006d3a8.png)

**leaf-wise**는 계산 속도가 빠르고 효율적이지만 ```overfitting```에 취약하다. 그렇기 때문에 10,000개 이상의 데이터 셋에 사용되는 것이 적절하다. 반면, **level-wise**는 비교적 ```overfitting```에 강하지만 계산 속도가 느리고 비효율적이다. 

```python
# Train
#-------------------------------------------------------------------------------------
# validation auc score를 확인하기 위해 정의
def f_pr_auc(probas_pred, y_true):
    labels=y_true.get_label()
    p, r, _ = precision_recall_curve(labels, probas_pred)
    score=auc(r,p) 
    return "pr_auc", score, True
#-------------------------------------------------------------------------------------
models     = []
recalls    = []
precisions = []
auc_scores   = []
threshold = 0.5
# 파라미터 설정
parameters =   {
                'boosting_type' : 'dart',
                'objective'     : 'binary',
                'metric'        : 'auc',
                'seed': 1015,
                'learning_rate' : params['learning_rate'],
                'max_depth': int(params['max_depth']),
                'num_leaves': int(params['num_leaves']),
                'bagging_fraction': params['bagging_fraction'],
                'feature_fraction': params['feature_fraction'],
                'reg_alpha' : params['reg_alpha'], 
                'reg_lambda' : params['reg_lambda']
                }
#-------------------------------------------------------------------------------------
# 5 Kfold cross validation
k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in k_fold.split(train_x):

    # split train, validation set
    X = train_x[train_idx]
    y = train_y[train_idx]
    valid_x = train_x[val_idx]
    valid_y = train_y[val_idx]

    d_train= lgb.Dataset(X, y)
    d_val  = lgb.Dataset(valid_x, valid_y)

    #run traning
    model = lgb.train(
                        parameters,
                        train_set       = d_train,
                        num_boost_round = int(params['num_boost_round']),
                        valid_sets      = d_val,
                        feval           = f_pr_auc,
                        verbose_eval    = 20, 
                        early_stopping_rounds = 3
                       )

    # cal valid prediction
    valid_prob = model.predict(valid_x)
    valid_pred = np.where(valid_prob > threshold, 1, 0)

    # cal scores
    recall    = recall_score(    valid_y, valid_pred)
    precision = precision_score( valid_y, valid_pred)
    auc_score = roc_auc_score(   valid_y, valid_prob)

    # append scores
    models.append(model)
    recalls.append(recall)
    precisions.append(precision)
    auc_scores.append(auc_score)

    print('==========================================================')
```

- 대회에서 점수 산정을 auc-score를 기준으로 했다.
- ```hyper parameter tunning``` 은 ```BayesianOptimization```을 통해 최적화 시킨 값들을 사용했다. (추후에 ```bayesian optimization```에 대해 업로드 예정+ 하이퍼 파라미터 중```dart```에 대해서, ```dropout```에 대해서도 업로드 예정)



## 3. 피드백

피드백을 하기 전에 당시 대회 데이터에 대한 설명이 필요할 것 같다.

### 3.1 데이터에 대한 간략한 설명 및 input_data생성 코드

데이터는 ```train_err```,```train_quality```,```train_problem```, ```test_err```, ```test_quality```로 나뉘어져 있었다. 데이터 이름만 봐도 알 수 있듯이, problem데이터가 우리가 예측하고 싶은 고객 불만 데이터이고 그 외의 데이터가 예측변수로 에러 로그, 퀄리티 로그 데이터이다.

앞서 말했듯이 모든 데이터는 양적 비교가 불가능한 코드화된 데이터이고 categorize과정이 필요한 데이터였다. 그래서 두 예측 변수 데이터를 하나의 ```input_data```로 변환하는 작업이 필요했고, ```string```데이터 타입이 없는 변수들만 골라서 ```categorize```시켜서 ```input_data```를 만들었다.

```pyhon
input_data = np.zeros((train_user_number,431))

id_error = train_err[['user_id','errtype']].values

for person_idx, err in tqdm(id_error):
    # person_idx - train_user_id_min 위치에 person_idx, errtype에 해당하는 error값을 +1
    input_data[person_idx - train_user_id_min, err-1 ] += 1

#input data에 변수 추가(방식은 위와 같음)    
id_model = train_err[['user_id','model_nm']].values

for person_idx, model in tqdm(id_model):
    # person_idx - train_user_id_min 위치에 person_idx, model_nm+42에 해당하는 error값을 +1 (model은 0부터 존재하므로)
    input_data[person_idx - train_user_id_min, model+42 ] += 1
    
id_quality_7 = train_quality[['user_id','quality_7']].values

for person_idx, quality in tqdm(id_quality_7):
    # person_idx - train_user_id_min 위치에 person_idx, quality에 해당하는 값을 +1 
    input_data[person_idx - train_user_id_min, quality+52 ] += 1
    
input_data.shape
```

또한 ```train```데이터와 ```test```데이터 간의 차이점도 존재했다. 사용했던 변수 중 fwver과 quality변수는 ```train```과 ```test```간 중복되지 않은 unique한 값들이 존재했다. 

***

### 3.2 전처리 복기

이런 식으로 인덱싱을 이용하여 ```user_id```별로 각 변수에 +1하는 방식으로 ```input_data```를 만들었다. 그러다보니 ```input_data```의 사이즈가 15,000*431정도 밖에 되지 않았다. (나중에 feature selection을 이용하여 의미있는 변수만 선택해서 적용시켰다.) 학습 데이터는 각각 1600만 행 정도가 되었는데 방대한 데이터를 1만5천행의 데이터로 줄이다보니 시계열적 정보가 사라졌다.

만약 **시계열 데이터를 이용했다면 좀 더 데이터가 많아졌을 것이고, 로그 데이터에서 의미가 있는 시간에 대해서도 반영할 수 있었을 것**이라는 아쉬움이 남는다. 2차원 array가 아닌 3차원 array로 구성을 했다면 시간 변수를 온전히 반영할 수 있었을테고, 더 높은 정확도를 가졌겠지만, 경험 부족으로 3차원 인풋 데이터를 2차원 아웃풋 데이터로 만들어 내는 코드를 구현하지 못했다.

그리고, 중복되지 않은 unique한 값들은 전부 하나의 값으로 통일 시킨 다음에 **중복되는 값들 여러개 + 중복되지 않는 값 하나**로 구성하여 ```categorize```시켰다. 

여기서 아쉬웠던 점이 있었는데, 중복되지 않는 값들도 군집화를 통해 하나의 값이 아닌 다양한 값들로 반영을 했더라면 어땠을까 하는 생각이 있다. 시간 부족 때문에 다양한 방법을 시도해보지 못했는데, 만약  ```train```데이터와 ```test```데이터 간의 명목형 변수의 값에 차이가 있다면 중복되지 않는 값들을 하나의 유형으로 퉁쳐버리는 것이 아닌, 적절한 유형으로 군집화해봐야겠다.