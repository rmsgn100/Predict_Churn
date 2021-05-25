# Predict Churn

어래 콜랩 링크를 클릭하시면 제가 작업한 코드를 보실 수 있습니다 : 
https://colab.research.google.com/drive/1UVgmnCvTdCCGrtj7OGwkWaNW7SBFIeXI?usp=sharing

케글에서 다운받은 데이터 :
https://www.kaggle.com/blastchar/telco-customer-churn?select=WA_Fn-UseC_-Telco-Customer-Churn.csv

<p align="center">
<img width="800" height="500" src="https://user-images.githubusercontent.com/61172021/98192944-644c2c80-1f5f-11eb-82fb-7669bdbd3fe1.png">
</p>

# 시나리오

이번에 다뤄볼 주제는 한 통신사의 고객 이탈에 관한 것입니다. 통신사 A의 고질병은 바로 줄어들지않는 고객 이탈률입니다. 이 통신사는 고객 이탈을 방지하기위해 이탈할 것 같은 고객들에게 사은품을 증정하기로 계획했습니다. 하지만, 사은품을 고객 모두에게 주는 것은 통신사에게도 큰 부담이니 최대한 이탈할것같은 고객들을 잘 찾아내고 싶어합니다. 

머신 러닝을 통해 이 통신사가 이탈고객을 찾는데에 도움이 될 수 있을까요??

# 데이터 소개 및 전처리

### 데이터
```py
print(df.shape)
print(df.head())
```
(7043, 21)
<img width="1172" alt="스크린샷 2020-11-05 오후 1 31 20" src="https://user-images.githubusercontent.com/61172021/98197952-37057b80-1f6b-11eb-96b4-bf83bcdf88bc.png">



<img align="left" width="300" height="230" src="https://user-images.githubusercontent.com/61172021/98194880-bd1dc400-1f63-11eb-8243-d67f3f704a5c.png">

타겟 설명 : 

* Churn : 이탈 여부 (Yes, No)

* 이 데이터는 고객의 정보를 바탕으로 타겟(고객이 이탈했는지 아닌지)를 보여주는 이진분류문제입니다. 

* 타겟의 분포는 약 73.5% : 26.5 % 입니다.

* 타겟의 분포가 불균형하기때문에  Accuracy 를 쓰기보단 F1 스코어를 중점적으로 보겠습니다. (Balanced 타겟은 Accuracy)

<br/>

피쳐설명 : 
* customerID : Customer ID
* gender : 남자 or 여자 (Male, Female)
* SeniorCitizen : 고령인지, 아닌지 (1, 0)
* Partner : 배우자 여부 (1, 0)
* Dependents : 부양할 가족이 여부 (1, 0)
* tenure : 서비스 이용 기간 (월)
* PhoneService : 폰 서비스 여부 (1, 0)
* MultipleLines : 여러개 라인을 쓰는지 (Yes, No, No phone service)
* InternetService : 인터넷 서비스 공급 (DSL, Fiber optic, No)
* OnlineSecurity : 온라인 보안 여부 (Yes, No, No internet service)
* OnlineBackup : 온라인 백업 여부 (Yes, No, No internet service)
* DeviceProtection : 기기 보험 여부 (Yes, No, No internet service)
* TechSupport : 기술 지원을 받는지 여부 (Yes, No, No internet service)
* StreamingTV : 스트리밍 TV 서비스 사용 여부 (Yes, No, No internet service)
* StreamingMovies : 스트리밍 영화 서비스 사용 여부 (Yes, No, No internet service)
* Contract : 계약 기간 조건 (Month-to-month, One year, Two year)
* PaperlessBilling : paperless 청구서를 받는지 (Yes, No)
* PaymentMethod : 지불 방식 (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
* MonthlyCharges : 한달 사용료
* TotalCharges : 전체 사용료


### 숫자형 피쳐들과 타겟관계 시각화
```py
# 숫자형 피쳐들과 타겟관계 시각화 (이용 기간, 월 사용료, 전체 사용료)
plt.figure(figsize=(25,7))
plt.subplot(1,3,1)
sns.histplot(df, x="tenure", hue="Churn", multiple="dodge")
plt.subplot(1,3,2)
sns.histplot(df, x="MonthlyCharges", hue="Churn", multiple="dodge")
plt.subplot(1,3,3)
sns.histplot(df, x="TotalCharges", hue="Churn", multiple="dodge")
```

<p align='center'>
<img width="535" alt="스크린샷 2020-12-11 오후 5 11 56" src="https://user-images.githubusercontent.com/61172021/101879028-fb964680-3bd3-11eb-894c-f91d52c96a14.png">
</p>

1. tenure(서비스 이용 기간(월)) : 서비스 이용 기간이 짧은 고객들의 이탈률이 높고, 오래 이용한 고객들은 이탈률이 낮습니다.

<p align='center'>
<img width="540" alt="스크린샷 2020-12-11 오후 5 09 10" src="https://user-images.githubusercontent.com/61172021/101878821-98a4af80-3bd3-11eb-9f6f-f36a9ea7bf29.png">
</p>

2. MonthlyCharges (한달 사용료) : 한달 사용료가 70 쯤일때 이탈률이 급격히 높아졌다가 점차 줄어듭니다. (아마 70~100달러 사이의 서비스가 가성비가 안좋을수도있다는 상상을 해봅니다 ㅎㅎ)

<p align='center'>
<img width="535" alt="스크린샷 2020-12-11 오후 5 10 51" src="https://user-images.githubusercontent.com/61172021/101878949-d43f7980-3bd3-11eb-9ddf-ae60df70235f.png">
</p>

3. TotalCharges (전체 사용료)가 낮은 고객들의 이탈이 가장 많은데, 이것은 아마 1번에서 말한 "서비스 이용 기간"과 연관이 있는것같습니다. ("서비스 이용 기간"이 짧은 고객들은 자연스럽게 "전체 사용료" 가 낮은게 당연해보입니다)



### 범주형 피쳐들과 타겟관계 시각화 (너무 많아서 4개만 보시겠습니다)

<img width="800" alt="스크린샷 2020-11-05 오후 1 52 33" src="https://user-images.githubusercontent.com/61172021/98199304-328e9200-1f6e-11eb-94ab-83c872529656.png">

1. DeviceProtection : 기기 보험이 없는 고객들의 이탈이 보험을 가진 고객들, 아예 인터넷이 없는 고객들보다 이탈률이 높습니다.
2. TechSupport : 기술 지원을 받지않는 고객들이 기술지원을 받는 고객들보다 이탈률이 높습니다.
3. PaperlessBilling : 종이형태로 청구서를 받는 고객들은 그렇지 않은 고객들보다 이탈률이 낮습니다. 제 생각으론, 아마 인터넷 사용이 서투신 분들이 이탈을 적게하는 것 같습니다.
4. PaymentMethod : 지불 방식이 Electronic (수동)인 고객들의 이탈률이 가장 높습니다. 그에 반해, 우편형식 또는 Electronic (자동)인 경우 이탈률이 낮습니다.

### 피쳐 엔지니어링

새로 만든 피쳐 NewBigSpender (새 고객중 이용료가 높은사람) 과 타겟의 관계 : 

<img align="left" width="300" height="230" src="https://user-images.githubusercontent.com/61172021/98319187-8e672280-2023-11eb-827e-887d43e78c9f.png">

<br/>

피쳐는 여러개를 새로 만들어봤지만 대부분 Permutation Importance 에서 낮은 값을 나타내서 한가지만 남겨놓았습니다. 위의 숫자형 데이터들은 특정 범위에서 높은 이탈률을 보였습니다. 그 중 이용 기간과 월 이용료를 이용해서 한가지 피쳐를 만들어 보았습니다. 이용기간은 6개월 이하가 이탈률이 가장높고, 월 이용료는 70~100 정도 사이가 가장 높으니 그 두 가지 모두에 해당하는 피쳐를 만들었습니다. 

<br/><br/><br/><br/>

# 모델 선택 및 성능 개선

## 1. 인코딩 방법 선택

직접매핑 vs Ordinal vs OneHot 어떤 방법을 써야할까?

이 대부분의 피쳐들이 범주형이었기에 이 두가지를 꽤 고민했습니다. 우선, 처음 제가 든 생각은 '대부분의 범주형 피쳐들이 순서를 가지면 의미가 있을 것 같다' 였습니다. 그래서 Ordinal 인코딩을 고려했지만 이것은 순서를 꼬이게 할 수 있어서 수동으로 매핑하는 작업을 했습니다. 그렇게 순서형으로 바꾼 데이터를 가지고 다양한 모델의 성능을 봤는데 원 핫인코딩을 한 데이터셋으로 예측한 결과가 더 좋길래 원 핫 인코딩을 선택했습니다. 

## 2. 다양한 모델들 비교

<img align="left" width="500" height="500" src="https://user-images.githubusercontent.com/61172021/98201629-746e0700-1f73-11eb-82d4-835cdfcb3470.png">

<br/>

    1. LogisticRegression 

    검증 f1 :  0.61

<br/>

    2. Decision Tree

    검증 f1 :  0.50
 
<br/>

    3. Random Forest

    검증 f1 :  0.56

<br/>

    4. XGBoosting

    검증 f1 :  0.61

<br/><br/>

성능이 낮게나온 Decision Tree 를 제외하고 LogisticRegression, RandomForestClassifier, XGBClassifier 이 세가지 모델을 하이퍼 파라미터 퓨닝을 통해 다시 비교해보겠습니다.

## 3. RandomizedSearchCV 를 이용한 하이퍼파라미터 튜닝

### LogisticRegression, RandomForestClassifier, XGBClassifier 세 모델 의 튜닝 후 성능 비교

1. LogisticRegression

    검증셋 F1 = 0.628

    검증셋 AUC = 0.840

2. RandomForestClassifier

    검증셋 F1 = 0.649

    검증셋 AUC = 0.845

3. XGBClassifier

    검증셋 F1 = 0.601

    검증셋 AUC = 0.833

=> RandomForestClassifier 가 F1, AUC 스코어 모두에서 다른 모델들보다 점수가 높으니 선택하겠습니다!

## 4. Permutation Importance 를 이용해 모델 복잡도 줄이기



<img align="left" width="350" height="400" src="https://user-images.githubusercontent.com/61172021/98202506-7c2eab00-1f75-11eb-9f57-9b769d92401d.png"> 
<br/><br/><br/><br/>

* 인코딩 방법으로 원핫인코딩을 선택했기 때문에 피쳐가 많아졌기 때문에 모델의 복잡도가 올라간 상태입니다. 이것은 과적합을 유발할 수도 있습니다. 

* Permutation importance 를 보니 0 이하의 값들이 꽤 많습니다 피쳐가 그 값들의 편차가 꽤 커서 플러스 값을 나타낼때도 있겠지만 모델의 복잡도를 올리는것은 좋지않습니다.

* 그리고, 성능이 비슷하다면 복잡한 모델보단 간단한 모델이 좋고, 간단한 모델이 모델학습 속도가 빠르기때문에, 덜 중요한 피쳐들을 없애고 다시 학습시키겠습니다!

<br/><br/><br/><br/><br/><br/>

```py
# 중요도가 0.0005 이상만 고르기
minimum_importance = 0.0005
mask = permuter.feature_importances_ >= minimum_importance
features_selected = X_train.columns[mask]
X_train_selected = X_train[features_selected]
X_val_selected = X_val[features_selected]
X_test_selected = X_test[features_selected]
```
```py
# 덜 중요한 특성들을 없애고 다시 학습시키기
rf1 = rf_clf.best_estimator_
rf1.fit(X_train_selected, y_train)
```
```py
# 특성 삭제 후
y_pred_rf1 = rf1.predict(X_val_selected)
print('검증 f1_score: ', f1_score(y_val, y_pred_rf1))
```
검증 f1_score:  0.655

Permutaion Importance 결과에서 0에 가깝거나 마이너스 값들을 보이는 피쳐들을 떨어뜨리고 다시 학습을 시켰습니다. 그 결과 성능에 개선이 있었어요. 

F1 스코어 0.649 => 0.655

## 5. 임계값 조절 

최적의 임계값을 적용했을때 성능 변화를 보겠습니다. 

<img width="514" alt="스크린샷 2020-11-05 오후 3 24 45" src="https://user-images.githubusercontent.com/61172021/98228539-0be75000-1f9c-11eb-8203-deed2f5d4e5d.png">

임계값 조절 후 F1 스코어 :  0.660

# 랜덤포레스트의 F1 스코어 변화 

1. 직접매핑 : 0.540

2. 원핫인코딩 : 0.560

3. 하이퍼파라미터 튜닝 : 0.649

4. Permutation Importance 을 이용해 중요한 피쳐들만 선택 : 0.655

5. 임계값 조절 : 0.660

# PDP(부분 의존도 플롯), SHAP 

### 1 피쳐 PDP 

1개의 피쳐 tenure(이용 기간) 과 타겟 (고객 이탈여부) 에 어떤 영향을 끼치는지 보겠습니다.

<img width="991" alt="스크린샷 2020-11-05 오후 4 05 48" src="https://user-images.githubusercontent.com/61172021/98208539-ca957700-1f80-11eb-93b9-457906c381cb.png">
타겟 = 고객의 이탈 여부 (1 = 이탈), tenure = 서비스 이용 기간 (월)

위에 보이는 얇은 파란색 라인들은 ICE 커브라고 불리는데, 이것은 각각 하나의 샘플을 가지고 제가 지정한 피쳐 (tenure)의 값만 바꿔가며 타겟을 예측합니다. 그리고 그 예측한 값들을 가지고 각각 선을 그리는데 그것들의 평균이 중간에있는 노란색 라인입니다!

위 그래프는 서비스 이용 기간이 고객의 이탈에 끼치는 영향을 보여주는 시각화입니다. x축은 서비스 이용기간(월)이고, y축은 타겟의 변화입니다(주의* 확률아님). 즉, 위 그림이 말하고 있는것은 '고객들의 서비스 이용 기간이 길수록 이탈률은 낮아진다' 입니다.

### 2 피쳐 PDP 

2개의 피쳐 (이용기간, 전체 사용료) 가 타겟 예측에 어떤 영향을 끼치는지 보겠습니다. 

<img width="608" alt="스크린샷 2020-11-05 오후 4 10 13" src="https://user-images.githubusercontent.com/61172021/98209024-9d959400-1f81-11eb-9017-0605c3e096d7.png">

위 그래프는 특성 2개(tenure: 서비스 이용 기간, TotalCharges: 총 이용 금액)과 타겟의 관계를 볼 수 있는 그래프입니다. 왼쪽 아래를 보시면, 이용기간이 1달이고 총 이용금액이 19.05인 고객의 이탈률은 64% 로 가장 높게 예측합니다. 그에반해, 오른쪽 상단에있는 이용기간이 72개월이고 총 이용금액이 8684.8인 고객의 이탈률은 20% 로 예측하고 있습니다. 즉, '이용기간이 짧을수록, 총 이용 금액이 낮을수록 이탈률은 높다'라고 말할 수 있겠습니다. 덧붙여 말하자면, 이용기간이 짧을수록 총 금액이 낮은것은 당연하겠죠?! ㅎㅎ

### SHAP

SHAP 은 샘플의 피쳐들이 타겟 예측에 각각 어떤 영향을 미쳤는지를 볼 수 있게 시각화 해줍니다. 

<img width="1205" alt="스크린샷 2020-11-05 오후 4 25 30" src="https://user-images.githubusercontent.com/61172021/98210211-89529680-1f83-11eb-8fe6-efde0d63c550.png">


모델은 이 고객(샘플)은 94%로 이탈할것이라는 예측을 했습니다. 

이탈 할것이라는 예측을 하게끔 만든 피쳐 top 4 :

1. tenure (서비스 이용 기간) = 1

    서비스 이용 기간이 짧은 고객들은 오래 이용한 고객들보다 이탈률이 훨씬 높습니다. 이 고객의 이용 기간이 고작 1달밖에 되지않았다는 점이 이 고객이 이탈할 것이라고 예측하는데에 크게 기여했습니다. 

2. Contract_Month-to-month (한달씩 계약) = 1 

    한 달씩 계약하는 사람들은 1년 혹은 2년씩 계약하는 사람들보다 이탈률이 굉장히 높은데요, 이 고객(샘플)은 한달씩 계약했다는 것이 이 고객이 이탈할 것이라고 예측하는데에 크게 기여했습니다. 

3. TotalCharges (전체 이용료) = 69.65

    전체 이용료가 낮을수록 고객의 이탈률은 크게 나타나는데요, 500 이하인 고객들은 이탈률이 상당히 높습니다. 이 고객은 전체 이용료가 69.65 인것이 이 고객이 이탈할 것이라고 예측하는데에 기여했습니다.

4. InternetService_Fiber optic (인터넷 공급방식) = 1

    인터넷 공급 방식중에 가장 높은 이탈률을 보이는것이 Fiber optic 입니다. 이 고객의 인터넷 공급방식도 Fiber optic 인데, 이것이 이 고객의 이탈 가능성을 높게 예측하는데에 기여했습니다.

94% 로 이탈 할 것이라고 예측을 했다면 6% 는 이탈하지 않을것이라고 예측을 했다는건데요, 여기에 기여한 피쳐들이 있겠지만, 큰 영향은 미치지 않는것으로 보입니다. 

# 마치며... 

<img align="left" width="400" height="400" src="https://user-images.githubusercontent.com/61172021/98212388-dc7a1880-1f86-11eb-8e37-558eb7908d31.png"> 
<br/>

시나리오를 다시 생각해보면, 제 시나리오의 목표는 이탈할 것으로 예상되는 고객들을 찾아내고, 그 분들이 다음계약을 진행할때 사은품을 증정함으로써 최대한 이탈을 막는 것이었습니다. 사은품을 모든 고객들에게 주면 좋겠지만 그럴 순 없으니 최대한 이탈고객을 잘 예측해야합니다. 

이 문제에서 주목해야 할 점은 두가지입니다. 

* 사은품 가격에 대한 부담 vs 고객이 이탈에 대한 리스크

이것은 Recall vs Precision 과 직결되는데요, Recall 값을 올리면 FP(이탈하지 않을 고객을 이탈한다고 예측) 이 증가할 수 있고, Precision 을 올리면 FN (이탈할 고객을 이탈하지 않는다고 예측) 이 증가할 수 있습니다. 저는 이 문제에선 Recall 값이 중요하다고 생각했습니다. Recall 값을 올리면 이탈하지 않을 고객에게도 사은품을 증정하게 되는데, 그것은 회사측에 큰 손실이 아닐뿐더러 오히려 새로운 고객유입을 유도할 수 있다고 봤기때문입니다. 

<br/>
<br/>

위 Confusion Matrix 를 보시면, 이 모델의 Precision 값은 약 0.54 이고, Recall 값은 약 0.82 입니다. Recall 값이 비교적 높게나와서 제가 원한대로 모델이 만들어 진것 같습니다! 하지만 이러한 평가지표는 문제에따라 달라질 수 있으니 유동적으로 생각해야겠습니다!

