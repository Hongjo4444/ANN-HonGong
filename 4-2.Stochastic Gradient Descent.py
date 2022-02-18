#확률적 경사 하강법(SGD)은 머신러닝 알고리즘이 아님, 훈련방법임
#확률적(랜덤 갯수)으로 샘플을 꺼내서 훈련하는 과정을 여러번 반복해서 가장 좋은 손실 함수(=경사, 나쁜 정도를 측정하는 함수)를 최적화 하는 과정

import pandas as pd #판다스 데이터프레임 만들기
fish=pd.read_csv('https://bit.ly/fish_csv_data')
fish_input=fish[['Weight','Length','Diagonal','Height','Width']].to_numpy() #입력 데이터
fish_target=fish['Species'].to_numpy() #타겟 데이터

from sklearn.model_selection import train_test_split #데이터를 훈련 세트, 테스트 세트로 나누기
train_input,test_input,train_target,test_target=train_test_split(fish_input,fish_target,random_state=42)

from sklearn.preprocessing import StandardScaler #가장 좋은 손실 함수(=경사)를 찾을때는 표준화 전처리로 스케일을 같게 해줘야함
ss=StandardScaler()
ss.fit(train_input) #훈련 세트에서 학습한 통계값으로 테스트 세트도 변환해야 함.
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

from sklearn.linear_model import SGDClassifier #분류 모델 확률적 경사 하강법(회귀 모델일 경우 SGDRegressor)
#SGDClassifier는 미니배치 경사 하강법, 배치 하강법 제공하지 않음
sc=SGDClassifier(loss='log',max_iter=10,random_state=42) #분류는 loss='log'로 로지스틱 손실함수를 추적 설정
#이진 분류:로지스틱 회귀or크로스엔트로피 손실함수, 다중 분류:크로스엔트로피 손실함수, 회귀:평균 절댓값 오차or평균 제곱 오차 손실함수 사용)
#max_iter로 에포크 설정(에포크가 너무 크면 과대적합, 에포크가 너무 작으면 과소적합)
sc.fit(train_scaled,train_target) #훈련
print(sc.score(train_scaled,train_target)) #정확도 출력(훈련 모델이므로)
print(sc.score(test_scaled,test_target))
sc.partial_fit(train_scaled,train_target) #partial_fit으로 기존에 훈련한 가중치,절편을 가지고 한번 더 훈련(그냥 fit하면 기존에 학습한 가중치,절편 버리고 훈련)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))

import numpy as np #조기종료:최적 에포크까지만 훈련하는 방법
sc=SGDClassifier(loss='log',random_state=42)
train_score=[]
test_score=[]
classes=np.unique(train_target)
for _ in range(0,300):
    sc.partial_fit(train_scaled,train_target,classes=classes) #partial_fit는 데이터 일부만 전달하므로 classes 매서드로 나올 수 있는 클래스 목록을 전달
    train_score.append(sc.score(train_scaled,train_target))
    test_score.append(sc.score(test_scaled,test_target))
import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show() #epoch 100일때가 최적으로 보임
sc=SGDClassifier(loss='log',max_iter=100,tol=None,random_state=42) #max_iter로 epoch 100까지만 훈련
#원래 SGDClassifier는 일정 에포크 동안 성능이 향상되지 않으면 더 훈련하지 않고 자동으로 종료, tol=None으로 자동으로 멈추지 않고 max_iter만큼 무조건 반복하게 설정
sc.fit(train_scaled,train_target)
print(sc.score(train_scaled,train_target))
print(sc.score(test_scaled,test_target))