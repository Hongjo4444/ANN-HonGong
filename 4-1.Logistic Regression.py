import pandas as pd #판다스로 웹상의 데이터 읽기
#왼쪽 0,1,2...은 행번호=판다스의 인덱스, 맨위의 Species,Weight...은 열제목=판다스 CSV파일의 첫줄(판다스가 알아서 만들어줌)
fish=pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head())
fish_input=fish[['Weight','Length','Diagonal','Height','Width']].to_numpy() #특성
fish_target=fish['Species'].to_numpy() #여기서는 생선 종류가 타깃

from sklearn.model_selection import train_test_split #훈련세트, 테스트세트로 나누기
train_input,test_input,train_target,test_target=train_test_split(fish_input,fish_target,random_state=42)

from sklearn.preprocessing import StandardScaler #특성을 스케일에 맞게 데이터 전처리
ss=StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier(n_neighbors=3) #k-최근접 이웃의 다중분류
kn.fit(train_scaled,train_target) #훈련
print(kn.score(train_scaled,train_target))
print(kn.score(test_scaled,test_target))
print(kn.classes_) #k-최근접 이웃에서 정렬된 타깃값(classes_에 저장된 것)
print(kn.predict(test_scaled[:5])) #테스트 세트의 처음 5개 샘플 예측
print(kn.predict_proba(test_scaled[:5])) #각 행(샘플)마다 열(클래스)에 대한 확률 나타남 >> 네번째 샘플은 Perch일 확률 0.6667, Pike일 확률 0.3333
import numpy as np #decimals을 이용한 확률의 소수점 4번째 자리까지 표기
proba=kn.predict_proba(test_scaled[:5]) #먼저 테스트 세트의 처음 5개 샘플에 대한 확률 출력
print(np.round(proba,decimals=4))

#로지스틱 회귀 이진 분류(2가지 경우(음성(0),양성(1))에 대해서만 분류), 이진분류는 로지스틱 함수(=시그모이드 함수) 이용해 확률 출력
bream_smelt_indexes=(train_target=='Bream')|(train_target=='Smelt') #불리언 인덱싱을 사용해 도미,빙어의 데이터만 뽑기
train_bream_smelt=train_scaled[bream_smelt_indexes]
target_bream_smelt=train_target[bream_smelt_indexes]
from sklearn.linear_model import LogisticRegression #로지스틱 회귀는 선형회귀와 같이 선형모델 아래에 있다.
lr=LogisticRegression()
lr.fit(train_bream_smelt,target_bream_smelt)
print(lr.predict(train_bream_smelt[:5])) #처음 5개에 대한 예측
print(lr.predict_proba(train_bream_smelt[:5])) #처음 5개에 대한 로지스틱 회귀 이진분류 확률 출력
print(lr.coef_,lr.intercept_) #로지스틱 회귀 계수 확인(특성이 5개니까 coef(계수)도 5개 나옴), intercept는 절편

decisions=lr.decision_function(train_bream_smelt[:5]) #처음 5개 샘플의 z값 출력
print(decisions)
from scipy.special import expit #양성값에 대한 z값을 시그노이드 함수에 넣어서 확률값 출력
print(expit(decisions))

#로지스틱 회귀 다중 분류(여러개의 클래스가 있는 경우), 다중분류는 소프트맥스 함수 이용해 확률 출력
lr=LogisticRegression(C=20,max_iter=1000)
#C값 조정으로 노름 규제 조절(기본값은 1, C 커지면 규제 약해짐(복잡해짐), C 작아지면 규제 강해짐), max_iter(반복횟수)를 늘려서 반복횟수 오류 방지
lr.fit(train_scaled,train_target) #로지스틱 회귀 클래스로 다중 분류 모델 훈련
print(lr.score(train_scaled,train_target)) #훈련 정확도
print(lr.score(test_scaled,test_target)) #테스트 정확도
print(lr.predict(test_scaled[:5])) #처음 5개에 대한 예측
proba=lr.predict_proba(test_scaled[:5]) #테스트 세트의 처음 5개 샘플에 대한 확률 출력
print(np.round(proba,decimals=3))
print(lr.coef_.shape,lr.intercept_.shape) #5개 샘플의 7개 클래스에 대한 계수 7 세트(7,5 사이즈), 절편 7개(7,0 사이즈) 출력

decision=lr.decision_function(test_scaled[:5]) #처음 5개 샘플의 z값 출력
print(np.round(decision,decimals=2))
from scipy.special import softmax #z값을 소프트맥스 함수에 넣어서 확률값 출력
proba=softmax(decision,axis=1) #softmax()의 axis로 소프트맥스를 계산할 축 지정(axis=1이면 각 샘플에 대해 소프트맥스 계산, 지정 안하면 배열 전체에 대해 계산)
print(np.round(proba,decimals=3))