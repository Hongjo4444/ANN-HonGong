import numpy as np
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target=train_test_split(perch_length,perch_weight,random_state=42)
train_input=train_input.reshape(-1,1)
test_input=test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor
knr=KNeighborsRegressor(n_neighbors=3)
knr.fit(train_input,train_target)

print(knr.predict([[50]])) #K최근접이웃은 훈련세트 밖의 값을 예측하기 어려움(농어 크기가 엄청 크더라도 훈련세트의 무게까지만 예측 가능)
import matplotlib.pyplot as plt
distances,indexes=knr.kneighbors([[50]])
plt.scatter(train_input,train_target) #훈련세트의 산점도
plt.scatter(train_input[indexes],train_target[indexes],marker='D') #훈련세트 중 이웃 샘플의 산점도
plt.scatter(50,1033,marker='^') #50cm농어 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(knr.predict([[100]])) #K최근접이웃은 훈련세트 밖의 값을 예측하기 어려움(농어 크기가 엄청 크더라도 훈련세트의 무게까지만 예측 가능)
import matplotlib.pyplot as plt
distances,indexes=knr.kneighbors([[100]])
plt.scatter(train_input,train_target) #훈련세트의 산점도
plt.scatter(train_input[indexes],train_target[indexes],marker='D') #훈련세트 중 이웃 샘플의 산점도
plt.scatter(100,1033,marker='^') #100cm농어 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.linear_model import LinearRegression
lr=LinearRegression() #선형 회귀 사용
lr.fit(train_input,train_target)
print(lr.predict([[50]])) #선형 회귀 적용결과 50cm 농어의 무게 1241.83으로 예측
print(lr.coef_,lr.intercept_) #선형 회귀 공식 y=ax+b에서 a(기울기):coef,b(절편):intercept에 저장(y는 무게,x는 길이)

plt.scatter(train_input,train_target) #훈련 세트의 산점도
plt.plot([15,50],[15*lr.coef_+lr.intercept_,50*lr.coef_+lr.intercept_]) #15~50의 1차 방정식 그래프
plt.scatter(50,1241.8,marker='^')
plt.show()
print(lr.score(train_input,train_target)) #선형 회귀를 사용한 훈련 세트의 R^2 점수
print(lr.score(test_input,test_target)) #선형 회귀를 사용한 테스트 세트의 R^2 점수, 테스트 세트의 점수가 너무 낮으므로 과소적합 예상

train_poly=np.column_stack((train_input**2,train_input))
test_poly=np.column_stack((test_input**2,test_input))
#다항 회귀(2차 함수 이상의 다항식) 사용(선형 회귀 사용시 길이가 아주 작을경우 무게를 -로 예측할 수 있으므로)을 위한 길이^2 만들기

lr.fit(train_poly,train_target) #다항 회귀로 재훈련
print(lr.predict([[50**2,50]]))
print(lr.coef_,lr.intercept_) #다항 회귀 공식 y=ax^2+bx+c에서 a,b(기울기):coef,c(절편):intercept에 저장(y는 무게,x는 길이)

point=np.arange(15,50) #구간별 직선 그려서 다항 회귀 곡선 그리기
plt.scatter(train_input,train_target) #훈련 세트의 산점도 그리기
plt.scatter(50,1574,marker='^')
plt.plot(point,1.01*point**2-21.6*point+116.05) #구간별 직선 그리기
plt.xlabel('length')
plt.ylabel('weight')
plt.show()