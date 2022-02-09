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

import matplotlib.pyplot as plt
plt.scatter(perch_length,perch_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

from sklearn.model_selection import train_test_split #훈련 세트 준비
train_input,test_input,train_target,test_target=train_test_split(perch_length,perch_weight,random_state=42) #회귀는 타겟값이 임의의 수이므로 stratify 설정 안한다.

train_input=train_input.reshape(-1,1) #사이킷런 사용을 위한 행렬 변형(행-1:나머지 차원을 결정하고 남은 차원을 사용하겠다, 열1:열에 1차원 적용)
test_input=test_input.reshape(-1,1)

from sklearn.neighbors import KNeighborsRegressor #k-최근접이웃 회귀(타겟값을 고르는 것이 아닌 임의의 수 예측) 사용
knr=KNeighborsRegressor()
knr.fit(train_input,train_target) #훈련
print(knr.score(test_input,test_target)) #테스트

from sklearn.metrics import mean_absolute_error #타깃과 예측의 절댓값 오차를 평균하여 반환
test_prediction=knr.predict(test_input)
mae=mean_absolute_error(test_target,test_prediction)
print(mae) #농어의 예측무게의 절댓값 오차는 19.15...정도 난다는 말(높은지,낮은지는 모름)

print(knr.score(train_input,train_target))
print(knr.score(test_input,test_target)) #과소적합(일반적으로 훈련세트의 정확도가 더 높아야하는데 그렇지 않은 경우), 과대적합:훈련세트의 정확도가 월등히 높은 경우

knr.n_neighbors=3 #이웃의 개수 줄여서 과소적합 해결
knr.fit(train_input,train_target)
print(knr.score(train_input,train_target))
print(knr.score(test_input,test_target))