import pandas as pd #판다스로 원격에 있는 데이터 정리(농어의 길이,높이,두께 데이터 받아옴)
df=pd.read_csv('https://bit.ly/perch_csv')
perch_full=df.to_numpy()
print(perch_full)

import numpy as np #타깃 데이터 준비
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target=train_test_split(perch_full,perch_weight,random_state=42)

from sklearn.preprocessing import PolynomialFeatures #다항특성 만들기
poly=PolynomialFeatures(include_bias=False) #bias를 False로 하면 1(절편을 위한 특성) 제외
poly.fit(train_input)
train_poly=poly.transform(train_input)
print(train_poly.shape) #42개의 행, 9개의 특성 만들어짐
print(poly.get_feature_names()) #9개의 특성이 어떻게 만들어졌는지
test_poly=poly.transform(test_input)

from sklearn.linear_model import LinearRegression #선형 회귀 실행
lr=LinearRegression()
lr.fit(train_poly,train_target) #타겟은 맞춰야하는 값이기 때문에 변하지 않음
print(lr.score(train_poly,train_target)) #특성을 이전보다 많이(9개) 넣어서 값이 스코어 값이 올라감
print(lr.score(test_poly,test_target))

poly=PolynomialFeatures(degree=5,include_bias=False) #더 많은 특성 만들기(degree(차수)의 기본값은 2이지만 더 복잡하게 5로 설정)
poly.fit(train_input)
train_poly=poly.transform(train_input)
print(train_poly.shape) #42개의 행, 55개의 특성 만들어짐
test_poly=poly.transform(test_input)
lr.fit(train_poly,train_target) #선형 회귀 재실행
print(lr.score(train_poly,train_target))
print(lr.score(test_poly,test_target)) #훈련데이터가 42개인데 특성이 55개나 만들어졌으므로 극도의 과대적합

from sklearn.preprocessing import StandardScaler #과대적합을 줄이기위한 규제(가중치(=기울기)작게 만들기) 전 표준화
ss=StandardScaler()
ss.fit(train_poly) #위의 55개 특성의 표준화(훈련세트에 너무 잘 맞는걸 규제해서 표준점수로 만들기)
train_scaled=ss.transform(train_poly)
test_scaled=ss.transform(test_poly)

from sklearn.linear_model import Ridge #규제가 있는 선형 회귀를 릿지 회귀로 실행(릿지 회귀:가중치의 제곱을 벌칙으로 실행)
ridge=Ridge()
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))
alpha_list=[0.001,0.01,0.1,1,10,100] #Ridge의 매개변수 alpha의 기본설정인 1을 바꿔가면서 적절한 규제 강도 찾기
train_score=[]
test_score=[]
for alpha in alpha_list:
       ridge=Ridge(alpha=alpha)
       ridge.fit(train_scaled,train_target)
       train_score.append(ridge.score(train_scaled,train_target))
       test_score.append(ridge.score(test_scaled,test_target))
import matplotlib.pyplot as plt #그래프로 확인
plt.plot(np.log10(alpha_list),train_score) #가시성 좋은 그래프를 위한 넘파이 상용로그 스케일 사용
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show() #alpha=0.1이 가장 적절한 규제강도인듯(훈련세트와 테스트세트의 점수가 가장 가까운 지점이 최적의 alpha)
ridge=Ridge(alpha=0.1) #가장 적절한 규제 강도로 릿지 회귀 재실행
ridge.fit(train_scaled,train_target)
print(ridge.score(train_scaled,train_target))
print(ridge.score(test_scaled,test_target))

from sklearn.linear_model import Lasso #규제가 있는 선형 회귀를 라쏘 회귀로 실행(라쏘 회귀:가중치의 절댓값을 벌칙으로 실행)
#일반적으로 라쏘보다는 릿지가 더 잘맞아서 릿지 선호(라쏘는 가중치를0으로 만들어버려서 특성을 아예 안써버릴수도 있으므로)
lasso=Lasso()
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target))
print(lasso.score(test_scaled,test_target))
alpha_list=[0.001,0.01,0.1,1,10,100] #라쏘의 매개변수 alpha의 기본설정인 1을 바꿔가면서 적절한 규제 강도 찾기
train_score=[]
test_score=[]
for alpha in alpha_list:
       lasso=Lasso(alpha=alpha,max_iter=10000) #라쏘모델이 최적의 계수를 찾는데 필요한 반복횟수가 부족할때 오류 발생(여기서는 괜찮음)
       lasso.fit(train_scaled,train_target)
       train_score.append(lasso.score(train_scaled,train_target))
       test_score.append(lasso.score(test_scaled,test_target))
import matplotlib.pyplot as plt #그래프로 확인
plt.plot(np.log10(alpha_list),train_score) #가시성 좋은 그래프를 위한 넘파이 상용로그 스케일 사용
plt.plot(np.log10(alpha_list),test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show() #alpha=10이 가장 적절한 규제강도인듯
lasso=Lasso(alpha=10) #가장 적절한 규제 강도로 라쏘 회귀 재실행
lasso.fit(train_scaled,train_target)
print(lasso.score(train_scaled,train_target))
print(lasso.score(test_scaled,test_target))
print(np.sum(lasso.coef_==0)) #55개 중 40개를 0으로 만들어서 사용 안했음