fish_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0, 9.8,
                10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
fish_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0, 6.7,
                7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import numpy as np
fish_data=np.column_stack((fish_length,fish_weight)) #넘파이로 데이터 준비
fish_target=np.concatenate((np.ones(35),np.zeros(14)))

from sklearn.model_selection import train_test_split
train_input,test_input,train_target,test_target=train_test_split(fish_data,fish_target,stratify=fish_target,random_state=42)
#train_test_split 함수로 fish_data,fish_target을 train_input,test_input,train_target,test_target으로 나눠준것(2개를 4개로, stratify:비율 맞춰서 섞는기능)

from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(train_input,train_target)
print(kn.score(test_input,test_target))
print(kn.predict([[25,150]])) #25cm, 150g의 도미 입력

distances,indexes=kn.kneighbors([[25,150]]) #최근접 5개의 인덱스 알려줌
import matplotlib.pyplot as plt
plt.scatter(train_input[:,0],train_input[:,1]) #전체의 0열(길이), 전체의 1열(무게)
plt.scatter(25,150,marker='^')
plt.scatter(train_input[indexes,0],train_input[indexes,1],marker='D')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

plt.scatter(train_input[:,0],train_input[:,1]) #전체의 0열(길이), 전체의 1열(무게)
plt.scatter(25,150,marker='^')
plt.scatter(train_input[indexes,0],train_input[indexes,1],marker='D')
plt.xlim((0,1000)) #스케일 같게 조정한것
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

mean=np.mean(train_input,axis=0) #표준 점수로 바꾸기(axis를 0으로 하면 열방향의 평균을 구해줌, 1로하면 행방향의 평균을 구해줌)
std=np.std(train_input,axis=0)
train_scaled=(train_input-mean)/std #표준 점수=(특성-평균)/표준 편차
new=([25,150]-mean)/std
plt.scatter(train_scaled[:,0],train_scaled[:,1]) #수상한 도미를 표준 점수로 바꿔서 다시 표시
plt.scatter(new[0],new[1],marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.fit(train_scaled,train_target) #전처리 데이터에서 모델 훈련
test_scaled=(test_input-mean)/std
print(kn.score(test_scaled,test_target)) #타겟은 전처리 하지 않아도 됨
print(kn.predict([new])) #수상한 도미 다시 예측

distances,indexes=kn.kneighbors([new])
plt.scatter(train_scaled[:,0],train_scaled[:,1])
plt.scatter(new[0],new[1],marker='^')
plt.scatter(train_scaled[indexes,0],train_scaled[indexes,1],marker='D') #최근접이웃 5개 다시 보기
plt.xlabel('length')
plt.ylabel('weight')
plt.show()