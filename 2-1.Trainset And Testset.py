bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0] #도미 자료 입력
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0] #빙어 자료 입력
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

length=bream_length+smelt_length #도미, 빙어 자료 합치기
weight=bream_weight+smelt_weight

fish_data=[[l,w] for l,w in zip(length,weight)] #생선 데이터를 사이킷런 형태인 2차원 리스트로 만들기
fish_target=[1]*35+[0]*14 #정답(1:도미,0:빙어)자료 입력

train_input=fish_data[:35] #훈련 세트 자료
train_target=fish_target[:35]
test_input=fish_data[35:] #테스트 세트 자료
test_target=fish_target[35:]

from sklearn.neighbors import KNeighborsClassifier
kn=KNeighborsClassifier()
kn.fit(train_input,train_target) #훈련 세트
print(kn.score(test_input,test_target)) #테스트 세트(훈련은 도미로하고, 테스트는 빙어로 하니 당연히 정확도는 0)

import numpy as np
input_arr=np.array(fish_data) #넘파이로 배열 만들기
target_arr=np.array(fish_target)
print(input_arr)
print(input_arr.shape)

index=np.arange(49) #arange함수로 인덱스 배열로 만들기(=배열 인덱싱)
np.random.shuffle(index) #shuffle함수로 무작위로 섞기
train_input=input_arr[index[:35]] #훈련 세트 자료
train_target=target_arr[index[:35]]
test_input=input_arr[index[35:]] #테스트 세트 자료
test_target=target_arr[index[35:]]

import matplotlib.pyplot as plt #선점도로 섞였는지 확인
plt.scatter(train_input[:,0],train_input[:,1]) #2차원 배열의 원소 선택시 ',' 사용([:,0]:전체 데이터의 0번째 위치의 원소)
plt.scatter(test_input[:,0],test_input[:,1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn=kn.fit(train_input,train_target)
print(kn.score(test_input,test_target))