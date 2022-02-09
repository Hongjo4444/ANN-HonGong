bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0,
                31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0,
                35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0] #도미 자료 입력
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0,
                500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0,
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0] #빙어 자료 입력
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

import matplotlib.pyplot as plt
plt.scatter(bream_length,bream_weight)
plt.scatter(smelt_length,smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

length=bream_length+smelt_length #도미, 빙어 자료 합치기
weight=bream_weight+smelt_weight

fish_data=[[l,w] for l,w in zip(length,weight)] #생선 데이터를 사이킷런 형태인 2차원 리스트로 만들기
fish_target=[1]*35+[0]*14 #정답(1:도미,0:빙어)자료 입력

from sklearn.neighbors import KNeighborsClassifier #k-최근접 이웃 알고리즘 클래스 임포트
kn=KNeighborsClassifier() #임포트한 k-최근접 이웃 알고리즘 클래스의 객체 만들기
kn.fit(fish_data,fish_target) #훈련(k-최근접 이웃 알고르즘 클래스를 제외한 사이킷런에서 모두 사용)
print(kn.score(fish_data,fish_target)) #정확도 출력
print(kn.predict([[30,600]])) #길이30,무게600인 데이터 입력 후 예측(predict), 리스트의 리스트로 만들어서 2차원 데이터로 넣어줘야함

kn49=KNeighborsClassifier(n_neighbors=49) #전체 비교 샘플을 49개로 설정(원래는 5개만으로 판단함)
kn49.fit(fish_data,fish_target)
print(kn49.score(fish_data,fish_target)) #전체 생선 49, 도미 35 이므로 35/49로 정확도는 0.7142857143

train_input=fish_data[:35]
train_target=fish_target[:35]
test_input=fish_data[35:]
test_target=fish_target[35:]

kn=KNeighborsClassifier()
kn.fit(train_input,train_target)
print(kn.score(test_input,test_target))