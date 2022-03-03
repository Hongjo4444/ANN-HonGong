from tensorflow import keras #패션 MNIST 데이터셋

(train_input,train_target),(test_input,test_target)=keras.datasets.fashion_mnist.load_data()
print(train_input.shape,train_target.shape)
print(test_input.shape,test_target.shape)

import matplotlib.pyplot as plt #이미지 데이터 10개만 출력
fig,axs=plt.subplots(1,10,figsize=(10,10))
for i in range(10):
    axs[i].imshow(train_input[i],cmap='gray_r') #색상 반전
    axs[i].axis('off')
plt.show()
print(train_target[i] for i in range(10)) #처음 10개샘플 타깃값을 리스트로 만든 후 출력

import numpy as np
print(np.unique(train_target,return_counts=True)) #레이블 당 샘플 개수 확인>>레이블 당 6000개의 샘플 있음

train_scaled=train_input/255.0 #로지스틱 회귀로 패션 아이템 분류하기 위한 스케일 전처리
train_scaled=train_scaled.reshape(-1,28*28) #2차원 배열인 각 샘플을 1차원으로 변경(SGDClassfier는 2차원 입력 못다룸)
print(train_scaled.shape)

from sklearn.linear_model import SGDClassifier #로지스틱 회귀(인공신경망(=딥러닝) 방식이랑 비슷)로 패션 아이템 분류
sc=SGDClassifier(loss='log',max_iter=5,random_state=42) #손실함수의 매개함수 log>>경사하강법(이진분류,다중분류 모두 가능), 에포크 5번
from sklearn.model_selection import cross_validate #SGDClassfier의 교차검증 점수 확인
scores=cross_validate(sc,train_scaled,train_target,n_jobs=-1)
print(np.mean(scores['test_score'])) #정확도 81.9%정도

from sklearn.model_selection import train_test_split #케라스 인공신경망 모델 만들기
train_scaled,val_scaled,train_target,val_target=train_test_split(train_scaled,train_target,test_size=0.2,random_state=42) #훈련세트, 검증세트 나누기
print(train_scaled.shape,train_target.shape) #훈련세트의 크기
print(val_scaled.shape,val_target.shape) #검증세트의 크기
dense=keras.layers.Dense(10,activation='softmax',input_shape=(784,)) #케라스의 밀집층 만들기
#dense:출력층 10개 뉴런(=유닛)(항상 클래스 개수랑 같아야함)
#activation(활성함수):이진분류>>sigmoid 함수, 다중분류>>softamx 함수(10개의 뉴런에서 출력되는 값을 확률로 바꾸기위해서)
#input_shape:입력값의 크기 입력(나중에 편리함,input_shape 크기=샘플의 크기)
model=keras.Sequential(dense) #신경망모델 만들기
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy') #모델 설정(손실함수 설정, 훈련과정에서 계산하고 싶은 측정값 지정)
print(train_target[:10]) #sparse 써서 원-핫인코딩이 아닌 그냥 정수 타겟값 출력됨
model.fit(train_scaled,train_target,epochs=5) #모델 훈련
model.evaluate(val_scaled,val_target) #케라스 모델의 성능 평가
