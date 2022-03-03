from tensorflow import keras #MNIST 데이터셋 적재
(train_input,train_target),(test_input,test_target)=keras.datasets.fashion_mnist.load_data()
from sklearn.model_selection import train_test_split #훈련세트,검증세트로 나누기
train_scaled=train_input.reshape(-1,28,28,1)/255.0 #Cpnv2D 층 사용을 위해 마지막에 채널 차원 추가(크기가 1인 차원 추가)
train_scaled,val_scaled,train_target,val_target=train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)

model=keras.Sequential() #첫 번째 합성곱 층 만들기
model.add(keras.layers.Conv2D(32,kernel_size=3,activation='relu',padding='same',input_shape=(28,281))) #필터 32개, 필터크기 3, 케라스 첫번째 층에는 무조건 입력 배열 넣어줘야함
model.add(keras.layers.MaxPooling2D(2)) #너비,높이 2인 최대풀링 적용

model.add(keras.layers.Conv2D(64,kernel_size=3,activation='relu',padding='same')) #두 번째 합성곱 층 만들기
model.add(keras.layers.MaxPooling2D(2))

model.add(keras.layers.Flatten()) #완전 연결 층 만들기
model.add(keras.layers.Dense(100,activation='relu')) #은닉층
model.add(keras.layers.Dropout(0.4)) #과대적합을 막기위한 Dropout
model.add(keras.layers.Dense(10,activation='softmax')) #출력층(10개를 분류하는 문제니까 10)

model.summary() #신경망의 모델구조 출력

keras.utils.plot_model(model) #층의 구성을 그림으로 보여줌
keras.utils.plot_model(model,show_shapes=True) #show_shapes=True로 입출력까지 보여줌

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy') #컴파일
checkpoint_cb=keras.callbacks.ModelCheckpoint('best-cnn-model.h5') #체크포인트 콜백 저장
early_stopping_cb=keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True) #조기종료 콜백 설정,restore했으므로 최적파라미터 자동 저장>>ModelCheckpoint콜백이 저장한 파일 다시 불러올 필요 없음
history=model.fit(train_scaled,train_target,epochs=20,validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb,early_stopping_cb]) #훈련,val:검증세트

import matplotlib.pyplot as plt #그래프 출력
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

model.evaluate(val_scaled,val_target) #평가(손실/정확도 출력)
plt.imshow(val_scaled[0].reshape(28,28),cmap='gray_r') #첫 번째 샘플 이미지
plt.show()
preds=model.predict(val_scaled[0:1]) #예측
print(preds) #9번(가방)의 확률이 절대적임
plt.bar(range(1,11),preds[0]) #막대그래프로 예측한것 그리기
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

classes=['티셔츠','바지','스웨터','드레스','코트','샌달','셔츠','스니커즈','가방','앵클 부츠'] #클래스 리스트 만들기
import numpy as np #레이블 출력(preds 배열에서 가장 큰 인덱스를 찾아서 classes 리스트의 인덱스로 사용)
print(classes[np.argmax(preds)])

test_scaled=test_input.reshape(-1,28,28,1)/255.0 #테스트 세트 만들기
model.evaluate(test_scaled,test_target) #테스트 세트 성능 측정