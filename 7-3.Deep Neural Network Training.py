from tensorflow import keras #MNIST 데이터셋 적재
(train_input,train_target),(test_input,test_target)=keras.datasets.fashion_mnist.load_data()
from sklearn.model_selection import train_test_split #훈련세트,검증세트로 나누기
train_scaled=train_input/255.0
train_scaled=train_scaled.reshape(-1,28,28)
train_scaled,val_scaled,train_target,val_target=train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)

def model_fn(a_layer=None): #모델을 만드는 함수 정의
    model=keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28,28)))
    model.add(keras.layers.Dense(100,activation='relu'))
    if a_layer:
        model.add(a_layer)
    model.add(keras.layers.Dense(10,activation='softmax'))
    return model

model=model_fn() #모델 만들기
model.summary() #모델구조 출력
model.compile(loss='sparse_categorical_crossentropy',metrics='accuracy') #모델 설정(손실함수 설정, 훈련과정에서 계산하고 싶은 측정값 지정)

history=model.fit(train_scaled,train_target,epochs=20,verbose=0) #에포크 20으로 모델 훈련,verbose:훈련과정의 출력 조절(0이면 훈련 과정 나타내지 않음)
print(history.history.keys()) #훈련 측정값이 담겨있는 history 딕셔더니 출력
import matplotlib.pyplot as plt #손실그래프 그리기
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

history=model.fit(train_scaled,train_target,epochs=20,verbose=0,validation_data=(val_scaled,val_target)) #검증손실 계산(validation_data:검증손실 자동으로 계산)
#에포크를 늘리면 무조건 손실감소>>과대적합>>검증손실로 검증해야함
print(history.history.keys())
plt.plot(history.history['loss']) #손실/검증손실 그래프 그리기
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

model=model_fn(keras.layers.Dropout(0.3)) #드롭아웃:신경망에서 사용하는 대표적인 규제 방법(0.3:30% 드롭아웃)
#은닉층의 뉴런 일부러 빼고 훈련>>모델이 훈련세트에 과대적합or어느 한 뉴런에 과도하게 의존하는것 방지, 훈련 후 평가나 예측에는 드롭아웃 적용하면 안됨(텐서플로,케라스에서 알아서 빼고 계산함)
model.summary()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')

history=model.fit(train_scaled,train_target,epochs=20,verbose=0,validation_data=(val_scaled,val_target)) #검증손실 계산
print(history.history.keys())
plt.plot(history.history['loss']) #손실/검증손실 그래프 그리기
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show() #과대적합 방지됨

model.save_weights('model-weights.h5') #모델 저장과 복원(save_weights:가중치(파라미터)만 저장(모델 구조 저장x),save:모델 구조 모두 저장)
model.load_weights('model-weights.h5')
model.save('model-whole.h5')
model.keras.models.load_model(('model-whole.h5'))

import numpy as np
val_labels=np.argmax(model.predict(val_scaled),axis=-1) #10개의 확률 중 가장 큰 값의 인덱스를 고르기
print(np.mean(val_labels==val_target)) #타겟 레이블과 비교>>정확도 계산

model=model_fn(keras.layers.Dropout(0.3)) #콜백:충분히 큰 에포크로 모델을 한번 훈련하고, 검증세트의 손실과 훈련세트의 손실을 비교해서 과대적합되는 지점까지 다시 한번 모델 훈련
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
checkpoint_cb=keras.callbacks.Modelcheckpoint('best=model.h5') #Modelcheckpoint 콜백:콜백의 종류, 지정된 파일에 훈련하는 도중에 가장 낮은 손실값을 가지는 모델 가중치 저장
model.fit(train_scaled,train_target,epoches=20,verbose=0,validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb])
model=keras.models.load_model('best-model.h5')
model.evaluate(val_scaled,val_target) #모델 예측

model=model_fn(keras.layers.Dropout(0.3)) #조기종료:조기종료 쓰면 에포크 수를 크게 지정해도 알아서 종료됨
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy')
checkpoint_cb=keras.callbacks.Modelcheckpoint('best=model.h5',save_best_only=True)
early_stiopping_cb=keras.callbacks.EarlyStopping(patience=2,restore_best_weights=True) #가장 손실이 낮은 epoch로 되돌려라,patience=2:2번 연속 검증점수가 향상되지 않으면 훈련 중지
history=model.fit(train_scaled,train_target,epoches=20,verbose=0,validation_data=(val_scaled,val_target),callbacks=[checkpoint_cb,early_stiopping_cb])
#EarlyStopping 콜백+ModelCheckpoint 콜백>>가장 낮은 검증 손실의 모델을 파일에 저장하고 검증 손실이 다시 상승할때 훈련 중지+훈련 중지 후 현재 파라미터를 최상의 파라미터로 되돌림
print(early_stiopping_cb.stopped_epoch) #12>>에포크 횟수가 0부터 시작하니까 13번째 에포크에서 훈련 중지됐다는 말+patience가 2니까 11번째에서 훈련을 멈췄다는 말
plt.plot(history.history['loss']) #손실/검증손실 그래프 그리기
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()
model.evaluate(val_scaled,val_target) #모델 예측