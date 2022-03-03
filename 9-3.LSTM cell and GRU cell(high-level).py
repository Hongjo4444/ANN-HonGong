from tensorflow.keras.datasets import imdb
from sklearn.model_selection import train_test_split
(train_input,train_target),(test_input,test_target)=imdb.load_data(num_words=500)
train_input,val_input,train_target,val_target=train_test_split(train_input,train_target,test_size=0.2,random_state=42)

from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq=pad_sequences(train_input,maxlen=100)
val_seq=pad_sequences(val_input,maxlen=100)

from tensorflow import keras #LSTM셀 모델 만들기
model=keras.Sequential()
model.add(keras.layers.Embedding(500,16,input_length=100))
model.add(keras.layers.LSTM(8)) #LSTM셀:다음 층으로 전달되지 않고, LSTM셀에서만 순환되는 값(=셀상태)을 가지는 순환신경망(작은 순환신경망을 여러개 포함하고 있는 큰 셀이라고 생각)
#삭제 게이트:셀 상태에 있는 정보를 제거하는 역할,입력 게이트:새로운 정보를 셀 상태에 추가,출력 게이트:셀 상태가 다음 은닉 상태로 출력
model.add(keras.layers.Dense(1,activation='sigmoid'))
model.summary()

rmsprop=keras.optimizers.RMSprop(learning_rate=1e-4) #모델 컴파일,훈련
model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
checkpoint_cb=keras.callbacks.ModelCheckpoint('best-lstm-model.h5',save_best_only=True)
early_stopping_cb=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
history=model.fit(train_seq,train_target,epochs=100,batch_size=64,validation_data=(val_seq,val_target),callbacks=[checkpoint_cb,early_stopping_cb])

import matplotlib.pyplot as plt #훈련손실,검증손실 그래프 그리기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

model2=keras.Sequential() #드롭아웃 적용한 모델 만들기(과대적합 억제 가능)
model2.add(keras.layers.Embedding(500,16,input_length=100))
model2.add(keras.layers.LSTM(8,dropout=0.3))
model2.add(keras.layers.Dense(1,activation='sigmoid'))
model2.summary()

model3=keras.Sequential() #드롭아웃+2개의 층 연결한 모델 만들기(오직 마지막 순환층만 마지막 타임스텝의 은닉 상태를 출력해야함)
model3.add(keras.layers.Embedding(500,16,input_length=100))
model3.add(keras.layers.LSTM(8,dropout=0.3,return_sequences=100)) #모든 타임스텝의 은닉 상태를 출력하려면 마지막을 제외한 다른 모든 순환층에서 return_sequences=True
model3.add(keras.layers.LSTM(8,dropout=0.3))
model3.add(keras.layers.Dense(1,activation='sigmoid'))
model3.summary()

#GRU셀 모델 만들기(LSTM셀의 간소화 버전,셀 상태 없이 은닉상태와 입력값의 조합만 있는 모델)
model4=keras.Sequential()
model4.add(keras.layers.Embedding(500,16,input_length=100))
model4.add(keras.layers.GRU(8))
model4.add(keras.layers.Dense(1,activation='sigmoid'))
model4.summary()

rmsprop=keras.optimizers.RMSprop(learning_rate=1e-4) #모델 컴파일,훈련
model4.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
checkpoint_cb=keras.callbacks.ModelCheckpoint('best-gru-model.h5',save_best_only=True)
early_stopping_cb=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
history=model4.fit(train_seq,train_target,epochs=100,batch_size=64,validation_data=(val_seq,val_target),callbacks=[checkpoint_cb,early_stopping_cb])

import matplotlib.pyplot as plt #훈련손실,검증손실 그래프 그리기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

test_seq=pad_sequences(test_input,maxlen=100) #테스트세트를 훈련세트와 동일한 방식으로 변환
rnn_model=keras.models.load_model('best-lstm-model.h5') #결과가 가장 좋았던 2개의 순환층을 쌓은 모델 로드
rnn_model.evaluate(test_seq,test_target) #테스트세트에서 성능 계산