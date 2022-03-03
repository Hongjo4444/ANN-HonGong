#IMDB 리뷰 데이터셋(영화 리뷰를 감상평에 따라 긍정,부정으로 분류해놓은 데이터셋)
#자연어 처리(NLP,natural language program):컴퓨터를 상ㅇ해 인간의 언어를 처리하는 분야, 말뭉치:자연어 처리 분야에서 훈련데이터
#토큰:분리해서 사용하는 단어(하나의 타임스텝에 해당함,0:패딩,1:문장의 시작,2:어휘 사전에 없는 토큰), 어휘사전:고유한 단어(=토큰) 목록 집합(ex)He)
#텍스트 데이터를 숫자로 만드는법:특정 단어를 숫자로 바꾼다
import numpy as np
from tensorflow.keras.datasets import imdb #케라스로 IMDB 데이터 불러오기
(train_input,train_target),(test_input,test_target)=imdb.load_data(num_words=500) #500개의 단어만 불러옴
print(train_input[0]) #1:샘플의 시작,2:어휘사전에 없는 단어
print(train_target[:20]) #1:긍정,2:부정

from sklearn.model_selection import train_test_split #훈련세트 준비
train_input,val_input,train_target,val_target=train_test_split(train_input,train_target,test_size=0.2,random_state=42) #20% 검증세트로 설정
lengths=np.array([len(x) for x in train_input]) #적절한 길이의 토큰 길이 설정을 위한 리뷰 문장들의 평균값,중간값 알아보기
print(np.mean(lengths),np.median(lengths))
import matplotlib.pyplot as plt #리뷰 문장들의 길이 분포 히스토그램으로 출력
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

from tensorflow.keras.preprocessing.sequence import pad_sequences #시퀀스 패딩(train input의 길이를 정하고, 리뷰가 train input보다 길면 자르고, 짧으면 0으로 채워줌)
train_seq=pad_sequences(train_input,maxlen=100) #패딩 사용해서 100으로 맞추기(리뷰대부분이 짧으므로)
print(train_seq.shape) #5000개로 검증세트로 쓰고, 나머지는 20000개는 토큰 100개만 훈련세트로 씀
print(train_seq[0])
print(train_input[0][-10:]) #뒷부분 확인>>긴 경우 기본적으로 pad_sequence로 앞부분 자른것 알 수 있음(뒷부분이 더 의미있다고 생각해서,뒤 자르고 싶으면 pre>>post)
print(train_seq[5]) #짧은 경우 앞부분에 0이 패딩된것 알 수 있음
val_seq=pad_sequences(val_input,maxlen=100) #검증세트의 길이도 100으로 맞추기

from tensorflow import keras #순환 신경망 모델 만들기
model=keras.Sequential()
model.add(keras.layers.SimpleRNN(8,input_shape=(100,500))) #뉴런 숫자 8개, input 사이즈 지정(100개 토큰(샘플 길이),원-핫 인코딩으로 표현하기 위한 배열의 길이 500)
model.add(keras.layers.Dense(1,activation='sigmoid'))
train_oh=keras.utils.to_categorical(train_seq) #훈련데이터 만들기(원-핫 인코딩 만들기)
print(train_oh.shape) #정수 하나마다 모두 500차원의 배열로 변경되어서 (20000,100)>>(20000,100,500)됨
print(train_oh[0][0][:12]) #첫번째 샘플의 첫번째 토큰 10이 잘 인코딩 되었는지 확인
print(np.sum(train_oh[0][0])) #나머지 원소가 모두 0인지 확인>>합이 1이므로 나머지는 다0임
val_oh=keras.utils.to_categorical(val_seq) #검증세트 원-핫 인코딩 하기
model.summary #모델 구조 확인

rmsprop=keras.optimizers.RMSprop(learning_rate=1e-4) #순환신경망 모델 훈련
model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
checkpoint_cb=keras.callbacks.ModelCheckpoint('best_simplernn-model.h5')
early_stopping_cb=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
history=model.fit(train_oh,train_target,epochs=100,batch_size=64,validation_data=(val_oh,val_target),callbacks=[checkpoint_cb,early_stopping_cb])
plt.plot(history.histiory['loss']) #훈련손실,검증손실 그래프로 그리기
plt.plot(history.histiory['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()

model12=keras.Sequential() #⭐️원-핫 인코딩을 단어 임베딩으로(원-핫 인코딩은 훈련 데이터가 엄청 커짐)>>토큰을 각 토큰간 연결도에 따라 고정된 크기의 실수 벡터로 바꾸어줌(2차원 배열 되긴 하지만 원-핫 인코딩보다 훨씬 작은 크기)
model12.add(keras.layers.Embedding(500,16,input_length=100)) #500:어휘사전의 크기, 16:임베딩 벡터 크기>>16개의 벡터로 토큰 하나 출력, input_length:입력 시퀀스의 길이(샘플길이름 100워로 맞췄으므로 100)
model12.add(keras.layers.SimpleRNN(8))
model12.add(keras.layers.Dense(1,activation='sigmoid'))
model.summary() #모델 구조 출력

rmsprop=keras.optimizers.RMSprop(learning_rate=1e-4) #단어 임베딩한 순환신경망 모델 훈련
model.compile(optimizer=rmsprop,loss='binary_crossentropy',metrics=['accuracy'])
checkpoint_cb=keras.callbacks.ModelCheckpoint('best_simplernn-model.h5',save_best_only=True)
early_stopping_cb=keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)
history=model12.fit(train_seq,train_target,epochs=100,batch_size=64,validation_data=(val_seq,val_target),callbacks=[checkpoint_cb,early_stopping_cb])
#원-핫 인코딩과 비슷한 성능을 내지만, 순환층의 가중치 수, 훈련세트 크기가 훨씬 줄어듬
plt.plot(history.histiory['loss']) #단어 임베딩한 훈련손실,검증손실 그래프로 그리기
plt.plot(history.histiory['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train','val'])
plt.show()