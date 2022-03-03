from tensorflow import keras #MNIST 데이터셋 적재
(train_input,train_target),(test_input,test_target)=keras.datasets.fashion_mnist.load_data()

from sklearn.model_selection import train_test_split #훈련세트,검증세트로 나누기
train_scaled=train_input/255.0
train_scaled=train_scaled.reshape(-1,28,28)
train_scaled,val_scaled,train_target,val_target=train_test_split(train_scaled,train_target,test_size=0.2,random_state=42)

dense1=keras.layers.Dense(100,activation='sigmoid',input_shape=(784,)) #은닉층(입력층과 출력층 사이에 있는 모든 층) 만들기(출력층보다는 많게+갯수는 특별한 기준없음,경험)
dense2=keras.layers.Dense(10,activation='softmax') #출력층
model=keras.Sequential([dense1,dense2]) #심층 신경망 만들기(은닉층~출력층 순서대로 리스트에 넣기)
model.summary() #summary 메서드로 층에 대한 정보 얻기
#은닉층 차원의 샘플수 미지정(어떤 배치에도 대응할 수 있게),100개의 뉴런/파라미터(각층의 가중치,절편) 78500(가중치:입력층 784개 뉴런x은닉층 100개 뉴런+100개 절편)
#출력층 차원의 샘플수 미지정(어떤 배치에도 대응할 수 있게),10개의 뉴런/파라미터 1010(가중치:은닉층 100개 뉴런x출력층 10개 뉴런+10개 절편)

model=keras.Sequential([keras.layers.Dense(100,activation='sigmoid',input_shape=(784,),name='hidden'), #name 매개변수로 층 이름 지정(안하면 dense로 자동 지정)
                        keras.layers.Dense(10,activation='softmax',name='output')],
                       name='패션 MNIST 모델') #keras에서 Sequential 클래스를 사용하는 다른 방법층을 추가하는 다른 방법)1:Dense를 만들면서 바로 넣는것
model=keras.Sequential() #⭐️keras에서 Sequential 클래스를 사용하는 다른 방법층을 추가하는 다른 방법)2:Sequential클래스를 하나 만들고 add로 하나씩 더하는 것
model.add(keras.layers.Dense(100,activation='sigmoid',input_shape=(784,))) #케라스에서 신경망의 첫번째 층은 입력의 크기를 꼭 지정해줘야함
model.add(keras.layers.Dense(10,activation='softmax'))

model.compile(loss='sparse_categoricla_crossentropy',metrics='accuracy') #모델 설정(손실함수 설정, 훈련과정에서 계산하고 싶은 측정값 지정)
model.fit(train_scaled,train_target,epochs=5) #모델 훈련
model.evaluate(val_scaled,val_target) #검증세트에서 성능 확인

model=keras.Sequential() #렐루함수(이미지 분류 문제에서 높은 성능을 내는 활성함수)와 Flatten층>>시그모이드 함수보다 정확도 조금 높아짐
model.add(keras.layers.Flatten(input_shape=(28,28))) #학습하는 층이 아니여서 층 셀때 카운트 안함
model.add(keras.layers.Dense(100,activation='relu'))
#활성함수로 시그모이드 함수 대신 렐루함수 사용(시그모이드 함수는 출력값이 너무 크거나 너무 작아질 경우 시그모이드 값의 차이가 너무 작아지는 단점이 있음>>층을 깊게 쌓기가 힘듬)
#활성함수:신경망 층의 선형방정식 계산 값에 적용하는 함수(출력층에는 이진분류>>시그모이드 함수,다중분류>>소프트맥스 함수로 고정, 회귀 신경망에는 활성함수 미사용)
model.add(keras.layers.Dense(10,activation='softmax'))
#Flatten층:1차원 배열을 펼치는 작업을 해줌(28x28>>784),파라미터는 0(계산에 관여 안하는 유틸리티 층)
model.summary()

model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics='accuracy') #옵티마이저:케라스의 다양한 종류의 확률적 경사 하강법 알고리즘
#sgd=keras.optimizer.SGD() + model.compile(optimizer=sgd,loss='sparse_categorical_crossentropy',metrics='accuracy')의 요약 버전
#model.compile(optimizer='adagrad',loss='sparse_categorical_crossentropy',metrics='accuracy') #'Adagrad' 사용시(적응적 학습률 사용)
#model.compile(optimizer='rmsprop',loss='sparse_categorical_crossentropy',metrics='accuracy') #'RMSprop' 사용시(적응적 학습률 사용)
#model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics='accuracy') #'Adam(모멘텀 최적화+RMSprop)' 사용시(적응적 학습률 사용)
sgd=keras.optimizers.SGD(learning_rate=0.1) #학습률 변경시
sgd=keras.optimizers.SGD(momentum=0.9,nesterov=True) #'모멘텀 최적화(보통 0.9이상 사용)' 중 '네스테로프 모멘텀 최적화(모멘텀 최적화 2번 진행)' 사용시
model.fit(train_scaled,train_target,epochs=5)
model.evaluate(val_scaled,val_target) #검증세트에서 성능 확인
