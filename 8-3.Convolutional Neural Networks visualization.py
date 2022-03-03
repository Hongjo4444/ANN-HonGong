from tensorflow import keras #이전 파일의 체크포인트 파일 읽기
model=keras.models.load_model('best-cnn-model.h5')

model.layers #추가한 층 확인

conv=model.layers[0] #첫번째(인덱스 0) 층의 정보 확인
print(conv.weights[0].shape,conv.weights[1].shape) #합성곱 층의 가중치(weights[0],커널 크기(3,3,1)와 필터 수(32),절편(weights[1]) 출력

conv_weights=conv.weights[0].numpy() #합성곱 층의 가중치 시각화(합성곱 층이 이미지에서 어떤것을 학습했는지 알아보기위한 방법1)
print(conv_weights.mean(),conv_weights.std()) #가중치 배열의 평균,표준편차 출력
import matplotlib.pyplot as plt #가중치 분포 히스토그램으로 그리기
plt.hist(conv_weights.reshape(-1,1)) #히스토그램 그리려면 1차원 배열로 전달해야하므로 배열 변환함
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

fig,axs=plt.subplots(2,16,figsize=(15,2)) #커널 출력
for i in range(2):
    for j in range(16):
        axs[i,j].imshow(conv_weights[:,:,0,i*16+j],vmin=-0.5,vmax=0.5) #출력되는 32개의 밝기가 기준을 가지도록 vmin,vmax 설정
        axs[i,j].axis('off')
plt.show()

#함수형 API:복잡한 신경망 모델(ex)입력2개,출력2개)을 만들때 Sequential클래스 대신 Model클래스를 쓰는것
#케라스에서 input layer class를 지정해줘야하지만 sequential class에서는 편의상 자동으로 만들어주고,함수형 API에서는 keras.Input(shape=)함수로 명시적으로 지정해줘야함

(train_input,train_target),(test_input,test_target)=keras.datasets.fashion_mnist.load_data()

conv_acti=keras.Model(model.input,model.layers[0].output) #첫 번째 특성맵 시각화(합성곱 층이 이미지에서 어떤것을 학습했는지 알아보기위한 방법2)
inputs=train_input[0:1].reshape(-1,28,28,1)/255.0
feature_maps=conv_acti.predict(inputs)
print(feature_maps.shape)
fig,axs=plt.subplots(4,8,figsize=(15,8)) #특성맵 그리기
for i in range(4):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0,:,:,i*8+j])
        axs[i,j].axis('off')
plt.show()

conv2_acti=keras.Model(model.input,model.layers[2].output) #두 번째 특성맵 시각화(첫번째 특성맵을 기반으로)
inputs=train_input[0:1].reshape(-1,28,28,1)/255.0
feature_maps=conv2_acti.predict(inputs)
print(feature_maps.shape)
fig,axs=plt.subplots(8,8,figsize=(12,12)) #특성맵 그리기
for i in range(8):
    for j in range(8):
        axs[i,j].imshow(feature_maps[0,:,:,i*8+j])
        axs[i,j].axis('off')
plt.show() #특성맵 시각화는 층이 깊어질수록 이해하기 난해할 수 있다