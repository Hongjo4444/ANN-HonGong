#군집 알고리즘(군집:비슷한 샘플끼리 그룹으로 모으는 작업, 군집 알고리즘으로 만든 그룹을 '클러스터'라고 부름)

# !wget http://bit.ly/fruits_300_data -0 fruits_300.npy #파이참으로 할때는 http://bit.ly/fruits_300_data 입력해서 다운

import numpy as np
import matplotlib.pyplot as plt
fruits=np.load('/Users/hongjo/PycharmProjects/pythonProject/Machine Learning and Deep Learning/fruits_300.npy')
print(fruits.shape)
print(fruits[0,0,:]) #첫번째 샘플의 첫번째 행의 모든 열을 출력(높은값=하얀색(사과꼭지부분))
plt.imshow(fruits[0],cmap='gray') #넘파이 배열로 저장되어 있는 값을 이미지로 출력하는 imshow(이미지 쇼)
plt.show()
plt.imshow(fruits[0],cmap='gray_r') #gray_r로 반전 >> 색 있는 부분 짙고, 색 없는 부분 하얗게
plt.show()

apple=fruits[0:100].reshape(-1,100*100) #2차원 배열에서 1차원 배열로 샘플 차원 변경
pineapple=fruits[100:200].reshape(-1,100*100)
banana=fruits[200:300].reshape(-1,100*100)
print(apple.shape)

plt.hist(np.mean(apple,axis=1),alpha=0.8) #샘플 평균의 히스토그램(axis=0이면 행끼리(픽셀 평균), axis=1이면 열끼리(샘플 평균) 진행),alpha 1보다 작으면 투명도
plt.hist(np.mean(pineapple,axis=1),alpha=0.8)
plt.hist(np.mean(banana,axis=1),alpha=0.8)
plt.legend(['apple','pineapple','banana'])
plt.show()

fig,axs=plt.subplots(1,3,figsize=(20,5)) #픽셀 평균의 히스토그램, subplot:1,3이면 1개의 행에 3개의 열
axs[0].bar(range(10000),np.mean(apple,axis=0))
axs[1].bar(range(10000),np.mean(pineapple,axis=0))
axs[2].bar(range(10000),np.mean(banana,axis=0))
plt.show()

apple_mean=np.mean(apple,axis=0).reshape(100,100) #평균 이미지 그리기(픽셀평균 낸것을 다시 100*100크기로(1차원 배열>>2차원 배열)
pineapple_mean=np.mean(pineapple,axis=0).reshape(100,100)
banana_mean=np.mean(banana,axis=0).reshape(100,100)
fig,axs=plt.subplots(1,3,figsize=(20,5))
axs[0].imshow(apple_mean,cmap='gray_r') #사과 사진들의 평균 모습
axs[1].imshow(pineapple_mean,cmap='gray_r') #파인애플 사진들의 평균 모습
axs[2].imshow(banana_mean,cmap='gray_r') #바나나 사진들의 평균 모습
plt.show()

abs_diff=np.abs(fruits-apple_mean) #평균과 가까운 사진 고르기
abs_mean=np.mean(abs_diff,axis=(1,2)) #300개의 1차원 배열 생성
print(abs_mean.shape)
apple_index=np.argsort(abs_mean)[:100] #np.argsort()는 작은 것에서 큰 순서대로 나열한 abs_mean 배열의 인덱스 반환
fig,axs=plt.subplots(10,10,figsize=(10,10))
for i in range(10):
    for j in range(10):
        axs[i,j].imshow(fruits[apple_index[i*10+j]],cmap='gray_r')
        axs[i,j].axis('off') #깔끔하게 이미지만 그리기위해 좌표축 안그림
plt.show()