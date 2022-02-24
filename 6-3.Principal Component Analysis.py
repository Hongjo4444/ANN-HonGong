#주성분(가장 데이터의 분포를 잘 표현하는 방향) 분석:대표적인 차원축소 알고리즘
#차원(=특성) 축소:주성분을 선택하여 샘플 데이터를 주성분에 투영해서 차원을 줄이는것>>데이터 크기를 줄이고, 다른 지도 학습or비지도 학습 모델의 성능을 향상시킴, 시각화도 쉬워짐

import numpy as np
fruits=np.load('/Users/hongjo/PycharmProjects/pythonProject/Machine Learning and Deep Learning/fruits_300.npy')
fruits_2d=fruits.reshape(-1,100*100)

from sklearn.decomposition import PCA
pca=PCA(n_components=50) #찾을 주성분의 개수
pca.fit(fruits_2d)
print(pca.components_.shape) #PCA 클래스가 찾은 주성분은 'components_'속성에 저장됨(첫번째 차원:찾은 주성분 수 50, 두번째 차원:원본데이터의 특성개수 10000)

import matplotlib.pyplot as plt
def draw_fruits(arr,ratio=1): #클러스터 출력 함수(실제로는 데이터가 너무 많아서 못써먹음)
    n=len(arr) #샘플의 개수
    rows=int(np.ceil(n/10)) #한줄에 10개씩 이미지 그리기(샘플 개수를 10으로 나눠 전체 행 개수 계산)
    cols=n if rows<2 else 10 #행이 1개면 열의 개수는 샘플개수, 그렇지 않으면 열은 10개
    fig,axs=plt.subplots(rows,cols,figsize=(cols*ratio,rows*ratio),squeeze=False) #squeeze=False로 rows,cols에 따라 axs의 차원이 안바뀌게 설정
    for i in range(rows):
        for j in range(cols):
            if i*10+j<n: #n개까지만 그리기
                axs[i,j].imshow(arr[i*10+j],cmap='gray_r')
            axs[i,j].axis('off')
    plt.show()
draw_fruits(pca.components_.reshape(-1,100,100)) #주성분 그려보기(100*100 크기의 이미지처럼 출력)

print(fruits_2d.shape) #원본 데이터의 특성의 개수
fruits_pca=pca.transform(fruits_2d) #transform 메소드로 원본(10000개의 특성(=픽셀)을 가진 300개의 이미지)을 주성분에 투영시킴(특성 50개 됨)
print(fruits_pca.shape) #주성분에 투영시킨 특성의 개수

fruits_inverse=pca.inverse_transform(fruits_pca) #주성분을 다시 원본으로 복원
print(fruits_inverse.shape)
fruits_reconstruct=fruits_inverse.eshape(-1,100,100) #데이터를 100*100 크기로 바꾸어
for start in[0,100,200]: #100개씩 나누어 출력
    draw_fruits(fruits_reconstruct[start:start+100])
    print('\n')

print(np.sum(pca.explained_variance_ratio_)) #설명된 분산:주성분이 원본 데이터의 분산을 얼마나 잘 나타내는지 기록한 값(92%가 넘는 분산을 보존한다)
plt.plot(pca.explained_variance_ratio_) #'설명된 분산'의 비율을 그래프로 그려보기>>적절한 주성분의 개수를 찾는데 도움됨
plt.show() #처음 10개의 주성분이 대부분의 분산을 표현함

from sklearn.linear_model import LinearRegression #PCA를 분류기와 함께 사용하기
lr=LinearRegression()
target=np.array([0]*100+[1]*100+[2]*100) #타겟 만들기(0(사과) 100개,1(파인애플) 100개,2(바나나) 100개)
from sklearn.model_selection import cross_validate
scores=cross_validate(lr,fruits_2d,target) #원본과 타겟으로 교차검증
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time'])) #훈련 시간
scores=cross_validate(lr,fruits_pca,target) #주성분으로 변환한 원본과 타겟으로 교차검증
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

pca=PCA(n_components=0.5) #n_components로 설명된 분산의 비율 설정도 가능>>PCA클래스는 지정된 비율에 도달할때까지 자동으로 주성분 찾음
pca.fit(fruits_2d)
print(pca.n_components_) #주성분 2개만 있으면 50%의 분산 설명 가능
fruits_pca=pca.transform(fruits_2d)
print(fruits_pca.shape)
scores=cross_validate(lr,fruits_pca,target) #2개의 특성만으로 타겟과 교차검증
print(np.mean(scores['test_score']))
print(np.mean(scores['fit_time']))

from sklearn.cluster import KMeans #군집과 함께 사용하기
km=KMeans(n_clusters=3,random_state=42)
km.fit(fruits_pca)
print(np.unique(km.labels_,return_counts=True))
for label in range(0,3):
    draw_fruits(fruits[km.labels_==label])
    print('\n')

for label in range(0,3): #시각화를 위해서 차원축소 알고리즘 사용
    data=fruits_pca[km.labels_==label]
    plt.scatter(data[:,0],data[:,1])
plt.legend(['apple','banana','pineapple'])
plt.show()