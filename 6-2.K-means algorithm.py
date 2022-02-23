#k평균 알고리즘:어떤 사진이 들어있는지 모를때 사진의 평균 구하는 법
#k평균 알고리즘 작동방법:1.무작위로 k개의 클러스터 중심 정하기>>2.각 샘플에서 가장 가까운 클러스터 중심을 찾아 해당 클러스터의 샘플로 지정>>3.클러스터에 속한 샘플의 평균값으로 클러스터 중심 변경>>4.클러스터 중심에 변화가 없을때까지 2번으로 돌아가 반복

# !wget http://bit.ly/fruits_300_data -0 fruits_300.npy #파이참으로 할때는 http://bit.ly/fruits_300_data 입력해서 다운
import numpy as np
fruits=np.load('/Users/hongjo/PycharmProjects/pythonProject/Machine Learning and Deep Learning/fruits_300.npy')
fruits_2d=fruits.reshape(-1,100*100)

from sklearn.cluster import KMeans
km=KMeans(n_clusters=3,random_state=42) #n_clusters로 클러스터의 개수 지정
km.fit(fruits_2d)
print(km.labels_) #훈련 결과 출력(클러스터가 3개니까 인덱스는 0,1,2)
print(np.unique(km.labels_,return_counts=True)) #클러스터 0,1,2가 몇번씩 등장했는지 출력

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

draw_fruits(fruits[km.labels_==0]) #첫번째 클러스터 출력
draw_fruits(fruits[km.labels_==1]) #첫번째 클러스터 출력
draw_fruits(fruits[km.labels_==2]) #첫번째 클러스터 출력

draw_fruits(km.cluster_centers_.reshape(-1,100,100),ratio=3) #클러스터의 중심(평균값) 그리기(km.cluster_centers_에 클러스터의 중심 위치 저장되어있음)
print(km.transform(fruits_2d[100:101])) #101번째 샘플~클러스터 3개의 중심까지의 거리, 3번째 클러스터 중심까지 거리가 가장 짧으므로 3번째 클러스터에 속할 확률 높음
print(km.predict(fruits_2d[100:101])) #위치를 찾아보면 인덱스가 2인 클러스터(3번째 클러스터)에 속한것 확인 가능
draw_fruits(fruits[100:101]) #101번째 샘플을 그려보면 파인애플(3번째 클러스터)임(클러스터 중심 그렸을때 2는 파인애플이였음)
print(km.n_iter_) #k평균을 위해 몇번 반복했는지

inertia=[] #최적의 k(클러스터) 찾기
for k in range(2,7): #클러스터의 개수를 바꿔가며 테스트
    km=KMeans(n_clusters=k,random_state=42)
    km.fit(fruits_2d)
    inertia.append(km.inertia_) #클러스터의 줌심~클러스터에 속한 샘플들의 거리들을 제곱하고 합한것
plt.plot(range(2,7),inertia)
plt.xlabel('k')
plt.ylabel('inertia')
plt.show() #k=3일때가 최적(급격하다가 완만해지는 지점이 최적의 지점,k를 늘려도 잘 밀집된 정도가 크게 바뀌지 않으므로)