import pandas as pd
wine=pd.read_csv('https://bit.ly/wine_csv_data')

wine.head() #처음 5개 샘플 확인
wine.info() #info 메소드 사용 후 'Non-Null Count'로 누락된 값이 있는지 확인 가능
wine.describe() #열에 대한 간략한 통계 출력(최소,최대,평균값 등)

data=wine[['alcohol','sugar','pH']].to_numpy() #판다스 데이터프레임을 넘파이 배열로 바꾸기(처음 3개열 data 배열에 저장, 마지막 class열은 target 배열에 저장)
target=wine['class'].to_numpy()

from sklearn.model_selection import train_test_split #훈련 세트, 테스트 세트로 나누기
train_input,test_input,train_target,test_target=train_test_split(data,target,test_size=0.2,random_state=42)
#샘플의 개수가 충분히 많으므로 test_size=0.2로 20%만 테스트 세트로 나눔(기본값 0.25)
print(train_input.shape,test_input.shape) #만들어진 훈련 세트, 테스트 세트 크기 확인 >> 훈련 세트 5197개, 테스트 세트 1300개

from sklearn.preprocessing import StandardScaler #전처리
ss=StandardScaler()
ss.fit(train_input)
train_scaled=ss.transform(train_input)
test_scaled=ss.transform(test_input)

from sklearn.linear_model import LogisticRegression #로지스틱 회귀 모델 훈련
lr=LogisticRegression()
lr.fit(train_scaled,train_target)
print(lr.score(train_scaled,train_target))
print(lr.score(test_scaled,test_target)) #점수가 높지 않음 >> 화이트 와인 골라내는게 어려움, 훈련 세트와 테스트 세트 점수 모두 낮음 >> 과소적합
print(lr.coef_,lr.intercept_) #계수(가중치,기울기),절편 출력

from sklearn.tree import DecisionTreeClassifier #결정 트리(질문을 통해 예측값을 만들어가는 것, 다른사람에게 설명하기 쉬움), 회귀 모델은 Regressor
dt=DecisionTreeClassifier(random_state=42)
dt.fit(train_scaled,train_target)
dt.fit(train_scaled,train_target)
print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled,test_target)) #훈련 세트는 점수 높지만, 테스트 세트는 점수 낮음 >> 과대적합
import matplotlib.pyplot as plt #그림으로 확인(루트 노드에 여러가지 질문을 통해 최종 리프 노드(불순도가 0인 순수 노드)까지 출력)
from sklearn.tree import plot_tree
plt.figure(figsize=(10,7)) #사이즈 설정
plot_tree(dt)
plt.show()

plt.figure(figsize=(10,7)) #결정트리 분석
# 결정트리는 부모 노드와 자식 노드의 지니 불순도 차이(=정보 이득)가 가능한 크도록 트리를 성장시킴, 그리고 마지막 노드의 클래스 비율을 보고 예측을 만듬
plot_tree(dt,max_depth=1,filled=True,feature_names=('alcohol','sugar','pH'))
#max_depth=1:루트노드 밑에 1개만 더 그려줌,filled=True:색으로 표현해줌,feature_names 설정:훈련데이터 순서와 같게 넣어주면 표현해줌
plt.show() #sugar로 나누는게 효과가 좋아서 처음 sugar로 나눔, value:샘플의 음석클래스,양성클래스의 갯수, gini:지니 불순도
dt=DecisionTreeClassifier(max_depth=3,random_state=42) #가지치기(최대깊이 조정을 통해 결정트리 과대적합 방지(=일반화 잘되게 함)
dt.fit(train_scaled,train_target)
print(dt.score(train_scaled,train_target))
print(dt.score(test_scaled,test_target))
plt.figure(figsize=(20,15))
plot_tree(dt,filled=True,feature_names=['alcohol','sugar','pH'])
plt.show()
dt=DecisionTreeClassifier(max_depth=3,random_state=42) #전처리하지 않은 결정트리(결정트리는 훈련하는게 아니여서 표준화 전처리를 안해도 사용 가능)
dt.fit(train_input,train_target)
print(dt.score(train_input,train_target))
print(dt.score(test_input,test_target))
plt.figure(figsize=(20,15))
plot_tree(dt,filled=True,feature_names=['alcohol','sugar','pH'])
plt.show()
print(dt.feature_importances_) #특성 중요도(어떤 특성이 가장 중요한지) 출력, 0.868인 sugar가 가장 중요한 특성임
