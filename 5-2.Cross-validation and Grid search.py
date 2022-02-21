import pandas as pd #판다스로 CSV 데이터 읽기
wine=pd.read_csv('https://bit.ly/wine_csv_data')
data=wine[['alcohol','sugar','pH']].to_numpy() #class 열은 타켓으로, 나머지 열은 특성 배열에 저장
target=wine['class'].to_numpy()

from sklearn.model_selection import train_test_split #훈련 세트와 테스트 세트를 나누기
train_input,test_input,train_target,test_target=train_test_split(data,target,test_size=0.2,random_state=42)
#전체의 20% 테스트 세트로 사용
sub_input,val_input,suv_target,val_target=train_test_split(train_input,train_target,test_size=0.2,random_state=42)
#검증 세트는 매개변수 튜닝을 위한 것, 훈련 세트의 20% 검증 세트로 사용
#테스트 세트를 사용해 성능을 확인하면 점점 테스트 세트에 맞춰지므로 마지막에 1번만 쓰는게 좋음 >> 훈련세트에서 모델을 훈련하고, 검증세트로 평가한다. 마지막에 테스트세트 사용
#테스트하고 싶은 매개변수를 바꿔가며 가장 좋은 모델 고르기 >> 매개변수를 사용해 훈련세트, 검증세트를 합쳐 전체 훈련데이터 모델 다시 훈련 >> 마지막에 테스트 세트에서 점수 평가
print(sub_input.shape,val_input.shape) #훈련 세트 5193개 >> 훈련 세트 4157개, 검증 세트 1040개

from sklearn.tree import DecisionTreeClassifier #검증 세트로 결정트리 모델 만들고 평가
dt=DecisionTreeClassifier(random_state=42)
dt.fit(sub_input,suv_target)
print(dt.score(sub_input,suv_target))
print(dt.score(val_input,val_target)) #괴대적합

from sklearn.model_selection import cross_validate #교차 검증(cross_validate)은 기본적으로 5-폴드 교차 검증을 수행하므로 각 키마다 5개의 숫자 있음
#검증 세트를 만드느라 훈련세트가 줄어들었음 >> 교차검증(검증 세트를 떼어내어 평가하는 과정을 여러번 반복)을 통해 안정적인 검증 점수를 얻고, 훈련에 더 많은 데이터 사용
#대부분 머신러닝은 투입 전 교차 검증으로 해야함(딥러닝은x)
score=cross_validate(dt,train_input,train_target)
print(score) #훈련 시간, 평가 시간, 검증 세트 평가 점수 출력
import numpy as np
print(np.mean(score['test_score'])) #교차 검증의 최종 점수는 검증폴드의 점수를 평균하여 얻을 수 있음

from sklearn.model_selection import StratifiedKFold #교차 검증을 할때 훈련 세트를 섞으려면 분할기(spliter)를 사용해야 함
scores=cross_validate(dt,train_input,train_target,cv=StratifiedKFold()) #'dt'가 분류 모델일때 StratifiedKFold, 회귀 모델은 KFold 사용(사이킷런 분할기)
print(np.mean(scores['test_score']))
spliter=StratifiedKFold(n_splits=10,shuffle=True,random_state=42) #spliter객체 따로 만들어 분할기 세부설정 가능(n_splits로 몇 폴드 교차 검증 할지 설정)
scores=cross_validate(dt,train_input,train_target,cv=spliter)
print(np.mean(scores['test_score']))

from sklearn.model_selection import GridSearchCV #그리드 서치(교차 검증(매개변수 바꾸는것) 여러번 실행)
#최적의 매개변수를 순차적으로 찾는게 아니고, 한꺼번에 찾는것이므로 for문으로 찾는게 아니고 그리드 서치 활용
params={'min_impurity_decrease':[0.0001,0.0002,0.0003,0.0004,0.0005]} #매개변수를 딕셔너리로 지정
#모델이 학습할 수 없는 하이퍼파라미터는 사용자가 지정해줘야함(라이브러리가 제공하는 기본값을 그대로 사용해 모델을 훈련 >> 검증세트의 점수나 교차검증을 통해 매개변수 바꿔보기)
gs=GridSearchCV(DecisionTreeClassifier(random_state=42),params,n_jobs=-1) #결정트리 객체, 파라미터 딕셔너리 넣기(n_jobs로 사용 컴퓨터 코어(성능) 설정)
gs.fit(train_input,train_target) #훈련(그리드 서치 객체는 결정트리 모델의 매개변수(min_impurity_decrease) 값을 바꿔가며 실행됨(5폴드 이므로 총 25번)
dt=gs.best_estimator_ #⭐️가장 점수가 높은 매개변수 조합을 dt에 넣기
print(dt.score(train_input,train_target)) #⭐검증세트, 훈련세트 합쳐서 최적의 파라미터로 훈련한 최적모델 출력
print(gs.best_params_) #⭐최적의 파라미터값 출력
print(gs.cv_results_['mean_test_score']) #⭐5번 매개변수 변경하면서 수행한 교차검증의 평균점수

#참고
from scipy.stats import uniform, randint #랜덤 서치를 위한 확률 분포 선택(unifor,randint 모두 균등분포 샘플링) 방법(scipy 이용)
rgen=randint(0,10) #randint는 0~10 사이 정수값을 램덤하게 샘플링
print(rgen.rvs(10)) #10개
ugen=uniform(0,1) #uniform은 0~1 사이 실수값을 랜덤하게 샘플링
print(ugen.rvs(10)) #10개

params={'min_impurity_decrease':uniform(0.0001,0.001),
        'max_depth':randint(20,50),
        'min_samples_split':randint(2,25),
        'min_samples_leaf':randint(1,25),
        } #랜덤 서치(매개변수의 값 범위나 간격을 정하기 어려울때)
from sklearn.model_selection import RandomizedSearchCV
gs=RandomizedSearchCV(DecisionTreeClassifier(random_state=42),params,n_iter=100,n_jobs=-1,random_state=42) #랜덤서치 클래스 사용(100번 실행)
gs.fit(train_input,train_target) #랜덤서치 로 훈련
print(gs.best_params_) #랜덤서치 최적 매개변수 조합
print(np.max(gs.cv_results_['mean_test_score'])) #랜덤서치 최고 교차 검증 점수

dt=gs.best_estimator_ #최적의 모델을 최종 모델로 결정
print(dt.score(test_input,test_target)) #테스트 세트의 점수 확인