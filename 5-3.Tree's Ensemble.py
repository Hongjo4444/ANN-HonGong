#트리의 앙상블:랜덤한 여러개의 트리를 합쳐서 1개의 성능이 좋은 모델을 만드는 것(정형 데이터 다를때 가장 성과가 좋은 알고리즘, 비정형 데이터는 신경망 알고리즘이 제일 좋음)

import numpy as np #와인 데이터셋을 판다스로 불러오고, 훈련세트와 테스트세트로 나누기
import pandas as pd
from sklearn.model_selection import train_test_split
wine=pd.read_csv('https://bit.ly/wine_csv_data')
data=wine[['alcohol','sugar','pH']]
target=wine['class'].to_numpy()
train_input,test_input,train_target,test_target=train_test_split(data,target,test_size=0.2,random_state=42)

from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier #랜덤 포레스트 훈련:결정트리를 랜덤하게 만들어서 결정트리의 숲을 만들고, 각 결정트리의 예측으로 최종 예측 만듬
#랜덤 포레스트 훈련방법:훈련세트를 랜덤샘플링해서 부트스트랩 샘플을 만들고(뽑은것 중복으로 뽑을수도 있음), 부트스트랩 샘플로 결정 트리 훈련
rf=RandomForestClassifier(n_jobs=-1,random_state=42) #각 노드를 분할할때 전체 특성 중 일부를 무작위로 고른 후 이 중에서 최선의 분할 찾음
#분류는 제곱근만큼 특성 사용, 회귀는 전체 특성 사용
scores=cross_validate(rf,train_input,train_target,return_train_score=True,n_jobs=-1) #return_train_score로 훈련세트의 점수 출력
print(np.mean(scores['train_score']),np.mean(scores['test_score'])) #훈련세트의 점수, 검증세트의 점수 출력
rf.fit(train_input,train_target) #랜덤 포레스트 모델을 훈련 세트에 훈련
print(rf.feature_importances_) #특성 중요도 출력
rf=RandomForestClassifier(oob_score=True,n_jobs=-1,random_state=42) #oob_score로 트리를 만들때 남는 샘플(oob 샘플) 출력
rf.fit(train_input,train_target)
print(rf.oob_score_) #oob_score로 검증세트처럼 성능 평가(oob점수를 사용하면 교차검증을 대신할 수 있어 훈련세트에 더 많은 샘플 사용 가능)

from sklearn.ensemble import ExtraTreesClassifier #엑스트라 트리 훈련:과대적합을 막고, 검증세트의 점수를 높이는 효과
#엑스트라 트리 훈련방법:부트스트랩 샘플 사용X(즉각 결정트리를 만들때 전체 훈련세트 사용), 노드를 분할할때 가장 좋은 분할을 찾는것이 아니라 무작위로 분할(무작위여서 많이해야함)
et=ExtraTreesClassifier(n_jobs=-1,random_state=42)
scores=cross_validate(et,train_input,train_target,return_train_score=True,n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))
et.fit(train_input,train_target)
print(et.feature_importances_)

from  sklearn.ensemble import GradientBoostingClassifier #그레디언트 부스팅:깊이가 얕은 결정트리로 이전 트리의 오차 보완 >> 과대적합에 강함, 높은 일반화 성능
#그레디언트 부스팅 훈련방법:결정트리를 계속 추가하면서 로지스틱 손실(분류) or 평균 제곱 오차 함수(회귀)가 가장 낮은 곳을 찾음
gb=GradientBoostingClassifier(random_state=42)
scores=cross_validate(gb,train_input,train_target,return_train_score=True,n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))
gb=GradientBoostingClassifier(n_estimators=500,learning_rate=0.2,random_state=42) #n_estimators로 트리 개수 조정, learning_rate로 학습률 제어
scores=cross_validate(gb,train_input,train_target,return_train_score=True,n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))
gb.fit(train_input,train_target)
print(gb.feature_importances_)

from sklearn.experimental import enable_hist_gradient_boosting #⭐️히스토그램 기반 그레디언트 부스팅:그레디언트 부스팅의 속도,성능을 개선한 것
#히스토그램 기반 그레디언트 부스팅 훈련방법: 입력 특성을 256개 구간으로 나누고 각 구간별로 학습(노드를 분할할때 최적의 분할을 매우 빠르게 찾을 수 있음)
from sklearn.ensemble import HistGradientBoostingClassifier
hgb=HistGradientBoostingClassifier(random_state=42)
scores=cross_validate(hgb,train_input,train_target,return_train_score=True,n_jobs=-1)
print(np.mean(scores['train_score']),np.mean(scores['test_score']))
from sklearn.inspection import permutation_importance #치환 중요도:특성을 하나씩 랜덤으로 섞어서 성능 변화 관찰 >> 어떤 특성이 중요한지 계산
hgb.fit(train_input,train_target)
result=permutation_importance(hgb,train_input,train_target,n_repeats=10,random_state=42,n_jobs=-1) #n_repeats로 섞는 횟수 지정
print(result.importances_mean)
result=permutation_importance(hgb,test_input,test_target,n_repeats=10,random_state=42,n_jobs=-1)
print(result.importances_mean)
print(hgb.score(test_input,test_target))

# from xgboost import XGBClassifier #XGBoost:히스토그램 기반 그레디언트 부스팅 라이브러리 종류
# xgb=XGBClassifier(tree_method='hist',random_state=42)
# scores=cross_validate(xgb,train_input,train_target,return_train_score=True,n_jobs=-1)
# print(np.mean(scores['train_score']),np.mean(scores['test_score']))
#
# from lightgbm import LGBMClassifier #LightGBM:히스토그램 기반 그레디언트 부스팅 라이브러리 종류
# lgb=LGBMClassifier(random_state=42)
# scores=cross_validate(lgb,train_input,train_target,return_train_score=True,n_jobs=-1)
# print(np.mean(scores['train_score']),np.mean(scores['test_score']))