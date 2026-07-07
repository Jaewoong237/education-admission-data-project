# 신입생 충원율 예측모형 분석

## 분석 개요
본 분석은 2023~2025년 22개 대학 자료를 활용하여 신입생 충원율을 설명하거나 예측하는 주요 요인을 탐색하기 위해 수행하였다.
종속변수는 신입생 충원율이며, 설명변수로 경쟁률, 취업률, 진학률, 연도, 지역, 설립구분, 수도권/지방, 분석그룹 등을 활용하였다.

## 사용 변수

### 수치형 변수
- competition_rate
- employment_rate
- advancement_rate
- year
- admission_quota
- recruitment_quota
- applicants

### 범주형 변수
- region
- foundation_type
- metro_local
- representative_group
- analysis_group

## 모형 비교 결과

훈련/검증 데이터 분할 결과 기준으로 가장 낮은 테스트 RMSE를 보인 모형은 다음과 같다.

- 최적 모형: Linear_Regression
- Test MAE: 2.138
- Test RMSE: 2.484
- Test R-squared: 0.552

5-fold 교차검증 기준 결과는 다음과 같다.

- CV MAE 평균: 2.921
- CV RMSE 평균: 3.858
- CV R-squared 평균: -0.425

## 랜덤포레스트 변수 중요도 상위 변수

- competition_rate: 0.4400
- advancement_rate: 0.3096
- admission_quota: 0.0610
- employment_rate: 0.0562
- region_경북: 0.0308
- applicants: 0.0263
- recruitment_quota: 0.0181
- year: 0.0156
- analysis_group_수도권-사립-비교: 0.0073
- analysis_group_지방-사립-비교: 0.0054

## 해석
본 분석은 신입생 충원율을 단순히 관찰하는 데서 나아가, 경쟁률·취업률·진학률·지역 특성 등이 충원율 예측에 어느 정도 기여하는지 확인하기 위한 탐색적 예측모형 분석이다.
다만 분석 대상이 22개 대학의 3개년 자료, 총 66개 관측치로 제한되어 있으므로 예측 성능은 일반화보다는 변수 간 관계와 설명 가능성을 확인하는 데 초점을 두어 해석해야 한다.
따라서 본 결과는 대학 입학성과를 예측하는 최종 모형이라기보다, 입시 전략 수립을 위한 주요 지표의 상대적 중요도를 확인하는 보조 분석으로 활용하는 것이 적절하다.