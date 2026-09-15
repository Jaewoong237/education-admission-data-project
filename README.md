# Education Admission Data Project

> **실제 공공데이터의 구축·검증부터 통계분석과 머신러닝 모형 비교까지**  
> 대학 입학성과를 사례로 데이터 처리, 분석, 해석, 시각화의 전 과정을 수행한 프로젝트

[Project Summary · 1페이지 요약](reports/project_summary/education_admission_project_summary.pdf) · [Full Report · 최종보고서](reports/final_report/education_admission_final_report.pdf) · [발표자료](reports/presentation/education_admission_project_presentation_final.pdf) · [분석 코드](scripts/)

## Overview | 프로젝트 개요

서로 다른 공공데이터의 지표 정의와 기준연도를 확인하고, 분석 가능한 데이터셋으로 통합한 뒤 통계적 관계와 예측 가능성을 탐색하였다. 분석 사례는 **2023~2025년 22개 대학의 입학성과**이며, 대학알리미·KOSIS·교육통계서비스 자료를 활용하였다.

핵심은 원자료의 다중 헤더 처리와 지표 재계산, 데이터 검증, 탐색적 분석, 연도를 통제한 회귀분석, 머신러닝 모형 및 기준모형 비교, Power BI 시각화를 연결한 경험이다. 교육 분야의 구체적인 문제를 통해 실제 데이터의 품질과 분석 결과의 해석 가능성을 함께 다루었다.

주요 질문은 다음과 같다.

- 지역의 18세 인구와 고교 졸업·진학 지표는 대학 입학성과와 어떤 관계가 있는가?
- 취업률·졸업생 진학률·경쟁률은 신입생 충원율과 어떻게 관련되는가?
- 여러 지표를 활용한 예측모형과 대학별 유형화는 어떤 정보를 제공하며, 해석의 한계는 무엇인가?

## Data | 데이터

| 출처 | 활용 자료 | 분석 단위 및 역할 |
| --- | --- | --- |
| 대학알리미 | 신입생 충원현황, 졸업생 취업현황 | 22개 대학 × 3개 공시연도, 총 **66개 대학-연도 관측치** |
| KOSIS 국가통계포털 | 2023~2025년 18세 인구 | 지역·연도별 외부 환경 지표 |
| 교육통계서비스 | 고교 졸업자수·진학자수·진학률 | 지역·연도별 진학 환경 지표 |

- **분석 대상 지표:** 정원내 신입생 충원율. 경쟁률, 취업률, 대학 졸업생 진학률, 대학 규모·지역·설립구분 등을 함께 활용하였다.
- **연도 정합성:** 취업성과의 공시연도 2023~2025년은 자료연도 2022~2024년에 대응한다. 두 연도를 별도 변수로 관리하였다.
- **지표 구분:** 고등학교 진학률과 대학 졸업생 진학률은 서로 다른 지표이다. 아래 취업성과 관계분석과 예측모형의 진학률은 **대학 졸업생 진학률**을 뜻한다.
- **검증:** 취업률 계산분모와 취업률·진학률을 원자료에서 재계산하여 66건 모두 검증을 통과하였다. 신입생 충원현황도 최종 대학-연도 66건 모두 검증을 통과하였다.

[대상 대학 목록](data/interim/target_universities.csv) · [가공 데이터](data/processed/) · [데이터 검증 결과](reports/data_validation/) · [데이터 확보 검토 기록](docs/data_availability_check.md)

## Methods | 분석 방법

| 단계 | 수행 내용 |
| --- | --- |
| 전처리·검증 | Excel 다중 헤더 처리, 대학별 자료 집계, 지표 재계산, 대학·연도 키 및 원자료 대조 |
| 탐색적 분석 | 기술통계, 연도·지역·대학 그룹별 비교, 산점도 및 분포 시각화 |
| 통계분석 | Pearson·Spearman 상관분석, 연도 더미변수를 포함한 다중회귀분석 |
| 예측모형 | 평균 예측 기준모형, 선형회귀, Ridge, Random Forest 비교; 수치형 표준화와 범주형 인코딩을 모형별 파이프라인에 구성 |
| 모형 평가 | 무작위 훈련/테스트 분할(75%/25%) 및 셔플을 적용한 5-fold 교차검증, MAE·RMSE·R² 비교 |
| 유형화·전달 | 3개년 평균과 충원율 변동성의 임계값에 따른 규칙 기반 유형화, 전략 점수, Power BI 대시보드 |

## Key Results | 핵심 결과

아래 수치는 저장소의 현재 분석 결과표를 기준으로 정리하였다. 상세 수치와 해석은 각 근거 링크에서 확인할 수 있다.

### 1. 상관관계와 다중회귀 결과

신입생 충원율과 경쟁률은 **r=0.254 (p=.039)**, 대학 졸업생 진학률은 **r=0.264 (p=.032)**의 양의 상관을 보였다. 취업률과의 상관은 **r=-0.077 (p=.539)**로 통계적으로 유의하지 않았다.

연도를 통제한 다중회귀의 **R²=0.178, 수정 R²=0.110, 모형 p=.034**였다. 다만 경쟁률(p=.056), 졸업생 진학률(p=.107), 취업률(p=.078)의 개별 계수는 모두 5% 유의수준을 충족하지 않았다. 단순 상관과 다른 변수를 함께 고려한 결과를 구분하여 해석하였다.

근거: [상관계수·p값](reports/spss_validation/spss_validation_pearson_correlations.csv), [회귀계수](reports/employment_fill_rate_relation/tables/regression_coefficients.csv), [회귀모형 요약](reports/spss_validation/spss_validation_regression_model_summary.csv)

### 2. 예측모형 비교와 일반화의 한계

| 모형 | 테스트 RMSE | 테스트 R² | 5-fold 평균 RMSE | 5-fold 평균 R² |
| --- | ---: | ---: | ---: | ---: |
| 평균 예측 기준모형 | 2.264 | -0.016 | 1.916 | -0.359 |
| 선형회귀 | 1.661 | 0.453 | 1.960 | -2.381 |
| Ridge | 1.831 | 0.335 | 1.861 | -1.418 |
| Random Forest | 1.628 | 0.475 | 2.008 | -1.485 |

RMSE의 단위는 충원율의 **퍼센트포인트(%p)**이다. 단일 테스트셋에서는 Random Forest의 RMSE가 가장 낮았지만, 교차검증 평균 RMSE는 Ridge가 가장 낮았다. 모든 모형의 교차검증 평균 R²가 음수여서 안정적인 일반화 성능을 확보했다고 해석하기 어렵다. Random Forest에서는 졸업생 진학률과 경쟁률의 변수 중요도가 높았으나, 이를 인과적 영향으로 해석하지 않는다.

근거: [테스트 성능](reports/fill_rate_prediction_model/tables/model_performance_train_test.csv), [교차검증 성능](reports/fill_rate_prediction_model/tables/model_performance_cross_validation.csv), [변수 중요도](reports/fill_rate_prediction_model/tables/random_forest_feature_importance.csv)

### 3. 대학별 지표를 활용한 탐색적 유형화

3개년 평균 지표와 충원율 변동성을 바탕으로 충원 안정형 5개, 입학성과 보완 필요형 5개, 충원 안정·수요 강세형 4개, 취업성과 기반 잠재형 3개, 관찰 필요형 3개, 입시 수요 강세형 2개로 구분하였다. 이는 표본 내부의 임계값을 사용한 **규칙 기반 상대적 진단**이며, 대학의 절대적 순위나 학습된 군집모형의 결과가 아니다.

근거: [유형별 분포](reports/admission_strategy_types/tables/strategy_type_counts.csv), [분류 기준 및 해석](reports/admission_strategy_types/strategy_insight_note.md)

## Pipeline | 분석 흐름

**원자료 수집 → 전처리·지표 재계산 → 원자료 대조 → 대학·연도 및 지역·연도 기준 통합 → EDA·통계분석 → 예측모형 비교·유형화 → 대시보드·보고서**

| 작업 | 주요 코드 |
| --- | --- |
| 충원현황·취업성과 전처리 | [filter_freshman_fill_rate_v2.py](scripts/filter_freshman_fill_rate_v2.py), [process_employment_rate.py](scripts/process_employment_rate.py) |
| 인구·고교 지표 전처리 | [process_kosis_population_age18.py](scripts/process_kosis_population_age18.py), [process_highschool_graduation.py](scripts/process_highschool_graduation.py) |
| 원자료 검증 | [validate_freshman_fill_rate.py](scripts/validate_freshman_fill_rate.py), [validate_employment_processed.py](scripts/validate_employment_processed.py) |
| 자료 통합 | [merge_freshman_fill_rate_population.py](scripts/merge_freshman_fill_rate_population.py), [merge_population_graduation.py](scripts/merge_population_graduation.py), [merge_freshman_fill_rate_employment.py](scripts/merge_freshman_fill_rate_employment.py) |
| EDA·관계분석 | [eda_freshman_fill_rate.py](scripts/eda_freshman_fill_rate.py), [인구 관계분석](scripts/analyze_population_fill_rate_relation.py), [고교 지표 관계분석](scripts/analyze_graduation_fill_rate_relation.py), [취업성과 관계분석](scripts/analyze_employment_fill_rate_relation.py) |
| 모형 비교·유형화 | [analyze_fill_rate_prediction_model.py](scripts/analyze_fill_rate_prediction_model.py), [analyze_university_admission_strategy_types.py](scripts/analyze_university_admission_strategy_types.py) |
| 대시보드 데이터 생성 | [build_powerbi_dashboard_dataset.py](scripts/build_powerbi_dashboard_dataset.py) |

실행에 필요한 라이브러리는 [requirements.txt](requirements.txt)에 기록하였다. 원자료와 중간 산출물을 준비한 뒤 각 스크립트의 입력·출력 경로 및 위 단계의 의존관계를 확인하여 실행한다.

## Outputs & Reports | 주요 산출물

| 산출물 | 내용 |
| --- | --- |
| [1페이지 요약서](reports/project_summary/education_admission_project_summary.pdf) / [최종보고서](reports/final_report/education_admission_final_report.pdf) | 프로젝트 개요와 연구문제별 분석 내용 |
| [발표자료 PDF](reports/presentation/education_admission_project_presentation_final.pdf) | 분석 흐름과 주요 결과 발표 |
| [관계분석 결과](reports/employment_fill_rate_relation/) / [예측모형 결과](reports/fill_rate_prediction_model/) / [유형화 결과](reports/admission_strategy_types/) | 결과표, 그래프, 분석 메모 |
| [인구·고교 지표 분석표](data/processed/) / [EDA 그래프](outputs/figures/) | 외부 환경 지표와 입학성과 관계 탐색 |
| [Power BI 대시보드](reports/powerbi_dashboard_dataset/education_admission_dashboard.pbix) / [대시보드 데이터셋·가이드](reports/powerbi_dashboard_dataset/) | 대학-연도 지표, 유형, 모형 결과 시각화 |
| [검증 산출물](reports/data_validation/) / [통계 재현 결과](reports/spss_validation/) | 원자료 대조 및 SAV 데이터 기반 Python 통계분석 |
| [Documentation](docs/) | 프로젝트 계획, 데이터 확보 검토, 문헌검색 및 논문 리뷰 기록 |

## Repository Structure | 저장소 구조

```text
education-admission-data-project/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/          # 출처별 원자료
│   ├── interim/      # 분석 대상 대학 목록
│   └── processed/    # 가공·통합 데이터 및 인구·고교 지표 분석표
├── scripts/          # 전처리·검증·분석·시각화 코드
├── outputs/figures/  # 탐색적 분석 그래프
├── reports/          # 검증·관계분석·예측·유형화·대시보드·최종 산출물
└── docs/             # 계획 및 문헌 검토 기록
```

## Tech Stack | 사용 도구

- **데이터 처리:** Python, pandas, NumPy, openpyxl
- **통계·머신러닝:** SciPy, statsmodels, scikit-learn
- **시각화·결과 전달:** Matplotlib, Power BI
- **분석 기록·검증 자료:** Git/GitHub, pyreadstat(SAV 파일 읽기)

SPSS용 SAV 데이터셋과 이를 읽어 수행한 Python 통계 재현 결과를 포함한다. **SPSS GUI에서의 교차검증은 라이선스 문제로 보류**되어 있다.

## Limitations | 한계

- **표본·기간:** 22개 대학의 3개년 자료이므로 전체 대학이나 장기 추세로 일반화하기 어렵다.
- **반복 관측:** 동일 대학이 여러 연도에 반복된다. 현재 예측 평가는 행 단위 무작위 분할로, 대학 단위 분리나 미래 연도 예측을 검증한 결과가 아니다.
- **관찰자료:** 상관관계·회귀계수·변수 중요도는 인과적 효과를 입증하지 않는다. 지역 단위 지표를 개인의 선택이나 행동으로 해석할 수도 없다.
- **지표·시점:** 취업성과의 공시연도와 자료연도 사이에 시차가 있으며, 실제 사전 예측을 위해서는 각 설명변수의 확보 시점을 추가로 점검해야 한다.
- **예측·유형화:** 교차검증 성능이 불안정하고, 유형화는 표본 내 임계값에 의존한다. 실제 운영을 위한 예측모형이나 절대적 대학 평가로 사용하기에는 한계가 있다.

## Future Work | 향후 연구 방향

- 대학 수와 관측 기간을 늘리고 학과·전공계열 수준으로 분석 단위를 확장한다.
- 대학 단위 분리 검증, 시간 순서에 따른 검증, 반복 관측을 고려한 통계모형을 적용하여 결과의 안정성을 점검한다.
- 시계열 분석과 Statistical Learning·Machine Learning 방법론을 학습·적용하고, 기준모형 대비 성능과 해석 가능성을 함께 평가한다.
- 이 프로젝트에서 다룬 데이터 구축·검증·분석 경험을 **사회·공공·산업 분야의 다양한 실제 데이터와 예측 문제**로 확장한다.
