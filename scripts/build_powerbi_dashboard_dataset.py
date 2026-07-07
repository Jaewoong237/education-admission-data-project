# scripts/build_powerbi_dashboard_dataset.py

from pathlib import Path
import numpy as np
import pandas as pd


# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

MAIN_INPUT_PATH = BASE_DIR / "data" / "processed" / "freshman_fill_rate_with_employment.csv"

STRATEGY_SUMMARY_PATH = (
    BASE_DIR
    / "reports"
    / "admission_strategy_types"
    / "tables"
    / "university_strategy_type_summary.csv"
)

STRATEGY_TYPE_COUNTS_PATH = (
    BASE_DIR
    / "reports"
    / "admission_strategy_types"
    / "tables"
    / "strategy_type_counts.csv"
)

MODEL_PERFORMANCE_PATH = (
    BASE_DIR
    / "reports"
    / "fill_rate_prediction_model"
    / "tables"
    / "model_performance_train_test.csv"
)

CV_PERFORMANCE_PATH = (
    BASE_DIR
    / "reports"
    / "fill_rate_prediction_model"
    / "tables"
    / "model_performance_cross_validation.csv"
)

FEATURE_IMPORTANCE_PATH = (
    BASE_DIR
    / "reports"
    / "fill_rate_prediction_model"
    / "tables"
    / "random_forest_feature_importance.csv"
)

OUTPUT_DIR = BASE_DIR / "reports" / "powerbi_dashboard_dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 컬럼명 후보
# =========================

COLUMN_ALIASES = {
    "year": [
        "year", "연도", "기준연도"
    ],
    "university": [
        "university", "university_name", "대학명", "학교명"
    ],
    "region": [
        "region", "지역"
    ],
    "foundation_type": [
        "foundation_type", "설립구분"
    ],
    "metro_local": [
        "metro_local", "수도권/지방"
    ],
    "representative_group": [
        "representative_group", "대표/비교"
    ],
    "analysis_group": [
        "analysis_group", "분석그룹"
    ],
    "fill_rate": [
        "fill_rate",
        "freshman_fill_rate",
        "신입생충원율",
        "신입생 충원율",
        "충원율",
        "충원율(%)",
        "신입생 충원율(%)",
        "정원내 신입생 충원율(%)"
    ],
    "employment_rate": [
        "employment_rate",
        "취업률",
        "취업률(%)",
        "졸업생취업률",
        "졸업생 취업률",
        "취업률_원자료_평균"
    ],
    "advancement_rate": [
        "advancement_rate",
        "high_school_advancement_rate",
        "진학률",
        "진학률(%)",
        "고등학교진학률",
        "고등학교 진학률",
        "졸업자_대비_진학자비율"
    ],
    "competition_rate": [
        "competition_rate",
        "admission_competition_rate",
        "경쟁률",
        "입학경쟁률",
        "입학 경쟁률"
    ],
    "admission_quota": [
        "admission_quota", "입학정원"
    ],
    "recruitment_quota": [
        "recruitment_quota", "모집인원"
    ],
    "applicants": [
        "applicants", "지원자"
    ],
    "entrants": [
        "entrants", "입학자"
    ],
    "graduates": [
        "graduates", "졸업자"
    ],
    "employed": [
        "employed", "취업자"
    ],
    "advanced": [
        "advanced", "진학자"
    ],
}


# =========================
# 3. 유틸 함수
# =========================

def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949"]

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError(f"CSV 파일을 읽을 수 없습니다: {path}")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized_columns:
            return normalized_columns[key]

    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        found_col = find_column(df, aliases)
        if found_col is not None:
            rename_map[found_col] = standard_name

    return df.rename(columns=rename_map)


def to_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace(["-", "nan", "None", ""], np.nan)
        .astype(float)
    )


def clean_main_data(df: pd.DataFrame) -> pd.DataFrame:
    df = standardize_columns(df)

    required_columns = [
        "year",
        "university",
        "fill_rate",
        "employment_rate",
        "advancement_rate",
        "competition_rate",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing_columns}\n"
            f"현재 컬럼 목록: {list(df.columns)}"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    numeric_columns = [
        "fill_rate",
        "employment_rate",
        "advancement_rate",
        "competition_rate",
        "admission_quota",
        "recruitment_quota",
        "applicants",
        "entrants",
        "graduates",
        "employed",
        "advanced",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    categorical_columns = [
        "university",
        "region",
        "foundation_type",
        "metro_local",
        "representative_group",
        "analysis_group",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "미상")

    df = df.dropna(
        subset=[
            "university",
            "fill_rate",
            "employment_rate",
            "advancement_rate",
            "competition_rate",
        ]
    )

    return df


# =========================
# 4. Power BI용 데이터셋 생성
# =========================

def build_year_panel(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "year",
        "university",
        "region",
        "foundation_type",
        "metro_local",
        "representative_group",
        "analysis_group",
        "admission_quota",
        "recruitment_quota",
        "applicants",
        "entrants",
        "fill_rate",
        "competition_rate",
        "graduates",
        "employed",
        "advanced",
        "employment_rate",
        "advancement_rate",
    ]

    existing_columns = [col for col in columns if col in df.columns]

    panel = df[existing_columns].copy()

    if "applicants" in panel.columns and "recruitment_quota" in panel.columns:
        panel["applicants_per_recruitment"] = (
            panel["applicants"] / panel["recruitment_quota"]
        ).replace([np.inf, -np.inf], np.nan)

    if "entrants" in panel.columns and "recruitment_quota" in panel.columns:
        panel["entrants_per_recruitment"] = (
            panel["entrants"] / panel["recruitment_quota"]
        ).replace([np.inf, -np.inf], np.nan)

    panel = panel.sort_values(["university", "year"])

    return panel


def build_university_dashboard_summary(
    panel: pd.DataFrame,
    strategy_summary: pd.DataFrame | None
) -> pd.DataFrame:
    agg_dict = {
        "fill_rate": ["mean", "std", "min", "max"],
        "competition_rate": ["mean", "std"],
        "employment_rate": ["mean", "std"],
        "advancement_rate": ["mean", "std"],
    }

    for optional_col in ["admission_quota", "recruitment_quota", "applicants", "entrants"]:
        if optional_col in panel.columns:
            agg_dict[optional_col] = ["mean"]

    summary = panel.groupby("university").agg(agg_dict)

    summary.columns = [
        "_".join(col).strip()
        for col in summary.columns.to_flat_index()
    ]

    summary = summary.reset_index()

    summary = summary.rename(columns={
        "fill_rate_mean": "avg_fill_rate",
        "fill_rate_std": "std_fill_rate",
        "fill_rate_min": "min_fill_rate",
        "fill_rate_max": "max_fill_rate",
        "competition_rate_mean": "avg_competition_rate",
        "competition_rate_std": "std_competition_rate",
        "employment_rate_mean": "avg_employment_rate",
        "employment_rate_std": "std_employment_rate",
        "advancement_rate_mean": "avg_advancement_rate",
        "advancement_rate_std": "std_advancement_rate",
        "admission_quota_mean": "avg_admission_quota",
        "recruitment_quota_mean": "avg_recruitment_quota",
        "applicants_mean": "avg_applicants",
        "entrants_mean": "avg_entrants",
    })

    metadata_columns = [
        "region",
        "foundation_type",
        "metro_local",
        "representative_group",
        "analysis_group",
    ]

    existing_metadata = [col for col in metadata_columns if col in panel.columns]

    if existing_metadata:
        metadata = panel.groupby("university")[existing_metadata].first().reset_index()
        summary = summary.merge(metadata, on="university", how="left")

    if strategy_summary is not None:
        strategy_columns = [
            col for col in [
                "university",
                "strategy_score",
                "strategy_rank",
                "strategy_type",
                "strategy_insight",
                "fill_score",
                "competition_score",
                "employment_score",
                "stability_score",
            ]
            if col in strategy_summary.columns
        ]

        summary = summary.merge(
            strategy_summary[strategy_columns],
            on="university",
            how="left"
        )

    summary = summary.sort_values("avg_fill_rate", ascending=False)

    return summary


def build_research_question_summary() -> pd.DataFrame:
    rows = [
        {
            "research_question": "RQ1",
            "title": "학령인구·고교 졸업/진학 지표와 신입생 충원율 관계",
            "main_method": "EDA, 통합 데이터 분석, 상관분석",
            "main_output_path": "reports/fill_rate_population_relation/",
            "dashboard_use": "연도별 외부 교육환경 변화와 충원율 흐름 확인",
            "summary": "학령인구와 고교 졸업·진학 지표를 신입생 충원율과 연결하여 외부 교육환경과 입학성과의 관계를 탐색하였다.",
        },
        {
            "research_question": "RQ2",
            "title": "취업성과와 신입생 충원율 관계",
            "main_method": "상관분석, 회귀분석",
            "main_output_path": "reports/employment_fill_rate_relation/",
            "dashboard_use": "취업률·진학률·경쟁률과 충원율 관계 확인",
            "summary": "취업률은 충원율과 약한 양의 상관을 보였으나, 회귀분석에서는 진학률과 경쟁률이 더 뚜렷한 설명력을 보였다.",
        },
        {
            "research_question": "RQ3",
            "title": "대학별 입학성과 유형화 및 입시 전략 인사이트",
            "main_method": "대학별 평균 지표, 안정성 지표, 전략 점수, 유형 분류",
            "main_output_path": "reports/admission_strategy_types/",
            "dashboard_use": "대학별 전략 유형, 전략 점수, 유형별 분포 확인",
            "summary": "22개 대학을 입학성과 보완 필요형, 관찰 필요형, 충원 안정·수요 강세형 등으로 분류하였다.",
        },
        {
            "research_question": "RQ4",
            "title": "신입생 충원율 예측모형 분석",
            "main_method": "선형회귀, 릿지회귀, 랜덤포레스트, 교차검증",
            "main_output_path": "reports/fill_rate_prediction_model/",
            "dashboard_use": "예측모형 성능과 주요 변수 중요도 확인",
            "summary": "단일 검증셋에서는 선형회귀가 우수했고, 교차검증 기준에서는 랜덤포레스트가 상대적으로 안정적인 성능을 보였다.",
        },
        {
            "research_question": "RQ5",
            "title": "입학성과 종합 진단 및 대시보드 구성",
            "main_method": "Power BI용 데이터셋 구성, 종합 인사이트 정리",
            "main_output_path": "reports/powerbi_dashboard_dataset/",
            "dashboard_use": "최종 대시보드 및 보고서·PPT 연결",
            "summary": "연구문제 1~4의 결과를 통합하여 대학별 입학성과를 종합적으로 진단하고 시각화할 수 있는 데이터셋을 구성하였다.",
        },
    ]

    return pd.DataFrame(rows)


def build_dashboard_kpi_summary(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []

    rows.append({
        "kpi": "분석 대상 대학 수",
        "value": panel["university"].nunique(),
        "description": "분석에 포함된 대학 수"
    })

    rows.append({
        "kpi": "분석 연도 수",
        "value": panel["year"].nunique(),
        "description": "분석에 포함된 연도 수"
    })

    rows.append({
        "kpi": "전체 평균 신입생 충원율",
        "value": panel["fill_rate"].mean(),
        "description": "2023~2025년 22개 대학 전체 평균 신입생 충원율"
    })

    rows.append({
        "kpi": "전체 평균 경쟁률",
        "value": panel["competition_rate"].mean(),
        "description": "2023~2025년 22개 대학 전체 평균 경쟁률"
    })

    rows.append({
        "kpi": "전체 평균 취업률",
        "value": panel["employment_rate"].mean(),
        "description": "2023~2025년 22개 대학 전체 평균 취업률"
    })

    rows.append({
        "kpi": "전체 평균 진학률",
        "value": panel["advancement_rate"].mean(),
        "description": "2023~2025년 22개 대학 전체 평균 진학률"
    })

    return pd.DataFrame(rows)


def build_powerbi_guide() -> None:
    lines = []

    lines.append("# Power BI 대시보드 구성 가이드")
    lines.append("")
    lines.append("## 1. 사용 데이터")
    lines.append("")
    lines.append("- `powerbi_university_year_panel.csv`: 대학-연도 단위 패널 데이터")
    lines.append("- `powerbi_university_summary.csv`: 대학별 3개년 평균 및 전략 유형 요약 데이터")
    lines.append("- `powerbi_strategy_type_counts.csv`: 전략 유형별 대학 수")
    lines.append("- `powerbi_model_performance.csv`: 예측모형 성능 비교")
    lines.append("- `powerbi_cv_model_performance.csv`: 교차검증 성능 비교")
    lines.append("- `powerbi_feature_importance.csv`: 랜덤포레스트 변수 중요도")
    lines.append("- `powerbi_research_question_summary.csv`: 연구문제별 분석 요약")
    lines.append("- `powerbi_kpi_summary.csv`: 대시보드 핵심 KPI 요약")
    lines.append("")
    lines.append("## 2. 권장 대시보드 페이지")
    lines.append("")
    lines.append("### Page 1. 프로젝트 개요")
    lines.append("- 카드: 분석 대학 수, 분석 연도 수, 평균 충원율, 평균 경쟁률, 평균 취업률")
    lines.append("- 표: 연구문제별 분석 방법 및 주요 결과")
    lines.append("")
    lines.append("### Page 2. 입학성과 현황")
    lines.append("- 막대그래프: 대학별 평균 신입생 충원율")
    lines.append("- 꺾은선그래프: 연도별 평균 신입생 충원율")
    lines.append("- 슬라이서: 연도, 지역, 수도권/지방, 분석그룹")
    lines.append("")
    lines.append("### Page 3. 취업성과 및 입시지표 관계")
    lines.append("- 산점도: 취업률 vs 신입생 충원율")
    lines.append("- 산점도: 경쟁률 vs 신입생 충원율")
    lines.append("- 산점도: 진학률 vs 신입생 충원율")
    lines.append("")
    lines.append("### Page 4. 대학별 전략 유형")
    lines.append("- 막대그래프: 전략 유형별 대학 수")
    lines.append("- 표: 대학별 전략 유형, 전략 점수, 전략 인사이트")
    lines.append("- 슬라이서: 전략 유형, 지역, 수도권/지방")
    lines.append("")
    lines.append("### Page 5. 예측모형 결과")
    lines.append("- 막대그래프: 모형별 RMSE 비교")
    lines.append("- 막대그래프: 랜덤포레스트 변수 중요도")
    lines.append("- 산점도: 실제 충원율 vs 예측 충원율")
    lines.append("")
    lines.append("## 3. 대시보드 핵심 메시지")
    lines.append("")
    lines.append("본 대시보드는 신입생 충원율을 중심으로 학령인구 흐름, 취업성과, 경쟁률, 진학률, 대학별 전략 유형, 예측모형 결과를 통합적으로 확인할 수 있도록 구성한다.")
    lines.append("이를 통해 대학별 입학성과를 단일 지표가 아니라 외부 환경, 입시 수요, 취업성과, 안정성, 예측 가능성을 종합하여 진단할 수 있다.")

    with open(OUTPUT_DIR / "powerbi_dashboard_guide.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 5. 저장 함수
# =========================

def save_optional_dataset(source_path: Path, output_name: str) -> None:
    if source_path.exists():
        df = read_csv_safely(source_path)
        df.to_csv(OUTPUT_DIR / output_name, index=False, encoding="utf-8-sig")
    else:
        print(f"[건너뜀] 파일 없음: {source_path}")


def main() -> None:
    print("=" * 60)
    print("Power BI 대시보드용 데이터셋 생성 시작")
    print("=" * 60)

    if not MAIN_INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {MAIN_INPUT_PATH}")

    main_df = read_csv_safely(MAIN_INPUT_PATH)
    main_df = clean_main_data(main_df)

    print(f"메인 데이터 행 수: {len(main_df)}")
    print(f"분석 대학 수: {main_df['university'].nunique()}")
    print(f"분석 연도 수: {main_df['year'].nunique()}")

    strategy_summary = None

    if STRATEGY_SUMMARY_PATH.exists():
        strategy_summary = read_csv_safely(STRATEGY_SUMMARY_PATH)
    else:
        print(f"[주의] 전략 유형 요약 파일이 없습니다: {STRATEGY_SUMMARY_PATH}")

    year_panel = build_year_panel(main_df)
    university_summary = build_university_dashboard_summary(
        year_panel,
        strategy_summary
    )
    research_question_summary = build_research_question_summary()
    kpi_summary = build_dashboard_kpi_summary(year_panel)

    year_panel.to_csv(
        OUTPUT_DIR / "powerbi_university_year_panel.csv",
        index=False,
        encoding="utf-8-sig"
    )

    university_summary.to_csv(
        OUTPUT_DIR / "powerbi_university_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    research_question_summary.to_csv(
        OUTPUT_DIR / "powerbi_research_question_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    kpi_summary.to_csv(
        OUTPUT_DIR / "powerbi_kpi_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    save_optional_dataset(
        STRATEGY_TYPE_COUNTS_PATH,
        "powerbi_strategy_type_counts.csv"
    )

    save_optional_dataset(
        MODEL_PERFORMANCE_PATH,
        "powerbi_model_performance.csv"
    )

    save_optional_dataset(
        CV_PERFORMANCE_PATH,
        "powerbi_cv_model_performance.csv"
    )

    save_optional_dataset(
        FEATURE_IMPORTANCE_PATH,
        "powerbi_feature_importance.csv"
    )

    build_powerbi_guide()

    print("")
    print("생성 파일:")
    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"- {path.name}")

    print("=" * 60)
    print("Power BI 대시보드용 데이터셋 생성 완료")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()