# scripts/analyze_university_admission_strategy_types.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "processed" / "freshman_fill_rate_with_employment.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "admission_strategy_types"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 컬럼명 후보 설정
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

    raise UnicodeDecodeError("CSV 파일을 읽을 수 없습니다. 인코딩을 확인하세요.")


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

    df = df.rename(columns=rename_map)

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

    return df


def to_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace(["-", "nan", "None", ""], np.nan)
        .astype(float)
    )


def minmax_scale(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()

    if pd.isna(min_value) or pd.isna(max_value) or min_value == max_value:
        return pd.Series(0.5, index=series.index)

    return (series - min_value) / (max_value - min_value)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    numeric_columns = [
        "fill_rate",
        "employment_rate",
        "advancement_rate",
        "competition_rate",
    ]

    for col in numeric_columns:
        df[col] = to_numeric(df[col])

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
# 4. 대학별 종합지표 생성
# =========================

def build_university_summary(df: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "region",
        "foundation_type",
        "metro_local",
        "representative_group",
        "analysis_group",
    ]

    existing_metadata = [col for col in metadata_columns if col in df.columns]

    agg_dict = {
        "fill_rate": ["mean", "std", "min", "max"],
        "competition_rate": ["mean", "std"],
        "employment_rate": ["mean", "std"],
        "advancement_rate": ["mean", "std"],
    }

    summary = df.groupby("university").agg(agg_dict)

    summary.columns = [
        "_".join(col).strip()
        for col in summary.columns.to_flat_index()
    ]

    summary = summary.reset_index()

    if existing_metadata:
        metadata = (
            df.groupby("university")[existing_metadata]
            .first()
            .reset_index()
        )
        summary = summary.merge(metadata, on="university", how="left")

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
    })

    summary["std_fill_rate"] = summary["std_fill_rate"].fillna(0)
    summary["std_competition_rate"] = summary["std_competition_rate"].fillna(0)
    summary["std_employment_rate"] = summary["std_employment_rate"].fillna(0)
    summary["std_advancement_rate"] = summary["std_advancement_rate"].fillna(0)

    return summary


# =========================
# 5. 전략 점수 및 유형 분류
# =========================

def add_strategy_scores(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()

    summary["fill_score"] = minmax_scale(summary["avg_fill_rate"])
    summary["competition_score"] = minmax_scale(summary["avg_competition_rate"])
    summary["employment_score"] = minmax_scale(summary["avg_employment_rate"])

    # 충원율 변동성이 낮을수록 안정성이 높다고 봄
    summary["stability_score"] = 1 - minmax_scale(summary["std_fill_rate"])

    summary["strategy_score"] = (
        summary["fill_score"] * 0.40
        + summary["competition_score"] * 0.25
        + summary["employment_score"] * 0.20
        + summary["stability_score"] * 0.15
    )

    summary["strategy_rank"] = summary["strategy_score"].rank(
        ascending=False,
        method="min"
    ).astype(int)

    summary["fill_rate_rank"] = summary["avg_fill_rate"].rank(
        ascending=False,
        method="min"
    ).astype(int)

    summary["competition_rank"] = summary["avg_competition_rate"].rank(
        ascending=False,
        method="min"
    ).astype(int)

    summary["employment_rank"] = summary["avg_employment_rate"].rank(
        ascending=False,
        method="min"
    ).astype(int)

    return summary


def classify_university(row: pd.Series, thresholds: dict) -> str:
    high_fill = row["avg_fill_rate"] >= thresholds["median_fill"]
    high_competition = row["avg_competition_rate"] >= thresholds["median_competition"]
    high_employment = row["avg_employment_rate"] >= thresholds["q75_employment"]
    stable_fill = row["std_fill_rate"] <= thresholds["median_fill_std"]

    low_fill = row["avg_fill_rate"] < thresholds["median_fill"]
    low_competition = row["avg_competition_rate"] < thresholds["median_competition"]

    if high_fill and high_competition and stable_fill:
        return "충원 안정·수요 강세형"

    if high_fill and stable_fill:
        return "충원 안정형"

    if high_fill and high_competition:
        return "입시 수요 강세형"

    if high_employment and low_fill:
        return "취업성과 기반 잠재형"

    if low_fill and low_competition:
        return "입학성과 보완 필요형"

    return "관찰 필요형"


def add_strategy_types(summary: pd.DataFrame) -> pd.DataFrame:
    thresholds = {
        "median_fill": summary["avg_fill_rate"].median(),
        "median_competition": summary["avg_competition_rate"].median(),
        "q75_employment": summary["avg_employment_rate"].quantile(0.75),
        "median_fill_std": summary["std_fill_rate"].median(),
    }

    summary["strategy_type"] = summary.apply(
        lambda row: classify_university(row, thresholds),
        axis=1
    )

    return summary, thresholds


def make_strategy_message(strategy_type: str) -> str:
    messages = {
        "충원 안정·수요 강세형":
            "충원율과 경쟁률이 모두 양호하고 변동성도 낮은 유형이다. 현재의 모집 경쟁력을 유지하면서 우수 사례로 관리할 필요가 있다.",
        "충원 안정형":
            "신입생 충원율이 안정적으로 유지되는 유형이다. 급격한 모집 확대보다는 안정적인 지원자 풀 관리와 학과별 강점 홍보가 적절하다.",
        "입시 수요 강세형":
            "경쟁률과 충원율이 비교적 높아 입시 수요가 확인되는 유형이다. 지원자 관심을 실제 등록으로 연결하는 등록 유도 전략이 중요하다.",
        "취업성과 기반 잠재형":
            "취업성과는 높지만 입학성과가 상대적으로 약한 유형이다. 취업률, 졸업 후 진로, 산학협력 성과를 입시 홍보 메시지로 강화할 필요가 있다.",
        "입학성과 보완 필요형":
            "충원율과 경쟁률이 모두 상대적으로 낮은 유형이다. 모집단위별 수요 점검, 지역 맞춤 홍보, 전형 구조 개선 등 보완 전략이 필요하다.",
        "관찰 필요형":
            "일부 지표는 양호하지만 뚜렷한 유형으로 분류되지는 않는 대학이다. 세부 학과 단위 분석과 연도별 변화 추적이 필요하다.",
    }

    return messages.get(strategy_type, "추가적인 세부 분석이 필요하다.")


def add_strategy_messages(summary: pd.DataFrame) -> pd.DataFrame:
    summary["strategy_insight"] = summary["strategy_type"].apply(make_strategy_message)
    return summary


# =========================
# 6. 결과 저장
# =========================

def save_tables(summary: pd.DataFrame, thresholds: dict) -> None:
    summary_sorted = summary.sort_values("strategy_rank")

    summary_sorted.to_csv(
        TABLE_DIR / "university_strategy_type_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    type_counts = (
        summary["strategy_type"]
        .value_counts()
        .rename_axis("strategy_type")
        .reset_index(name="count")
    )

    type_counts.to_csv(
        TABLE_DIR / "strategy_type_counts.csv",
        index=False,
        encoding="utf-8-sig"
    )

    ranking_columns = [
        "strategy_rank",
        "university",
        "strategy_score",
        "strategy_type",
        "avg_fill_rate",
        "std_fill_rate",
        "avg_competition_rate",
        "avg_employment_rate",
        "avg_advancement_rate",
    ]

    existing_ranking_columns = [
        col for col in ranking_columns if col in summary_sorted.columns
    ]

    summary_sorted[existing_ranking_columns].to_csv(
        TABLE_DIR / "strategy_score_ranking.csv",
        index=False,
        encoding="utf-8-sig"
    )

    thresholds_df = pd.DataFrame(
        [{"threshold": key, "value": value} for key, value in thresholds.items()]
    )

    thresholds_df.to_csv(
        TABLE_DIR / "classification_thresholds.csv",
        index=False,
        encoding="utf-8-sig"
    )


def save_figures(summary: pd.DataFrame) -> None:
    type_counts = summary["strategy_type"].value_counts()

    plt.figure(figsize=(9, 5))
    type_counts.plot(kind="bar")
    plt.title("Strategy Type Counts")
    plt.xlabel("Strategy Type")
    plt.ylabel("Number of Universities")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "strategy_type_counts.png", dpi=300)
    plt.close()

    top10 = summary.sort_values("strategy_score", ascending=False).head(10)

    plt.figure(figsize=(9, 5))
    plt.bar(top10["university"], top10["strategy_score"])
    plt.title("Top 10 Universities by Strategy Score")
    plt.xlabel("University")
    plt.ylabel("Strategy Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "top10_strategy_score.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(summary["avg_competition_rate"], summary["avg_fill_rate"], alpha=0.8)
    plt.title("Average Fill Rate vs Average Competition Rate")
    plt.xlabel("Average Competition Rate")
    plt.ylabel("Average Fill Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "scatter_fill_rate_vs_competition_rate_by_university.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.scatter(summary["avg_employment_rate"], summary["avg_fill_rate"], alpha=0.8)
    plt.title("Average Fill Rate vs Average Employment Rate")
    plt.xlabel("Average Employment Rate")
    plt.ylabel("Average Fill Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "scatter_fill_rate_vs_employment_rate_by_university.png", dpi=300)
    plt.close()


def save_analysis_note(summary: pd.DataFrame, thresholds: dict) -> None:
    type_counts = summary["strategy_type"].value_counts()
    top5 = summary.sort_values("strategy_score", ascending=False).head(5)

    lines = []
    lines.append("# 대학별 입학성과 유형화 및 입시 전략 인사이트")
    lines.append("")
    lines.append("## 분석 개요")
    lines.append("본 분석은 2023~2025년 22개 대학의 신입생 충원율, 경쟁률, 취업률, 진학률을 종합하여 대학별 입학성과 유형을 분류하고 입시 전략 인사이트를 도출하기 위해 수행하였다.")
    lines.append("대학별 3개년 평균 지표와 충원율 변동성을 산출하고, 충원율·경쟁률·취업률·안정성 지표를 종합한 전략 점수를 계산하였다.")
    lines.append("")
    lines.append("## 분류 기준")
    lines.append(f"- 평균 신입생 충원율 중앙값: {thresholds['median_fill']:.3f}")
    lines.append(f"- 평균 경쟁률 중앙값: {thresholds['median_competition']:.3f}")
    lines.append(f"- 평균 취업률 75분위수: {thresholds['q75_employment']:.3f}")
    lines.append(f"- 신입생 충원율 표준편차 중앙값: {thresholds['median_fill_std']:.3f}")
    lines.append("")
    lines.append("## 대학 유형별 분포")
    lines.append("")

    for strategy_type, count in type_counts.items():
        lines.append(f"- {strategy_type}: {count}개 대학")

    lines.append("")
    lines.append("## 전략 점수 상위 대학")
    lines.append("")

    for _, row in top5.iterrows():
        lines.append(
            f"- {row['university']}: 전략점수 {row['strategy_score']:.3f}, 유형 {row['strategy_type']}"
        )

    lines.append("")
    lines.append("## 해석")
    lines.append("신입생 충원율이 안정적으로 유지되고 경쟁률이 높은 대학은 입시 수요가 비교적 견고한 유형으로 볼 수 있다.")
    lines.append("반면 취업률이 높지만 충원율이 상대적으로 낮은 대학은 취업성과를 입시 홍보와 모집 전략에 더 적극적으로 연결할 필요가 있다.")
    lines.append("충원율과 경쟁률이 모두 낮은 대학은 모집단위별 수요 점검, 지역 맞춤형 홍보, 전형 구조 개선 등 입학성과 보완 전략이 필요하다.")
    lines.append("")
    lines.append("## 유의사항")
    lines.append("본 분석은 2023~2025년 3개년, 22개 대학 자료를 바탕으로 한 탐색적 유형화 분석이다.")
    lines.append("따라서 분석 결과는 대학의 절대적 우열을 평가하기보다, 입시 전략 수립을 위한 상대적 진단 지표로 해석하는 것이 적절하다.")

    with open(OUTPUT_DIR / "strategy_insight_note.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 7. 메인 실행
# =========================

def main() -> None:
    print("=" * 60)
    print("대학별 입학성과 유형화 및 입시 전략 인사이트 분석 시작")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_PATH}")

    df = read_csv_safely(INPUT_PATH)

    print(f"원본 데이터 행 수: {len(df)}")
    print(f"원본 컬럼 목록: {list(df.columns)}")

    df = standardize_columns(df)
    df = clean_data(df)

    print(f"정제 후 데이터 행 수: {len(df)}")

    summary = build_university_summary(df)
    summary = add_strategy_scores(summary)
    summary, thresholds = add_strategy_types(summary)
    summary = add_strategy_messages(summary)

    save_tables(summary, thresholds)
    save_figures(summary)
    save_analysis_note(summary, thresholds)

    print(f"분석 대학 수: {len(summary)}")
    print("대학 유형 분포:")
    print(summary["strategy_type"].value_counts())

    print("=" * 60)
    print("분석 완료")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()