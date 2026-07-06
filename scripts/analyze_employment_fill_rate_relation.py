# scripts/analyze_employment_fill_rate_relation.py

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import statsmodels.formula.api as smf
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "processed" / "freshman_fill_rate_with_employment.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "employment_fill_rate_relation"
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
    """
    CSV 인코딩 문제를 방지하기 위해 여러 인코딩을 순차적으로 시도한다.
    """
    encodings = ["utf-8-sig", "utf-8", "cp949"]

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise UnicodeDecodeError("CSV 파일을 읽을 수 없습니다. 인코딩을 확인하세요.")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    실제 데이터프레임 컬럼명 중 후보명과 일치하는 컬럼을 찾는다.
    """
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
    """
    분석에 사용할 컬럼명을 표준명으로 변경한다.
    """
    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        found_col = find_column(df, aliases)
        if found_col is not None:
            rename_map[found_col] = standard_name

    df = df.rename(columns=rename_map)

    required_columns = ["year", "university", "fill_rate", "employment_rate"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing_columns}\n"
            f"현재 컬럼 목록: {list(df.columns)}"
        )

    return df


def to_numeric(series: pd.Series) -> pd.Series:
    """
    %, 쉼표, 공백 등이 포함된 값을 숫자형으로 변환한다.
    """
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace(["-", "nan", "None", ""], np.nan)
        .astype(float)
    )


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    분석에 필요한 컬럼을 숫자형으로 변환하고 결측치를 정리한다.
    """
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
        if col in df.columns:
            df[col] = to_numeric(df[col])

    df = df.dropna(subset=["year", "university", "fill_rate", "employment_rate"])

    return df


# =========================
# 4. 기초 통계 저장
# =========================

def save_basic_summary(df: pd.DataFrame, analysis_columns: list[str]) -> None:
    summary = df[analysis_columns].describe().T
    summary.to_csv(TABLE_DIR / "basic_summary.csv", encoding="utf-8-sig")

    yearly_summary = df.groupby("year")[analysis_columns].mean()
    yearly_summary.to_csv(TABLE_DIR / "yearly_mean_summary.csv", encoding="utf-8-sig")

    university_summary = (
        df.groupby("university")[analysis_columns]
        .mean()
        .sort_values("fill_rate", ascending=False)
    )
    university_summary.to_csv(TABLE_DIR / "university_mean_summary.csv", encoding="utf-8-sig")


# =========================
# 5. 상관분석 저장
# =========================

def save_correlation_results(df: pd.DataFrame, analysis_columns: list[str]) -> None:
    pearson_corr = df[analysis_columns].corr(method="pearson")
    spearman_corr = df[analysis_columns].corr(method="spearman")

    pearson_corr.to_csv(
        TABLE_DIR / "pearson_correlation_overall.csv",
        encoding="utf-8-sig"
    )

    spearman_corr.to_csv(
        TABLE_DIR / "spearman_correlation_overall.csv",
        encoding="utf-8-sig"
    )

    yearly_rows = []

    for year, group in df.groupby("year"):
        corr = group[analysis_columns].corr(method="pearson")

        for col in analysis_columns:
            if col == "fill_rate":
                continue

            yearly_rows.append({
                "year": year,
                "variable": col,
                "pearson_corr_with_fill_rate": corr.loc["fill_rate", col],
                "n": len(group)
            })

    yearly_corr = pd.DataFrame(yearly_rows)

    yearly_corr.to_csv(
        TABLE_DIR / "yearly_correlation_with_fill_rate.csv",
        index=False,
        encoding="utf-8-sig"
    )


# =========================
# 6. 산점도 저장
# =========================

def plot_scatter_with_trend(
    df: pd.DataFrame,
    x_col: str,
    y_col: str = "fill_rate"
) -> None:
    plot_df = df[[x_col, y_col]].dropna()

    if len(plot_df) < 3:
        print(f"[건너뜀] {x_col}: 유효 데이터가 3개 미만입니다.")
        return

    x = plot_df[x_col]
    y = plot_df[y_col]

    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, alpha=0.7)

    # 추세선
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)

    x_sorted = np.sort(x)
    plt.plot(x_sorted, p(x_sorted))

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    save_path = FIGURE_DIR / f"scatter_{y_col}_vs_{x_col}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


# =========================
# 7. 회귀분석 저장
# =========================

def run_regression(df: pd.DataFrame, predictors: list[str]) -> None:
    available_predictors = [col for col in predictors if col in df.columns]

    if len(available_predictors) == 0:
        with open(OUTPUT_DIR / "regression_result.txt", "w", encoding="utf-8") as f:
            f.write("회귀분석에 사용할 설명변수가 없습니다.\n")
        return

    if not HAS_STATSMODELS:
        with open(OUTPUT_DIR / "regression_result.txt", "w", encoding="utf-8") as f:
            f.write("statsmodels가 설치되어 있지 않아 회귀분석을 실행하지 못했습니다.\n")
            f.write("설치 명령어: pip install statsmodels\n")
        return

    model_columns = ["fill_rate", "year"] + available_predictors
    model_df = df[model_columns].dropna()

    formula = "fill_rate ~ " + " + ".join(available_predictors) + " + C(year)"

    model = smf.ols(formula=formula, data=model_df).fit()

    with open(OUTPUT_DIR / "regression_result.txt", "w", encoding="utf-8") as f:
        f.write("[회귀분석 모형]\n")
        f.write(formula)
        f.write("\n\n")
        f.write(str(model.summary()))

    coef_table = pd.DataFrame({
        "variable": model.params.index,
        "coefficient": model.params.values,
        "std_error": model.bse.values,
        "t_value": model.tvalues.values,
        "p_value": model.pvalues.values,
    })

    coef_table.to_csv(
        TABLE_DIR / "regression_coefficients.csv",
        index=False,
        encoding="utf-8-sig"
    )


# =========================
# 8. 주요 결과 요약 저장
# =========================

def save_interpretation_note(df: pd.DataFrame, analysis_columns: list[str]) -> None:
    pearson_corr = df[analysis_columns].corr(method="pearson")

    lines = []
    lines.append("# 신입생 충원율과 취업성과 관계 분석 요약")
    lines.append("")
    lines.append("## 분석 개요")
    lines.append("본 분석은 2023~2025년 22개 대학을 대상으로 신입생 충원율과 취업률, 진학률, 경쟁률 간의 관계를 탐색하였다.")
    lines.append("신입생 충원율은 입학성과 지표로, 취업률은 대학의 취업성과 지표로 보았다.")
    lines.append("")
    lines.append("## 전체 상관분석 결과")
    lines.append("")

    for col in analysis_columns:
        if col == "fill_rate":
            continue

        corr_value = pearson_corr.loc["fill_rate", col]

        lines.append(
            f"- 신입생 충원율과 {col}의 피어슨 상관계수: {corr_value:.4f}"
        )

    lines.append("")
    lines.append("## 해석 유의사항")
    lines.append("본 분석은 22개 대학의 3개년 자료를 활용한 탐색적 분석이다.")
    lines.append("따라서 분석 결과는 인과관계가 아니라 변수 간 관련성을 파악하는 용도로 해석해야 한다.")
    lines.append("특히 취업률이 신입생 충원율에 직접적인 영향을 준다고 단정하기보다는, 대학 선택과 입학성과에 관련될 수 있는 하나의 성과 지표로 해석하는 것이 적절하다.")

    with open(OUTPUT_DIR / "analysis_note.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 9. 메인 실행
# =========================

def main() -> None:
    print("=" * 60)
    print("신입생 충원율과 취업성과 관계 분석 시작")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_PATH}")

    df = read_csv_safely(INPUT_PATH)

    print(f"원본 데이터 행 수: {len(df)}")
    print(f"원본 컬럼 목록: {list(df.columns)}")

    df = standardize_columns(df)
    df = clean_data(df)

    print(f"정제 후 데이터 행 수: {len(df)}")

    candidate_columns = [
        "fill_rate",
        "employment_rate",
        "advancement_rate",
        "competition_rate",
    ]

    analysis_columns = [col for col in candidate_columns if col in df.columns]

    print(f"분석 변수: {analysis_columns}")

    df.to_csv(
        TABLE_DIR / "analysis_dataset.csv",
        index=False,
        encoding="utf-8-sig"
    )

    save_basic_summary(df, analysis_columns)
    save_correlation_results(df, analysis_columns)

    for x_col in analysis_columns:
        if x_col != "fill_rate":
            plot_scatter_with_trend(df, x_col)

    predictors = [
        "employment_rate",
        "advancement_rate",
        "competition_rate",
    ]

    run_regression(df, predictors)
    save_interpretation_note(df, analysis_columns)

    print("=" * 60)
    print("분석 완료")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()