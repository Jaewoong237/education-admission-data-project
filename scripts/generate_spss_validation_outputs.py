from pathlib import Path

import pandas as pd
import pyreadstat
import statsmodels.formula.api as smf
from scipy.stats import pearsonr


INPUT_PATH = Path(
    "reports/spss_validation/spss_fill_rate_validation_dataset.sav"
)
OUTPUT_DIR = Path("reports/spss_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 1. SPSS 검증용 SAV 불러오기
df, meta = pyreadstat.read_sav(INPUT_PATH)

variables = [
    "fill_rate",
    "competition_rate",
    "employment_rate",
    "advancement_rate",
]


# 2. 기술통계
descriptive = (
    df[variables]
    .describe()
    .T[["count", "mean", "std", "min", "max"]]
    .reset_index()
    .rename(columns={"index": "variable"})
)

descriptive.to_csv(
    OUTPUT_DIR / "spss_validation_descriptive_statistics.csv",
    index=False,
    encoding="utf-8-sig",
)


# 3. Pearson 상관분석
correlation_rows = []

for x in [
    "competition_rate",
    "employment_rate",
    "advancement_rate",
]:
    result = pearsonr(df["fill_rate"], df[x])

    correlation_rows.append(
        {
            "variable_1": "fill_rate",
            "variable_2": x,
            "pearson_r": result.statistic,
            "p_value": result.pvalue,
            "n": len(df),
        }
    )

correlations = pd.DataFrame(correlation_rows)

correlations.to_csv(
    OUTPUT_DIR / "spss_validation_pearson_correlations.csv",
    index=False,
    encoding="utf-8-sig",
)


# 4. 다중회귀분석
model = smf.ols(
    "fill_rate ~ employment_rate + advancement_rate "
    "+ competition_rate + C(year)",
    data=df,
).fit()


# 회귀계수표
conf_int = model.conf_int()

coefficients = pd.DataFrame(
    {
        "variable": model.params.index,
        "B": model.params.values,
        "std_error": model.bse.values,
        "t": model.tvalues.values,
        "p_value": model.pvalues.values,
        "ci_95_lower": conf_int[0].values,
        "ci_95_upper": conf_int[1].values,
    }
)

coefficients.to_csv(
    OUTPUT_DIR / "spss_validation_regression_coefficients.csv",
    index=False,
    encoding="utf-8-sig",
)


# 회귀모형 요약 지표
model_summary = pd.DataFrame(
    [
        {
            "n": int(model.nobs),
            "r_squared": model.rsquared,
            "adjusted_r_squared": model.rsquared_adj,
            "f_statistic": model.fvalue,
            "f_p_value": model.f_pvalue,
            "df_model": int(model.df_model),
            "df_resid": int(model.df_resid),
            "aic": model.aic,
            "bic": model.bic,
        }
    ]
)

model_summary.to_csv(
    OUTPUT_DIR / "spss_validation_regression_model_summary.csv",
    index=False,
    encoding="utf-8-sig",
)


# Statsmodels 전체 출력도 TXT로 저장
with open(
    OUTPUT_DIR / "spss_validation_regression_summary.txt",
    "w",
    encoding="utf-8",
) as f:
    f.write(model.summary().as_text())


print("=== SPSS validation outputs generated ===")
print(f"Dataset shape: {df.shape}")
print()
print("[Descriptive statistics]")
print(descriptive.to_string(index=False))
print()
print("[Pearson correlations]")
print(correlations.to_string(index=False))
print()
print("[Regression model]")
print(
    f"R²={model.rsquared:.6f}, "
    f"Adj.R²={model.rsquared_adj:.6f}, "
    f"F={model.fvalue:.6f}, "
    f"p={model.f_pvalue:.6f}"
)
print()
print("Output directory:", OUTPUT_DIR)