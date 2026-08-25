from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from adjustText import adjust_text

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# =========================================================
# 1. 기본 경로
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_TABLE_DIR = (
    BASE_DIR
    / "reports"
    / "fill_rate_prediction_model"
    / "tables"
)

PREDICTION_PATH = (
    MODEL_TABLE_DIR
    / "test_prediction_results.csv"
)

IMPORTANCE_PATH = (
    MODEL_TABLE_DIR
    / "random_forest_feature_importance.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "presentation"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# 2. 한글 폰트
# =========================================================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 3. 디자인 기준
# =========================================================

YEAR_COLORS = {
    2023: "#1f77b4",
    2024: "#ff7f0e",
    2025: "#2ca02c",
}

GROUP_MARKERS = {
    "대표대학": "o",
    "비교대학": "^",
}


# =========================================================
# 4. 데이터 불러오기
# =========================================================

pred = pd.read_csv(
    PREDICTION_PATH,
    encoding="utf-8-sig"
)

importance = pd.read_csv(
    IMPORTANCE_PATH,
    encoding="utf-8-sig"
)


required_prediction_columns = [
    "year",
    "university",
    "representative_group",
    "actual_fill_rate",
    "pred_Random_Forest",
]

missing_prediction_columns = [
    col
    for col in required_prediction_columns
    if col not in pred.columns
]

if missing_prediction_columns:
    raise ValueError(
        "예측결과 CSV에 필요한 컬럼이 없습니다: "
        f"{missing_prediction_columns}"
    )


required_importance_columns = [
    "feature",
    "importance",
]

missing_importance_columns = [
    col
    for col in required_importance_columns
    if col not in importance.columns
]

if missing_importance_columns:
    raise ValueError(
        "변수 중요도 CSV에 필요한 컬럼이 없습니다: "
        f"{missing_importance_columns}"
    )


# =========================================================
# 5. 데이터형 정리
# =========================================================

pred["year"] = pd.to_numeric(
    pred["year"],
    errors="coerce"
).astype("Int64")

pred["actual_fill_rate"] = pd.to_numeric(
    pred["actual_fill_rate"],
    errors="coerce"
)

pred["pred_Random_Forest"] = pd.to_numeric(
    pred["pred_Random_Forest"],
    errors="coerce"
)

pred = pred.dropna(
    subset=[
        "year",
        "university",
        "actual_fill_rate",
        "pred_Random_Forest",
    ]
).copy()

importance["importance"] = pd.to_numeric(
    importance["importance"],
    errors="coerce"
)

importance = importance.dropna(
    subset=[
        "feature",
        "importance",
    ]
).copy()


# =========================================================
# 6. Random Forest 테스트 성능 재검증
# =========================================================

y_true = pred["actual_fill_rate"]
y_pred = pred["pred_Random_Forest"]

mae = mean_absolute_error(
    y_true,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_true,
        y_pred
    )
)

r2 = r2_score(
    y_true,
    y_pred
)


# =========================================================
# 7. 실제값 vs 예측값 그래프
# =========================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.3)
)


# ---------------------------------------------------------
# 연도 = 색상
# 대표/비교 = 도형
# ---------------------------------------------------------

for year in [2023, 2024, 2025]:

    year_df = pred[
        pred["year"] == year
    ]

    for group in [
        "대표대학",
        "비교대학",
    ]:

        group_df = year_df[
            year_df["representative_group"]
            == group
        ]

        if group_df.empty:
            continue

        ax.scatter(
            group_df["actual_fill_rate"],
            group_df["pred_Random_Forest"],

            s=65,

            c=YEAR_COLORS[year],

            marker=GROUP_MARKERS[group],

            alpha=0.82,

            edgecolors="white",
            linewidths=0.8,

            zorder=3,
        )


# ---------------------------------------------------------
# 45도 기준선
# 실제값 = 예측값
# ---------------------------------------------------------

all_min = min(
    pred["actual_fill_rate"].min(),
    pred["pred_Random_Forest"].min(),
)

all_max = max(
    pred["actual_fill_rate"].max(),
    pred["pred_Random_Forest"].max(),
)

padding = 0.7

line_min = all_min - padding
line_max = all_max + padding

ax.plot(
    [line_min, line_max],
    [line_min, line_max],

    linestyle="--",
    linewidth=1.4,

    color="gray",
    alpha=0.70,

    zorder=2,
)


# =========================================================
# 8. 오차가 큰 관측치 자동 표시
# =========================================================

pred["absolute_error"] = (
    pred["actual_fill_rate"]
    - pred["pred_Random_Forest"]
).abs()


# 오차가 가장 큰 5개 관측치 표시
label_df = (
    pred
    .sort_values(
        "absolute_error",
        ascending=False
    )
    .head(4)
    .copy()
)


texts = []

for _, row in label_df.iterrows():

    label = (
        f"{row['university']} "
        f"({int(row['year'])})"
    )

    texts.append(
        ax.text(
            row["actual_fill_rate"],
            row["pred_Random_Forest"],

            label,

            fontsize=7.2,
            fontweight="bold",

            alpha=0.90,

            zorder=5,
        )
    )


if texts:

    adjust_text(
        texts,

        ax=ax,

        arrowprops=dict(
            arrowstyle="-",
            lw=0.5,
            color="gray",
            alpha=0.55,
        ),

        expand=(1.15, 1.22),

        force_text=(0.7, 0.9),

        force_static=(0.25, 0.40),

        ensure_inside_axes=True,
    )


# =========================================================
# 9. 범례
# =========================================================

year_handles = [
    Line2D(
        [0],
        [0],

        marker="o",
        linestyle="",

        markerfacecolor=YEAR_COLORS[year],
        markeredgecolor="white",

        markersize=8,

        label=str(year),
    )

    for year in [
        2023,
        2024,
        2025,
    ]
]


group_handles = [

    Line2D(
        [0],
        [0],

        marker="o",
        linestyle="",

        markerfacecolor="gray",
        markeredgecolor="white",

        markersize=8,

        label="대표대학",
    ),

    Line2D(
        [0],
        [0],

        marker="^",
        linestyle="",

        markerfacecolor="gray",
        markeredgecolor="white",

        markersize=8,

        label="비교대학",
    ),
]


legend_year = ax.legend(
    handles=year_handles,
    title="연도",
    loc="lower left",
    fontsize=8,
    title_fontsize=8,
    frameon=True,
)

ax.add_artist(
    legend_year
)


ax.legend(
    handles=group_handles,

    title="구분",

    loc="lower right",

    fontsize=8,
    title_fontsize=8,

    frameon=True,
)


# =========================================================
# 10. 실제값 vs 예측값 그래프 디자인
# =========================================================

ax.set_title(
    "Random Forest 실제값과 예측값",
    fontsize=14,
    fontweight="bold",
    pad=12,
)

ax.set_xlabel(
    "실제 신입생 충원율(%)",
    fontsize=10,
)

ax.set_ylabel(
    "예측 신입생 충원율(%)",
    fontsize=10,
)

ax.set_xlim(
    line_min,
    line_max,
)

ax.set_ylim(
    line_min,
    line_max,
)

ax.grid(
    alpha=0.20,
    linewidth=0.8,
)

ax.tick_params(
    axis="both",
    labelsize=8.5,
)


# ---------------------------------------------------------
# 성능은 검증 목적으로 그래프에도 작게 표시
# ---------------------------------------------------------

metric_text = (
    f"MAE = {mae:.3f}\n"
    f"RMSE = {rmse:.3f}\n"
    f"R² = {r2:.3f}"
)

ax.text(
    0.03,
    0.97,

    metric_text,

    transform=ax.transAxes,

    va="top",
    ha="left",

    fontsize=9,
    fontweight="bold",

    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#bbbbbb",
        alpha=0.92,
    ),
)


fig.tight_layout()


prediction_output_path = (
    OUTPUT_DIR
    / "rq4_random_forest_actual_vs_predicted_발표용.png"
)

fig.savefig(
    prediction_output_path,

    dpi=300,

    bbox_inches="tight",

    facecolor="white",
)

plt.close(fig)


# =========================================================
# 11. 변수 중요도 라벨 변환
# =========================================================

FEATURE_LABELS = {
    "advancement_rate":
        "대학 졸업생 진학률",

    "competition_rate":
        "경쟁률",

    "employment_rate":
        "취업률",

    "admission_quota":
        "입학정원",

    "applicants":
        "지원자 수",

    "recruitment_quota":
        "모집인원",

    "year":
        "연도",

    "region_경북":
        "지역: 경북",
}


def convert_feature_label(feature):

    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]

    if feature.startswith("region_"):
        return (
            "지역: "
            + feature.replace(
                "region_",
                ""
            )
        )

    if feature.startswith(
        "analysis_group_"
    ):
        return (
            "분석그룹: "
            + feature.replace(
                "analysis_group_",
                ""
            )
        )

    if feature.startswith(
        "metro_local_"
    ):
        return (
            "권역: "
            + feature.replace(
                "metro_local_",
                ""
            )
        )

    if feature.startswith(
        "foundation_type_"
    ):
        return (
            "설립: "
            + feature.replace(
                "foundation_type_",
                ""
            )
        )

    if feature.startswith(
        "representative_group_"
    ):
        return (
            "대표/비교: "
            + feature.replace(
                "representative_group_",
                ""
            )
        )

    return feature


# =========================================================
# 12. 상위 8개 변수 중요도
# =========================================================

top_importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
    .head(8)
    .copy()
)

top_importance["feature_label"] = (
    top_importance["feature"]
    .apply(
        convert_feature_label
    )
)


# 그래프에서 높은 값이 위로 오도록 뒤집기
plot_importance = (
    top_importance
    .sort_values(
        "importance",
        ascending=True
    )
)


# =========================================================
# 13. 변수 중요도 그래프
# =========================================================

fig, ax = plt.subplots(
    figsize=(7.2, 5.3)
)

bars = ax.barh(
    plot_importance[
        "feature_label"
    ],

    plot_importance[
        "importance"
    ],
)


# ---------------------------------------------------------
# 막대 끝 숫자 표시
# ---------------------------------------------------------

for bar, value in zip(
    bars,
    plot_importance["importance"]
):

    ax.text(
        value + 0.006,
        bar.get_y()
        + bar.get_height() / 2,

        f"{value:.3f}",

        va="center",

        fontsize=8.5,
    )


ax.set_title(
    "Random Forest 변수 중요도",
    fontsize=14,
    fontweight="bold",
    pad=12,
)

ax.set_xlabel(
    "상대적 변수 중요도",
    fontsize=10,
)

ax.set_ylabel(
    "",
)

ax.grid(
    axis="x",
    alpha=0.20,
    linewidth=0.8,
)

ax.tick_params(
    axis="both",
    labelsize=8.5,
)


# 오른쪽 숫자가 잘리지 않도록 여백
max_importance = (
    plot_importance[
        "importance"
    ].max()
)

ax.set_xlim(
    0,
    max_importance * 1.20
)


fig.tight_layout()


importance_output_path = (
    OUTPUT_DIR
    / "rq4_random_forest_feature_importance_발표용.png"
)

fig.savefig(
    importance_output_path,

    dpi=300,

    bbox_inches="tight",

    facecolor="white",
)

plt.close(fig)


# =========================================================
# 14. 검증 출력
# =========================================================

print("=" * 75)
print(
    "PPT RQ4 발표용 그래프 생성 완료"
)
print("=" * 75)


print()
print(
    "[Random Forest 단일 테스트셋]"
)

print(
    f"테스트 관측치 수: {len(pred)}"
)

print(
    f"MAE  = {mae:.6f}"
)

print(
    f"RMSE = {rmse:.6f}"
)

print(
    f"R²   = {r2:.6f}"
)

print(
    f"저장 = {prediction_output_path}"
)


print()
print(
    "[오차가 큰 테스트 관측치 TOP 5]"
)

print(
    label_df[
        [
            "year",
            "university",
            "actual_fill_rate",
            "pred_Random_Forest",
            "absolute_error",
        ]
    ]
    .to_string(
        index=False
    )
)


print()
print(
    "[Random Forest 변수 중요도 TOP 8]"
)

print(
    top_importance[
        [
            "feature",
            "feature_label",
            "importance",
        ]
    ]
    .to_string(
        index=False
    )
)

print(
    f"\n저장 = "
    f"{importance_output_path}"
)


print()
print(
    "※ 변수 중요도는 "
    "인과효과 또는 설명력 비율을 의미하지 않습니다."
)

print("=" * 75)