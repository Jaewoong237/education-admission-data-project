from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from adjustText import adjust_text
from scipy.stats import pearsonr


# =========================================================
# 1. 기본 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "freshman_fill_rate_with_employment.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "presentation"
    / "figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# =========================================================
# 2. 디자인 기준
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


# ---------------------------------------------------------
# 평상시에도 이름을 보여줄 핵심 대학
# ---------------------------------------------------------
# 교수님 피드백의 "대표/비교 대학 이름 표시"를 반영하되
# 22개 전체를 표시해 그래프가 난잡해지는 것은 방지
# ---------------------------------------------------------

KEY_UNIVERSITIES = [
    "부산외국어대학교",
    "부산대학교",
    "부경대학교",
    "울산대학교",
    "경북대학교",
    "서울대학교",
    "연세대학교",
    "고려대학교",
    "성균관대학교",
]


# ---------------------------------------------------------
# 이 충원율 미만이면 반드시 라벨 표시
# ---------------------------------------------------------
# 하단의 특이 관측치를 교수님이 질문할 가능성에 대비
# 대학명 + 연도를 표시
# ---------------------------------------------------------

OUTLIER_FILL_THRESHOLD = 98.0


# =========================================================
# 3. 데이터 불러오기
# =========================================================

df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8-sig"
)

required_columns = [
    "기준연도",
    "대학명",
    "대표/비교",
    "정원내 신입생 충원율(%)",
    "경쟁률",
    "취업률(%)",
    "진학률(%)",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_columns}"
    )


# =========================================================
# 4. 데이터형 정리
# =========================================================

df["기준연도"] = pd.to_numeric(
    df["기준연도"],
    errors="coerce"
).astype("Int64")

numeric_columns = [
    "정원내 신입생 충원율(%)",
    "경쟁률",
    "취업률(%)",
    "진학률(%)",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =========================================================
# 5. p-value 표시 형식
# =========================================================

def format_p_value(p):

    if p < 0.001:
        return "p < .001"

    return f"p = {p:.3f}".replace(
        "0.",
        "."
    )


# =========================================================
# 6. 산점도 생성 함수
# =========================================================

def make_scatter(
    data,
    x_col,
    y_col,
    title,
    x_label,
    y_label,
    output_name,
):

    plot_df = data[
        [
            "기준연도",
            "대학명",
            "대표/비교",
            x_col,
            y_col,
        ]
    ].dropna().copy()


    # =====================================================
    # Pearson 상관분석
    # =====================================================

    result = pearsonr(
        plot_df[x_col],
        plot_df[y_col]
    )

    r = result.statistic
    p = result.pvalue


    # =====================================================
    # 그래프 생성
    # =====================================================

    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )


    # =====================================================
    # 6-1. 산점도
    # =====================================================

    for year in [2023, 2024, 2025]:

        year_df = plot_df[
            plot_df["기준연도"] == year
        ]

        for group in [
            "대표대학",
            "비교대학",
        ]:

            group_df = year_df[
                year_df["대표/비교"] == group
            ]

            if group_df.empty:
                continue

            ax.scatter(
                group_df[x_col],
                group_df[y_col],

                s=48,

                c=YEAR_COLORS[year],

                marker=GROUP_MARKERS[
                    group
                ],

                alpha=0.80,

                edgecolors="white",
                linewidths=0.7,

                zorder=3,
            )


    # =====================================================
    # 6-2. 일반 대학 라벨
    #
    # 핵심 대학은 2025년 점에만 대학명 표시
    # =====================================================

    key_label_df = plot_df[
        (
            plot_df["기준연도"] == 2025
        )
        &
        (
            plot_df["대학명"].isin(
                KEY_UNIVERSITIES
            )
        )
        &
        (
            plot_df[y_col]
            >= OUTLIER_FILL_THRESHOLD
        )
    ].copy()


    # =====================================================
    # 6-3. 하단 특이 관측치
    #
    # 충원율 98% 미만은 모든 연도 표시
    # "대학명 (연도)" 형식
    # =====================================================

    outlier_df = plot_df[
        plot_df[y_col]
        < OUTLIER_FILL_THRESHOLD
    ].copy()


    texts = []


    # -----------------------------------------------------
    # 핵심 대학 라벨
    # -----------------------------------------------------

    for _, row in key_label_df.iterrows():

        texts.append(
            ax.text(
                row[x_col],
                row[y_col],

                row["대학명"],

                fontsize=6.3,
                alpha=0.85,

                zorder=4,
            )
        )


    # -----------------------------------------------------
    # 하단 특이 관측치 라벨
    # -----------------------------------------------------

    for _, row in outlier_df.iterrows():

        label = (
            f"{row['대학명']} "
            f"({int(row['기준연도'])})"
        )

        texts.append(
            ax.text(
                row[x_col],
                row[y_col],

                label,

                fontsize=6.2,
                fontweight="bold",

                alpha=0.92,

                zorder=5,
            )
        )


    # =====================================================
    # 6-4. 라벨 겹침 자동 조정
    # =====================================================

    if texts:

        adjust_text(
            texts,

            ax=ax,

            arrowprops=dict(
                arrowstyle="-",
                lw=0.45,
                color="gray",
                alpha=0.50,
            ),

            expand=(1.20, 1.28),

            force_text=(
                0.70,
                0.95
            ),

            force_static=(
                0.25,
                0.45
            ),

            ensure_inside_axes=True,
        )


    # =====================================================
    # 6-5. Pearson r + p-value 박스
    # =====================================================

    stat_text = (
        f"Pearson r = {r:.3f}\n"
        f"{format_p_value(p)}"
    )

    ax.text(
        0.025,
        0.965,

        stat_text,

        transform=ax.transAxes,

        fontsize=9.2,
        fontweight="bold",

        va="top",
        ha="left",

        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="#bbbbbb",
            alpha=0.92,
        ),

        zorder=10,
    )


    # =====================================================
    # 6-6. 범례
    # =====================================================

    year_handles = [

        Line2D(
            [0],
            [0],

            marker="o",
            linestyle="",

            markerfacecolor=YEAR_COLORS[
                year
            ],

            markeredgecolor="white",

            markersize=7,

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

            markersize=7,

            label="대표대학",
        ),

        Line2D(
            [0],
            [0],

            marker="^",
            linestyle="",

            markerfacecolor="gray",
            markeredgecolor="white",

            markersize=7,

            label="비교대학",
        ),
    ]


    legend_year = ax.legend(
        handles=year_handles,

        title="연도",

        loc="upper right",

        frameon=True,

        fontsize=7.3,
        title_fontsize=7.3,
    )

    ax.add_artist(
        legend_year
    )


    ax.legend(
        handles=group_handles,

        title="구분",

        loc="lower right",

        frameon=True,

        fontsize=7.3,
        title_fontsize=7.3,
    )


    # =====================================================
    # 6-7. 제목 / 축
    # =====================================================

    ax.set_title(
        title,

        fontsize=13,
        fontweight="bold",

        pad=11,
    )

    ax.set_xlabel(
        x_label,
        fontsize=9.5,
    )

    ax.set_ylabel(
        y_label,
        fontsize=9.5,
    )


    # -----------------------------------------------------
    # 3개 그래프 y축 범위를 동일하게 통일
    # -----------------------------------------------------
    # 세 그래프를 나란히 배치했을 때 비교하기 쉬움
    # -----------------------------------------------------

    ax.set_ylim(
        86.0,
        103.0
    )


    # =====================================================
    # 6-8. 격자
    # =====================================================

    ax.grid(
        alpha=0.20,
        linewidth=0.8,
    )

    ax.tick_params(
        axis="both",
        labelsize=8,
    )


    fig.tight_layout()


    # =====================================================
    # 6-9. 저장
    # =====================================================

    save_path = (
        OUTPUT_DIR
        / output_name
    )

    fig.savefig(
        save_path,

        dpi=300,

        bbox_inches="tight",

        facecolor="white",
    )

    plt.close(fig)


    return {
        "r": r,
        "p": p,
        "n": len(plot_df),
        "path": save_path,
        "outliers": outlier_df[
            [
                "기준연도",
                "대학명",
                x_col,
                y_col,
            ]
        ].copy(),
    }


# =========================================================
# 7. 그래프 1
# 취업률 × 충원율
# =========================================================

employment_result = make_scatter(

    data=df,

    x_col="취업률(%)",

    y_col="정원내 신입생 충원율(%)",

    title="취업률과 신입생 충원율",

    x_label="취업률(%)",

    y_label="정원내 신입생 충원율(%)",

    output_name=(
        "rq2_취업률_신입생충원율_발표용.png"
    ),
)


# =========================================================
# 8. 그래프 2
# 경쟁률 × 충원율
# =========================================================

competition_result = make_scatter(

    data=df,

    x_col="경쟁률",

    y_col="정원내 신입생 충원율(%)",

    title="경쟁률과 신입생 충원율",

    x_label="경쟁률",

    y_label="정원내 신입생 충원율(%)",

    output_name=(
        "rq2_경쟁률_신입생충원율_발표용.png"
    ),
)


# =========================================================
# 9. 그래프 3
# 졸업생 진학률 × 충원율
# =========================================================

advancement_result = make_scatter(

    data=df,

    x_col="진학률(%)",

    y_col="정원내 신입생 충원율(%)",

    title=(
        "졸업생 진학률과 "
        "신입생 충원율"
    ),

    x_label=(
        "대학 졸업생 진학률(%)"
    ),

    y_label=(
        "정원내 신입생 충원율(%)"
    ),

    output_name=(
        "rq2_졸업생진학률_신입생충원율_발표용.png"
    ),
)


# =========================================================
# 10. 검증 출력 함수
# =========================================================

def print_result(
    name,
    result
):

    print()
    print(f"[{name}]")

    print(
        f"Pearson r = "
        f"{result['r']:.6f}"
    )

    print(
        f"p-value   = "
        f"{result['p']:.6f}"
    )

    print(
        f"N         = "
        f"{result['n']}"
    )

    print(
        f"저장      = "
        f"{result['path']}"
    )

    print()
    print(
        f"[{name} 그래프 하단 관측치]"
    )

    if result["outliers"].empty:

        print(
            "충원율 98% 미만 관측치 없음"
        )

    else:

        print(
            result["outliers"]
            .sort_values(
                "정원내 신입생 충원율(%)"
            )
            .to_string(
                index=False
            )
        )


# =========================================================
# 11. 결과 출력
# =========================================================

print("=" * 75)
print(
    "PPT RQ2 상관분석 발표용 그래프 생성 완료"
)
print("=" * 75)

print(
    f"전체 관측치: {len(df)}"
)

print(
    f"대학 수: "
    f"{df['대학명'].nunique()}"
)

print(
    "연도:",
    sorted(
        df["기준연도"]
        .dropna()
        .unique()
        .tolist()
    )
)


print_result(
    "취업률 ↔ 충원율",
    employment_result
)

print_result(
    "경쟁률 ↔ 충원율",
    competition_result
)

print_result(
    "졸업생 진학률 ↔ 충원율",
    advancement_result
)


# =========================================================
# 12. 부산외대 검증
# =========================================================

print()
print(
    "[부산외국어대학교 최신값]"
)

bufs = df[
    df["대학명"]
    == "부산외국어대학교"
][
    [
        "기준연도",
        "대학명",
        "경쟁률",
        "정원내 신입생 충원율(%)",
        "취업률(%)",
        "진학률(%)",
        "대표/비교",
    ]
]

print(
    bufs.to_string(
        index=False
    )
)

print("=" * 75)