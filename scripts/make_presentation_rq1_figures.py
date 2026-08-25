from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from adjustText import adjust_text


# =========================================================
# 1. 기본 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "freshman_fill_rate_with_population_graduation.csv"
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


# 한글 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ---------------------------------------------------------
# 연도별 색상
# ---------------------------------------------------------

YEAR_COLORS = {
    2023: "#1f77b4",  # 파랑
    2024: "#ff7f0e",  # 주황
    2025: "#2ca02c",  # 초록
}


# ---------------------------------------------------------
# 대표대학 / 비교대학 도형
# ---------------------------------------------------------

GROUP_MARKERS = {
    "대표대학": "o",
    "비교대학": "^",
}


# =========================================================
# 2. 데이터 불러오기
# =========================================================

df = pd.read_csv(
    INPUT_PATH,
    encoding="utf-8-sig"
)


required_columns = [
    "기준연도",
    "대학명",
    "대표/비교",
    "18세_인구",
    "고등학교_진학률",
    "경쟁률",
    "정원내 신입생 충원율(%)",
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
# 3. 데이터형 정리
# =========================================================

df["기준연도"] = pd.to_numeric(
    df["기준연도"],
    errors="coerce"
).astype("Int64")


numeric_columns = [
    "18세_인구",
    "고등학교_진학률",
    "경쟁률",
    "정원내 신입생 충원율(%)",
]


for col in numeric_columns:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =========================================================
# 4. 공통 산점도 함수
# =========================================================

def make_scatter(
    data,
    x_col,
    y_col,
    title,
    x_label,
    y_label,
    output_name,
    label_fontsize=6.4,
    legend_outside=False,
):

    # -----------------------------------------------------
    # 분석 대상 데이터
    # -----------------------------------------------------

    plot_df = data[
        [
            "기준연도",
            "대학명",
            "대표/비교",
            x_col,
            y_col,
        ]
    ].dropna().copy()


    # -----------------------------------------------------
    # Pearson r
    #
    # 그래프에는 절대 표시하지 않음.
    # 데이터 검증용 콘솔 출력에만 사용.
    # -----------------------------------------------------

    r = plot_df[x_col].corr(
        plot_df[y_col],
        method="pearson"
    )


    # =====================================================
    # 그래프 생성
    # =====================================================

    if legend_outside:
        fig, ax = plt.subplots(
            figsize=(9.2, 5.3)
        )
    else:
        fig, ax = plt.subplots(
            figsize=(8.2, 5.3)
        )


    # =====================================================
    # 4-1. 산점도
    #
    # 연도 = 색상
    # 대표/비교 = 도형
    # =====================================================

    for year in sorted(
        plot_df["기준연도"]
        .dropna()
        .unique()
    ):

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

                c=YEAR_COLORS.get(
                    int(year),
                    "gray"
                ),

                marker=GROUP_MARKERS.get(
                    group,
                    "o"
                ),

                alpha=0.80,

                edgecolors="white",
                linewidths=0.7,

                zorder=3,
            )


    # =====================================================
    # 4-2. 대학명 표시
    #
    # 66개 점 모두에 표시하지 않고
    # 2025년 관측치에 대학명 한 번씩 표시
    # =====================================================

    label_df = (
        plot_df[
            plot_df["기준연도"] == 2025
        ]
        .sort_values("대학명")
        .reset_index(drop=True)
    )


    texts = []


    for _, row in label_df.iterrows():

        text = ax.text(
            row[x_col],
            row[y_col],
            row["대학명"],

            fontsize=label_fontsize,

            alpha=0.84,

            zorder=4,
        )

        texts.append(text)


    # =====================================================
    # 4-3. 대학명 겹침 자동 조정
    # =====================================================

    adjust_text(
        texts,

        ax=ax,

        arrowprops=dict(
            arrowstyle="-",
            lw=0.4,
            color="gray",
            alpha=0.45,
        ),

        expand=(1.22, 1.28),

        force_text=(0.65, 0.90),

        force_static=(0.25, 0.45),

        ensure_inside_axes=True,
    )


    # =====================================================
    # 4-4. 범례 생성
    # =====================================================

    year_handles = [

        Line2D(
            [0],
            [0],

            marker="o",
            linestyle="",

            markerfacecolor=YEAR_COLORS[year],
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


    # =====================================================
    # 4-5. 범례 위치
    # =====================================================

    if legend_outside:

        # 오른쪽 그래프:
        # 두 범례 모두 그래프 밖 오른쪽에 배치

        legend_year = ax.legend(
            handles=year_handles,

            title="연도",

            loc="upper left",

            bbox_to_anchor=(
                1.01,
                1.00
            ),

            frameon=True,

            fontsize=8,
            title_fontsize=8,

            borderaxespad=0,
        )


        ax.add_artist(
            legend_year
        )


        ax.legend(
            handles=group_handles,

            title="구분",

            loc="upper left",

            bbox_to_anchor=(
                1.01,
                0.70
            ),

            frameon=True,

            fontsize=8,
            title_fontsize=8,

            borderaxespad=0,
        )


    else:

        # 왼쪽 그래프:
        # 현재처럼 그래프 내부에 배치

        legend_year = ax.legend(
            handles=year_handles,

            title="연도",

            loc="upper right",

            frameon=True,

            fontsize=8,
            title_fontsize=8,
        )


        ax.add_artist(
            legend_year
        )


        ax.legend(
            handles=group_handles,

            title="구분",

            loc="lower right",

            frameon=True,

            fontsize=8,
            title_fontsize=8,
        )


    # =====================================================
    # 4-6. 제목 / 축
    # =====================================================

    ax.set_title(
        title,

        fontsize=14,
        fontweight="bold",

        pad=12,
    )


    ax.set_xlabel(
        x_label,
        fontsize=10,
    )


    ax.set_ylabel(
        y_label,
        fontsize=10,
    )


    # =====================================================
    # 4-7. 충원율 그래프 y축 여백
    # =====================================================

    if y_col == "정원내 신입생 충원율(%)":

        y_min = plot_df[y_col].min()
        y_max = plot_df[y_col].max()

        ax.set_ylim(
            y_min - 0.5,
            y_max + 1.7
        )


    # =====================================================
    # 4-8. 격자 / 눈금
    # =====================================================

    ax.grid(
        alpha=0.20,
        linewidth=0.8,
    )


    ax.tick_params(
        axis="both",
        labelsize=8.5,
    )


    # =====================================================
    # 4-9. 여백 조정
    # =====================================================

    if legend_outside:

        fig.tight_layout(
            rect=[
                0,
                0,
                0.82,
                1,
            ]
        )

    else:

        fig.tight_layout()


    # =====================================================
    # 4-10. 저장
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


    return r, save_path


# =========================================================
# 5. 그래프 1
#
# 지역 18세 인구 × 대학 경쟁률
# =========================================================

r_population, path_population = make_scatter(

    data=df,

    x_col="18세_인구",

    y_col="경쟁률",

    title="지역 18세 인구와 대학 경쟁률",

    x_label="지역 18세 인구(명)",

    y_label="경쟁률",

    output_name=(
        "rq1_18세인구_대학경쟁률_발표용.png"
    ),

    label_fontsize=6.4,

    legend_outside=False,
)


# =========================================================
# 6. 그래프 2
#
# 고교 졸업자의 대학 진학률
# ×
# 정원내 신입생 충원율
# =========================================================

r_graduation, path_graduation = make_scatter(

    data=df,

    x_col="고등학교_진학률",

    y_col="정원내 신입생 충원율(%)",

    title=(
        "고교 졸업자의 대학 진학률과 "
        "신입생 충원율"
    ),

    x_label=(
        "고교 졸업자의 대학 진학률(%)"
    ),

    y_label=(
        "정원내 신입생 충원율(%)"
    ),

    output_name=(
        "rq1_고교진학률_신입생충원율_발표용.png"
    ),

    # 오른쪽은 대학들이 몰려 있으므로
    # 글자 크기를 약간 작게
    label_fontsize=5.8,

    # 범례는 그래프 바깥으로
    legend_outside=True,
)


# =========================================================
# 7. 검증용 출력
# =========================================================

print("=" * 70)

print(
    "PPT RQ1 그래프 생성 완료"
)

print("=" * 70)


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


# ---------------------------------------------------------
# 그래프 1
# ---------------------------------------------------------

print()

print(
    "[그래프 1]"
)

print(
    "지역 18세 인구 ↔ 대학 경쟁률"
)

print(
    f"Pearson r = "
    f"{r_population:.6f}"
)

print(
    "※ r값은 검증용이며 "
    "PPT 그래프에는 표시하지 않음"
)

print(
    f"저장: "
    f"{path_population}"
)


# ---------------------------------------------------------
# 그래프 2
# ---------------------------------------------------------

print()

print(
    "[그래프 2]"
)

print(
    "고교 졸업자의 대학 진학률 "
    "↔ 정원내 신입생 충원율"
)

print(
    f"Pearson r = "
    f"{r_graduation:.6f}"
)

print(
    "※ r값은 검증용이며 "
    "PPT 그래프에는 표시하지 않음"
)

print(
    f"저장: "
    f"{path_graduation}"
)


# ---------------------------------------------------------
# 부산외국어대학교 확인
# ---------------------------------------------------------

print()

print(
    "[부산외국어대학교 확인]"
)


bufs = df[
    df["대학명"]
    == "부산외국어대학교"
][
    [
        "기준연도",
        "대학명",
        "18세_인구",
        "고등학교_진학률",
        "경쟁률",
        "정원내 신입생 충원율(%)",
        "대표/비교",
    ]
]


print(
    bufs.to_string(
        index=False
    )
)


print("=" * 70)