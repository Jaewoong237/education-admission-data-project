import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "processed" / "freshman_fill_rate_with_population.csv"

OUTPUT_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 데이터 불러오기
# =========================

df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")

print("데이터 기본 정보")
print(df.info())

print("\n데이터 미리보기")
print(df.head())

# =========================
# 3. 숫자형 컬럼 정리
# =========================

numeric_cols = [
    "18세_인구",
    "입학정원",
    "모집인원",
    "지원자",
    "입학자",
    "정원내 신입생 충원율(%)",
    "경쟁률",
    "18세인구_대비_모집인원비율",
    "18세인구_대비_입학자비율",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 4. 상관계수 계산 함수
# =========================

def calc_corr(data, x_col, y_col):
    temp = data[[x_col, y_col]].dropna()

    if len(temp) < 3:
        return None

    return {
        "변수_X": x_col,
        "변수_Y": y_col,
        "상관계수": round(temp[x_col].corr(temp[y_col]), 4),
        "표본수": len(temp),
    }

# =========================
# 5. 전체 대학 단위 상관분석
# =========================

target_pairs = [
    ("18세_인구", "정원내 신입생 충원율(%)"),
    ("18세_인구", "경쟁률"),
    ("18세_인구", "모집인원"),
    ("18세_인구", "지원자"),
    ("18세_인구", "입학자"),
    ("18세인구_대비_모집인원비율", "정원내 신입생 충원율(%)"),
    ("18세인구_대비_입학자비율", "정원내 신입생 충원율(%)"),
]

overall_corr_list = []

for x_col, y_col in target_pairs:
    result = calc_corr(df, x_col, y_col)
    if result is not None:
        result["분석단위"] = "대학-연도"
        result["그룹"] = "전체"
        overall_corr_list.append(result)

overall_corr = pd.DataFrame(overall_corr_list)

print("\n[전체 대학-연도 단위 상관분석]")
print(overall_corr)

overall_corr.to_csv(
    OUTPUT_DATA_DIR / "population_fill_rate_overall_correlation.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 6. 수도권/지방별 상관분석
# =========================

group_corr_list = []

for group_name, group_df in df.groupby("수도권/지방"):
    for x_col, y_col in target_pairs:
        result = calc_corr(group_df, x_col, y_col)
        if result is not None:
            result["분석단위"] = "대학-연도"
            result["그룹"] = group_name
            group_corr_list.append(result)

region_corr = pd.DataFrame(group_corr_list)

print("\n[수도권/지방별 상관분석]")
print(region_corr)

region_corr.to_csv(
    OUTPUT_DATA_DIR / "population_fill_rate_region_correlation.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 7. 분석그룹별 상관분석
# =========================

analysis_group_corr_list = []

for group_name, group_df in df.groupby("분석그룹"):
    for x_col, y_col in target_pairs:
        result = calc_corr(group_df, x_col, y_col)
        if result is not None:
            result["분석단위"] = "대학-연도"
            result["그룹"] = group_name
            analysis_group_corr_list.append(result)

analysis_group_corr = pd.DataFrame(analysis_group_corr_list)

print("\n[분석그룹별 상관분석]")
print(analysis_group_corr)

analysis_group_corr.to_csv(
    OUTPUT_DATA_DIR / "population_fill_rate_analysis_group_correlation.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 8. 지역-연도 단위 요약
# =========================
# 같은 지역에 여러 대학이 있으므로,
# 지역-연도 단위로 평균 충원율과 총 모집규모를 다시 계산한다.

region_year_summary = (
    df.groupby(["기준연도", "지역", "수도권/지방"], as_index=False)
    .agg(
        인구_18세=("18세_인구", "first"),
        평균_충원율=("정원내 신입생 충원율(%)", "mean"),
        평균_경쟁률=("경쟁률", "mean"),
        총_모집인원=("모집인원", "sum"),
        총_지원자=("지원자", "sum"),
        총_입학자=("입학자", "sum"),
        대학수=("대학명", "nunique"),
    )
)

region_year_summary["평균_충원율"] = region_year_summary["평균_충원율"].round(2)
region_year_summary["평균_경쟁률"] = region_year_summary["평균_경쟁률"].round(2)

region_year_summary["18세인구_대비_총모집인원비율"] = (
    region_year_summary["총_모집인원"] / region_year_summary["인구_18세"] * 100
).round(3)

region_year_summary["18세인구_대비_총입학자비율"] = (
    region_year_summary["총_입학자"] / region_year_summary["인구_18세"] * 100
).round(3)

print("\n[지역-연도 단위 요약]")
print(region_year_summary)

region_year_summary.to_csv(
    OUTPUT_DATA_DIR / "population_fill_rate_region_year_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 9. 지역-연도 단위 상관분석
# =========================

region_year_pairs = [
    ("인구_18세", "평균_충원율"),
    ("인구_18세", "평균_경쟁률"),
    ("인구_18세", "총_모집인원"),
    ("인구_18세", "총_지원자"),
    ("인구_18세", "총_입학자"),
    ("18세인구_대비_총모집인원비율", "평균_충원율"),
    ("18세인구_대비_총입학자비율", "평균_충원율"),
]

region_year_corr_list = []

for x_col, y_col in region_year_pairs:
    result = calc_corr(region_year_summary, x_col, y_col)
    if result is not None:
        result["분석단위"] = "지역-연도"
        result["그룹"] = "전체"
        region_year_corr_list.append(result)

region_year_corr = pd.DataFrame(region_year_corr_list)

print("\n[지역-연도 단위 상관분석]")
print(region_year_corr)

region_year_corr.to_csv(
    OUTPUT_DATA_DIR / "population_fill_rate_region_year_correlation.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 10. 그래프 설정
# =========================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 11. 산점도: 18세 인구 vs 충원율
# =========================

plt.figure(figsize=(8, 5))
plt.scatter(df["18세_인구"], df["정원내 신입생 충원율(%)"])
plt.title("지역 18세 인구와 대학 신입생 충원율")
plt.xlabel("지역 18세 인구")
plt.ylabel("정원내 신입생 충원율(%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "population_age18_vs_fill_rate_scatter.png", dpi=300)
plt.close()

# =========================
# 12. 산점도: 18세 인구 vs 경쟁률
# =========================

plt.figure(figsize=(8, 5))
plt.scatter(df["18세_인구"], df["경쟁률"])
plt.title("지역 18세 인구와 대학 경쟁률")
plt.xlabel("지역 18세 인구")
plt.ylabel("경쟁률")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "population_age18_vs_competition_rate_scatter.png", dpi=300)
plt.close()

# =========================
# 13. 지역-연도 단위 산점도: 18세 인구 vs 평균 충원율
# =========================

plt.figure(figsize=(8, 5))
plt.scatter(region_year_summary["인구_18세"], region_year_summary["평균_충원율"])
plt.title("지역 18세 인구와 지역 평균 신입생 충원율")
plt.xlabel("지역 18세 인구")
plt.ylabel("지역 평균 신입생 충원율(%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "region_year_population_vs_average_fill_rate.png", dpi=300)
plt.close()

# =========================
# 14. 지역-연도 단위 산점도: 18세 인구 vs 평균 경쟁률
# =========================

plt.figure(figsize=(8, 5))
plt.scatter(region_year_summary["인구_18세"], region_year_summary["평균_경쟁률"])
plt.title("지역 18세 인구와 지역 평균 경쟁률")
plt.xlabel("지역 18세 인구")
plt.ylabel("지역 평균 경쟁률")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "region_year_population_vs_average_competition_rate.png", dpi=300)
plt.close()

# =========================
# 15. 로그 스케일 산점도: 지역 18세 인구 vs 평균 충원율
# =========================

plt.figure(figsize=(8, 5))
plt.scatter(region_year_summary["인구_18세"], region_year_summary["평균_충원율"])
plt.xscale("log")
plt.title("지역 18세 인구와 지역 평균 신입생 충원율")
plt.xlabel("지역 18세 인구 로그 스케일")
plt.ylabel("지역 평균 신입생 충원율(%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "region_year_population_log_vs_average_fill_rate.png", dpi=300)
plt.close()

# =========================
# 16. 마무리
# =========================

print("\n관계분석 완료!")
print(f"상관분석 CSV 저장 위치: {OUTPUT_DATA_DIR}")
print(f"그래프 저장 위치: {FIGURE_DIR}")

print("\n주의:")
print("이 분석은 22개 표본 대학 기준의 탐색적 분석입니다.")
print("지역 18세 인구는 해당 지역 대학의 직접적인 지원자 풀 전체를 의미하지 않습니다.")
print("따라서 상관계수는 인과관계가 아니라 관계의 방향을 보는 참고 지표로 해석해야 합니다.")