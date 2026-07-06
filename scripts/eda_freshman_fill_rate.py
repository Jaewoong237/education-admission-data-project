import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "freshman_fill_rate_target_universities.csv"

OUTPUT_DATA_DIR = BASE_DIR / "data" / "processed"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 2. 데이터 불러오기
# =========================

df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

print("데이터 기본 정보")
print(df.info())

print("\n데이터 미리보기")
print(df.head())

print("\n행/열 개수")
print(df.shape)

# =========================
# 3. 숫자형 컬럼 정리
# =========================

numeric_cols = [
    "입학정원",
    "모집인원",
    "지원자",
    "입학자",
    "정원내 신입생 충원율(%)",
    "경쟁률",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# =========================
# 4. 기본 요약 통계
# =========================

print("\n신입생 충원율 요약 통계")
print(df["정원내 신입생 충원율(%)"].describe())

# =========================
# 5. 연도별 평균 충원율
# =========================

year_summary = (
    df.groupby("기준연도", as_index=False)
    .agg(
        평균_충원율=("정원내 신입생 충원율(%)", "mean"),
        평균_경쟁률=("경쟁률", "mean"),
        총_모집인원=("모집인원", "sum"),
        총_지원자=("지원자", "sum"),
        총_입학자=("입학자", "sum"),
    )
)

year_summary["평균_충원율"] = year_summary["평균_충원율"].round(2)
year_summary["평균_경쟁률"] = year_summary["평균_경쟁률"].round(2)

print("\n연도별 요약")
print(year_summary)

year_summary.to_csv(
    OUTPUT_DATA_DIR / "freshman_fill_rate_year_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 6. 수도권/지방별 평균 충원율
# =========================

region_summary = (
    df.groupby(["기준연도", "수도권/지방"], as_index=False)
    .agg(
        평균_충원율=("정원내 신입생 충원율(%)", "mean"),
        평균_경쟁률=("경쟁률", "mean"),
        대학수=("대학명", "nunique"),
    )
)

region_summary["평균_충원율"] = region_summary["평균_충원율"].round(2)
region_summary["평균_경쟁률"] = region_summary["평균_경쟁률"].round(2)

print("\n수도권/지방별 요약")
print(region_summary)

region_summary.to_csv(
    OUTPUT_DATA_DIR / "freshman_fill_rate_region_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 7. 설립유형별 평균 충원율
# =========================

type_summary = (
    df.groupby(["기준연도", "설립구분"], as_index=False)
    .agg(
        평균_충원율=("정원내 신입생 충원율(%)", "mean"),
        평균_경쟁률=("경쟁률", "mean"),
        대학수=("대학명", "nunique"),
    )
)

type_summary["평균_충원율"] = type_summary["평균_충원율"].round(2)
type_summary["평균_경쟁률"] = type_summary["평균_경쟁률"].round(2)

print("\n설립유형별 요약")
print(type_summary)

type_summary.to_csv(
    OUTPUT_DATA_DIR / "freshman_fill_rate_type_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 8. 분석그룹별 평균 충원율
# =========================

group_summary = (
    df.groupby(["기준연도", "분석그룹"], as_index=False)
    .agg(
        평균_충원율=("정원내 신입생 충원율(%)", "mean"),
        평균_경쟁률=("경쟁률", "mean"),
        대학수=("대학명", "nunique"),
    )
)

group_summary["평균_충원율"] = group_summary["평균_충원율"].round(2)
group_summary["평균_경쟁률"] = group_summary["평균_경쟁률"].round(2)

print("\n분석그룹별 요약")
print(group_summary)

group_summary.to_csv(
    OUTPUT_DATA_DIR / "freshman_fill_rate_group_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 9. 대학별 3년 평균 충원율
# =========================

university_summary = (
    df.groupby(["대학명", "수도권/지방", "설립구분", "대표/비교", "분석그룹"], as_index=False)
    .agg(
        평균_충원율=("정원내 신입생 충원율(%)", "mean"),
        최소_충원율=("정원내 신입생 충원율(%)", "min"),
        최대_충원율=("정원내 신입생 충원율(%)", "max"),
        평균_경쟁률=("경쟁률", "mean"),
    )
)

university_summary["평균_충원율"] = university_summary["평균_충원율"].round(2)
university_summary["최소_충원율"] = university_summary["최소_충원율"].round(2)
university_summary["최대_충원율"] = university_summary["최대_충원율"].round(2)
university_summary["평균_경쟁률"] = university_summary["평균_경쟁률"].round(2)

university_summary = university_summary.sort_values("평균_충원율", ascending=False)

print("\n대학별 3년 평균 충원율")
print(university_summary)

university_summary.to_csv(
    OUTPUT_DATA_DIR / "freshman_fill_rate_university_summary.csv",
    index=False,
    encoding="utf-8-sig",
)

# =========================
# 10. 그래프 설정
# =========================

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# =========================
# 11. 연도별 평균 충원율 그래프
# =========================

plt.figure(figsize=(8, 5))
plt.plot(year_summary["기준연도"], year_summary["평균_충원율"], marker="o")
plt.title("연도별 평균 신입생 충원율")
plt.xlabel("기준연도")
plt.ylabel("평균 신입생 충원율(%)")
plt.xticks(year_summary["기준연도"])
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "yearly_average_fill_rate.png", dpi=300)
plt.close()

# =========================
# 12. 수도권/지방별 평균 충원율 그래프
# =========================

pivot_region = region_summary.pivot(
    index="기준연도",
    columns="수도권/지방",
    values="평균_충원율",
)

plt.figure(figsize=(8, 5))
for col in pivot_region.columns:
    plt.plot(pivot_region.index, pivot_region[col], marker="o", label=col)

plt.title("수도권/지방별 평균 신입생 충원율")
plt.xlabel("기준연도")
plt.ylabel("평균 신입생 충원율(%)")
plt.xticks(pivot_region.index)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "region_average_fill_rate.png", dpi=300)
plt.close()

# =========================
# 13. 설립유형별 평균 충원율 그래프
# =========================

pivot_type = type_summary.pivot(
    index="기준연도",
    columns="설립구분",
    values="평균_충원율",
)

plt.figure(figsize=(8, 5))
for col in pivot_type.columns:
    plt.plot(pivot_type.index, pivot_type[col], marker="o", label=col)

plt.title("설립유형별 평균 신입생 충원율")
plt.xlabel("기준연도")
plt.ylabel("평균 신입생 충원율(%)")
plt.xticks(pivot_type.index)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "type_average_fill_rate.png", dpi=300)
plt.close()

# =========================
# 14. 분석그룹별 평균 충원율 그래프
# =========================

pivot_group = group_summary.pivot(
    index="기준연도",
    columns="분석그룹",
    values="평균_충원율",
)

plt.figure(figsize=(12, 6))
for col in pivot_group.columns:
    plt.plot(pivot_group.index, pivot_group[col], marker="o", label=col)

plt.title("분석그룹별 평균 신입생 충원율")
plt.xlabel("기준연도")
plt.ylabel("평균 신입생 충원율(%)")
plt.xticks(pivot_group.index)
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "group_average_fill_rate.png", dpi=300)
plt.close()

# =========================
# 15. 대학별 평균 충원율 막대그래프
# =========================

plt.figure(figsize=(12, 8))
plt.barh(university_summary["대학명"], university_summary["평균_충원율"])
plt.title("대학별 3년 평균 신입생 충원율")
plt.xlabel("평균 신입생 충원율(%)")
plt.ylabel("대학명")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(FIGURE_DIR / "university_average_fill_rate.png", dpi=300)
plt.close()

print("\nEDA 완료!")
print(f"요약 CSV 저장 위치: {OUTPUT_DATA_DIR}")
print(f"그래프 저장 위치: {FIGURE_DIR}")