import pandas as pd
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

FRESHMAN_PATH = BASE_DIR / "data" / "processed" / "freshman_fill_rate_target_universities.csv"
POPULATION_PATH = BASE_DIR / "data" / "processed" / "population_age18_by_region.csv"

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "freshman_fill_rate_with_population.csv"


# =========================
# 2. 데이터 불러오기
# =========================

freshman = pd.read_csv(FRESHMAN_PATH, encoding="utf-8-sig")
population = pd.read_csv(POPULATION_PATH, encoding="utf-8-sig")

print("신입생 충원현황 데이터")
print(freshman.head())
print(freshman.shape)

print("\n18세 인구 데이터")
print(population.head())
print(population.shape)


# =========================
# 3. 병합 기준 컬럼 정리
# =========================

freshman["기준연도"] = freshman["기준연도"].astype(int)
population["기준연도"] = population["기준연도"].astype(int)

freshman["지역"] = freshman["지역"].astype(str).str.strip()
population["지역"] = population["지역"].astype(str).str.strip()


# =========================
# 4. 기준연도 + 지역 기준 병합
# =========================

merged = freshman.merge(
    population,
    on=["기준연도", "지역"],
    how="left"
)


# =========================
# 5. 병합 결과 확인
# =========================

print("\n병합 결과")
print(merged.head())
print(merged.shape)

print("\n18세 인구 결측치 개수")
print(merged["18세_인구"].isna().sum())

if merged["18세_인구"].isna().sum() > 0:
    print("\n18세 인구가 붙지 않은 행")
    print(merged[merged["18세_인구"].isna()][["기준연도", "대학명", "지역"]])


# =========================
# 6. 파생변수 생성
# =========================
# 지역 18세 인구 대비 대학 모집 규모를 보기 위한 보조 지표

merged["18세인구_대비_모집인원비율"] = (
    merged["모집인원"] / merged["18세_인구"] * 100
).round(3)

merged["18세인구_대비_입학자비율"] = (
    merged["입학자"] / merged["18세_인구"] * 100
).round(3)


# =========================
# 7. 저장
# =========================

merged.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n완료!")
print(f"저장 위치: {OUTPUT_PATH}")
print(f"전체 행 수: {len(merged)}")

print("\n최종 미리보기")
print(merged.head())