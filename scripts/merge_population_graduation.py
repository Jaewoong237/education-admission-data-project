import pandas as pd
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

FRESHMAN_POP_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "freshman_fill_rate_with_population.csv"
)

GRADUATION_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "highschool_graduation_by_region.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "freshman_fill_rate_with_population_graduation.csv"
)


# =========================
# 2. 데이터 불러오기
# =========================

freshman_pop = pd.read_csv(FRESHMAN_POP_PATH, encoding="utf-8-sig")
graduation = pd.read_csv(GRADUATION_PATH, encoding="utf-8-sig")

print("신입생 충원율 + 18세 인구 데이터")
print(freshman_pop.head())
print(freshman_pop.shape)

print("\n고등학교 졸업 후 상황 데이터")
print(graduation.head())
print(graduation.shape)


# =========================
# 3. 병합 기준 컬럼 정리
# =========================

freshman_pop["기준연도"] = freshman_pop["기준연도"].astype(int)
graduation["기준연도"] = graduation["기준연도"].astype(int)

freshman_pop["지역"] = freshman_pop["지역"].astype(str).str.strip()
graduation["지역"] = graduation["지역"].astype(str).str.strip()


# =========================
# 4. 기준연도 + 지역 기준 병합
# =========================

merged = freshman_pop.merge(
    graduation,
    on=["기준연도", "지역"],
    how="left"
)


# =========================
# 5. 병합 결과 확인
# =========================

print("\n병합 결과")
print(merged.head())
print(merged.shape)

missing_cols = [
    "고등학교_졸업자수",
    "고등학교_진학자수",
    "고등학교_진학률",
]

print("\n결측치 확인")
print(merged[missing_cols].isna().sum())

if merged[missing_cols].isna().sum().sum() > 0:
    print("\n고등학교 졸업 데이터가 붙지 않은 행")
    print(
        merged[
            merged[missing_cols].isna().any(axis=1)
        ][["기준연도", "대학명", "지역"]]
    )


# =========================
# 6. 파생변수 생성
# =========================
# 지역 고등학교 졸업자/진학자 규모 대비 대학 모집 규모를 보는 보조 지표

merged["고등학교졸업자수_대비_모집인원비율"] = (
    merged["모집인원"] / merged["고등학교_졸업자수"] * 100
).round(3)

merged["고등학교진학자수_대비_모집인원비율"] = (
    merged["모집인원"] / merged["고등학교_진학자수"] * 100
).round(3)

merged["고등학교진학자수_대비_입학자비율"] = (
    merged["입학자"] / merged["고등학교_진학자수"] * 100
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

print("\n연도별 행 수")
print(merged["기준연도"].value_counts().sort_index())

print("\n대학별 행 수")
print(merged["대학명"].value_counts().sort_index())