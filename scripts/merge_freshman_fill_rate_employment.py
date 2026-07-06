import pandas as pd
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

FRESHMAN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "freshman_fill_rate_target_universities.csv"
)

EMPLOYMENT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "employment_rate_target_universities.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "freshman_fill_rate_with_employment.csv"
)


# =========================
# 2. 데이터 불러오기
# =========================

freshman = pd.read_csv(FRESHMAN_PATH, encoding="utf-8-sig")
employment = pd.read_csv(EMPLOYMENT_PATH, encoding="utf-8-sig")

print("신입생 충원율 데이터")
print(freshman.head())
print(freshman.shape)

print("\n취업현황 데이터")
print(employment.head())
print(employment.shape)


# =========================
# 3. 병합 기준 컬럼 정리
# =========================

freshman["기준연도"] = freshman["기준연도"].astype(int)
employment["공시연도"] = employment["공시연도"].astype(int)

freshman["대학명"] = freshman["대학명"].astype(str).str.strip()
employment["대학명"] = employment["대학명"].astype(str).str.strip()


# =========================
# 4. 신입생 충원율 + 취업현황 병합
# =========================
# 신입생 충원율의 기준연도와 취업현황의 공시연도를 연결한다.
# 예:
# 2023년 신입생 충원율 ↔ 2023년 공시 취업현황
# 2024년 신입생 충원율 ↔ 2024년 공시 취업현황
# 2025년 신입생 충원율 ↔ 2025년 공시 취업현황

merged = freshman.merge(
    employment,
    left_on=["기준연도", "대학명"],
    right_on=["공시연도", "대학명"],
    how="left",
    suffixes=("", "_취업"),
)


# =========================
# 5. 병합 결과 확인
# =========================

print("\n병합 결과")
print(merged.head())
print(merged.shape)

employment_check_cols = [
    "자료연도",
    "졸업자",
    "취업자",
    "진학자",
    "취업률(%)",
    "진학률(%)",
]

print("\n취업현황 결측치 확인")
print(merged[employment_check_cols].isna().sum())

if merged[employment_check_cols].isna().sum().sum() > 0:
    print("\n취업현황이 붙지 않은 행")
    print(
        merged[
            merged[employment_check_cols].isna().any(axis=1)
        ][["기준연도", "대학명", "지역"]]
    )


# =========================
# 6. 파생변수 생성
# =========================
# 취업현황은 공시연도와 실제 자료연도가 다르므로 시차를 명시한다.
# 예: 공시연도 2023 → 자료연도 2022

merged["취업성과_자료시차"] = merged["기준연도"] - merged["자료연도"]

merged["졸업자_대비_취업자비율"] = (
    merged["취업자"] / merged["졸업자"].replace(0, pd.NA) * 100
).round(2)

merged["졸업자_대비_진학자비율"] = (
    merged["진학자"] / merged["졸업자"].replace(0, pd.NA) * 100
).round(2)


# =========================
# 7. 최종 컬럼 정리
# =========================

result = merged[
    [
        # 연도 정보
        "기준연도",
        "공시연도",
        "자료연도",
        "취업성과_자료시차",

        # 대학 기본 정보
        "대학명",
        "지역",
        "설립구분",
        "수도권/지방",
        "대표/비교",
        "분석그룹",

        # 신입생 충원현황
        "입학정원",
        "모집인원",
        "지원자",
        "입학자",
        "정원내 신입생 충원율(%)",
        "경쟁률",

        # 졸업생 취업현황
        "졸업자",
        "취업자",
        "진학자",
        "입대자",
        "취업불가능자",
        "외국인유학생",
        "제외인정자",
        "취업률_계산분모",
        "취업률(%)",
        "취업률_원자료_평균",
        "진학률(%)",
        "유지취업률_평균",

        # 파생변수
        "졸업자_대비_취업자비율",
        "졸업자_대비_진학자비율",
    ]
].copy()

result = result.sort_values(["대학명", "기준연도"])


# =========================
# 8. 저장
# =========================

result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")


# =========================
# 9. 확인 출력
# =========================

print("\n완료!")
print(f"저장 위치: {OUTPUT_PATH}")
print(f"전체 행 수: {len(result)}")

print("\n연도별 행 수")
print(result["기준연도"].value_counts().sort_index())

print("\n대학별 행 수")
print(result["대학명"].value_counts().sort_index())

print("\n공시연도 / 자료연도 확인")
print(
    result[["기준연도", "공시연도", "자료연도", "취업성과_자료시차"]]
    .drop_duplicates()
    .sort_values(["기준연도"])
)

print("\n결측치 확인")
print(result.isna().sum())

print("\n미리보기")
print(result.head())