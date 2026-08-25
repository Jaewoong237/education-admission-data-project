import pandas as pd
from pathlib import Path

print("### filter_freshman_fill_rate_v2.py 실행 중 ###")

# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

TARGET_PATH = BASE_DIR / "data" / "interim" / "target_universities.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "academyinfo"

OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "freshman_fill_rate_target_universities.csv"


# =========================
# 2. 대상 대학 목록
# =========================

target = pd.read_csv(TARGET_PATH, encoding="utf-8-sig")
target["대학명"] = target["대학명"].astype(str).str.strip()


# =========================
# 3. 숫자 변환
# =========================

def to_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("-", "0", regex=False)
        .str.strip(),
        errors="coerce",
    ).fillna(0)


# =========================
# 4. 대학알리미 원자료 읽기
# =========================
# 2023~2025 원자료 헤더 구조 직접 확인 완료
#
# 6  : 입학정원(A)
# 7  : 모집인원 계
# 8  : 정원내 모집인원(B)
# 10 : 지원자 계
# 11 : 정원내 지원자(C)
# 13 : 입학자 계
# 14 : 정원내 입학자 남
# 15 : 정원내 입학자 여
# 18 : 공시 정원내 신입생 충원율
# 19 : 공시 경쟁률

def read_academyinfo_excel(file_path: Path) -> pd.DataFrame:
    raw = pd.read_excel(file_path, header=None)

    # 실제 데이터는 6행(index=6)부터
    df = raw.iloc[6:].copy()

    result = pd.DataFrame({
        "기준연도": df.iloc[:, 0],
        "설립구분": df.iloc[:, 2],
        "학교": df.iloc[:, 5],

        "입학정원": to_number(df.iloc[:, 6]),

        # 전체 기준값 - 검증용으로 보존
        "모집인원_전체": to_number(df.iloc[:, 7]),
        "지원자_전체": to_number(df.iloc[:, 10]),
        "입학자_전체": to_number(df.iloc[:, 13]),

        # 본 분석에 사용할 정원내 기준값
        "모집인원": to_number(df.iloc[:, 8]),
        "지원자": to_number(df.iloc[:, 11]),

        "정원내입학자_남": to_number(df.iloc[:, 14]),
        "정원내입학자_여": to_number(df.iloc[:, 15]),

        # 대학알리미 공시값 - 검증용
        "공시_정원내_충원율": to_number(df.iloc[:, 18]),
        "공시_경쟁률": to_number(df.iloc[:, 19]),
    })

    result["입학자"] = (
        result["정원내입학자_남"]
        + result["정원내입학자_여"]
    )

    result["기준연도"] = pd.to_numeric(
        result["기준연도"], errors="coerce"
    )

    result = result.dropna(subset=["기준연도", "학교"]).copy()
    result["기준연도"] = result["기준연도"].astype(int)
    result["학교"] = result["학교"].astype(str).str.strip()
    result["설립구분"] = result["설립구분"].astype(str).str.strip()

    return result


# =========================
# 5. 2023~2025 원자료 합치기
# =========================

excel_files = sorted(
    RAW_DIR.glob("academyinfo_freshman_fill_rate_*.xlsx")
)

if not excel_files:
    raise FileNotFoundError(
        f"원본 엑셀 파일을 찾지 못했습니다: {RAW_DIR}"
    )

data_list = []

for file in excel_files:
    print(f"읽는 중: {file.name}")
    df = read_academyinfo_excel(file)
    data_list.append(df)

raw = pd.concat(data_list, ignore_index=True)


# =========================
# 6. 학교명 표준화
# =========================

name_map = {
    "국립부경대학교": "부경대학교",
    "영산대학교_제2캠퍼스": "영산대학교",
    "영산대학교(양산)_제2캠퍼스": "영산대학교",
    "영산대학교(해운대)": "영산대학교",
}

raw["학교"] = raw["학교"].replace(name_map)


# =========================
# 7. 22개 대학만 필터링
# =========================

filtered = raw[
    raw["학교"].isin(target["대학명"])
].copy()


# =========================
# 8. 캠퍼스 분리 대학 합산
# =========================

numeric_columns = [
    "입학정원",
    "모집인원_전체",
    "지원자_전체",
    "입학자_전체",
    "모집인원",
    "지원자",
    "입학자",
]

grouped = (
    filtered
    .groupby(
        ["기준연도", "학교", "설립구분"],
        as_index=False
    )[numeric_columns]
    .sum()
)


# =========================
# 9. 정원내 지표 재계산
# =========================

grouped["정원내 신입생 충원율(%)"] = (
    grouped["입학자"]
    / grouped["모집인원"].replace(0, pd.NA)
    * 100
).round(1)

grouped["경쟁률"] = (
    grouped["지원자"]
    / grouped["모집인원"].replace(0, pd.NA)
).round(1)


# =========================
# 10. 대상 대학 정보 결합
# =========================

merged = grouped.merge(
    target,
    left_on="학교",
    right_on="대학명",
    how="left"
)


# =========================
# 11. 최종 컬럼
# =========================

result = merged[
    [
        "기준연도",
        "대학명",
        "지역",
        "설립구분",
        "수도권/지방",
        "대표/비교",
        "분석그룹",

        "입학정원",

        # 정원내 분석 변수
        "모집인원",
        "지원자",
        "입학자",
        "정원내 신입생 충원율(%)",
        "경쟁률",

        # 원자료 전체합 - 검증용
        "모집인원_전체",
        "지원자_전체",
        "입학자_전체",
    ]
].copy()

result = result.sort_values(
    ["대학명", "기준연도"]
)

result.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 12. 확인
# =========================

print("\n완료!")
print(f"저장 위치: {OUTPUT_PATH}")
print(f"전체 행 수: {len(result)}")

print("\n부산외국어대학교 확인:")
print(
    result[
        result["대학명"] == "부산외국어대학교"
    ].to_string(index=False)
)

print("\n대학별 행 수:")
print(
    result["대학명"]
    .value_counts()
    .sort_index()
)

print("\n결측치:")
print(result.isna().sum())