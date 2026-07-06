import pandas as pd
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "kess"
    / "kess_highschool_graduation_after_status_1999_2025.xlsx"
)

OUTPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "highschool_graduation_by_region.csv"
)


# =========================
# 2. 원본 엑셀 읽기
# =========================
# 원본 엑셀은 상단에 병합 헤더가 있으므로 header=None으로 읽는다.
# 실제 데이터는 A열 연도, B열 시도, C열 학제, D열 졸업자 전체, G열 진학자 전체를 사용한다.

raw = pd.read_excel(INPUT_PATH, sheet_name=0, header=None)

print("원본 데이터 미리보기")
print(raw.head(10))


# =========================
# 3. 필요한 열만 추출
# =========================
# A열: 연도           -> index 0
# B열: 시도           -> index 1
# C열: 학제           -> index 2
# D열: 졸업자 전체     -> index 3
# G열: 진학자 전체     -> index 6

df = raw[[0, 1, 2, 3, 6]].copy()

df.columns = [
    "기준연도",
    "지역",
    "학제",
    "고등학교_졸업자수",
    "고등학교_진학자수",
]


# =========================
# 4. 실제 데이터 행만 남기기
# =========================
# 헤더 행에는 기준연도가 숫자가 아니므로 제거된다.

df["기준연도"] = pd.to_numeric(df["기준연도"], errors="coerce")
df = df.dropna(subset=["기준연도"]).copy()
df["기준연도"] = df["기준연도"].astype(int)

df["지역"] = df["지역"].astype(str).str.strip()
df["학제"] = df["학제"].astype(str).str.strip()


# =========================
# 5. 고등학교 전체 + 2023~2025년만 필터링
# =========================
# 일반계고, 전문계고 등 세부 유형은 나중에 별도 분석용으로 보류하고,
# 지금은 전체 고등학교 기준만 사용한다.

df = df[
    (df["학제"] == "고등학교")
    & (df["기준연도"].isin([2023, 2024, 2025]))
].copy()


# =========================
# 6. 지역명 표준화
# =========================

region_map = {
    "전국": "전국",
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "경기도": "경기",
    "강원도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라북도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
}

df["지역"] = df["지역"].replace(region_map)


# =========================
# 7. 숫자형 변환 함수
# =========================

def to_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("-", "0", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


df["고등학교_졸업자수"] = to_number(df["고등학교_졸업자수"])
df["고등학교_진학자수"] = to_number(df["고등학교_진학자수"])


# =========================
# 8. 진학률 계산
# =========================

df["고등학교_진학률"] = (
    df["고등학교_진학자수"] / df["고등학교_졸업자수"] * 100
).round(2)


# =========================
# 9. 최종 컬럼 정리
# =========================

result = df[
    [
        "기준연도",
        "지역",
        "고등학교_졸업자수",
        "고등학교_진학자수",
        "고등학교_진학률",
    ]
].copy()

result = result.sort_values(["지역", "기준연도"])


# =========================
# 10. 저장
# =========================

result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")


# =========================
# 11. 확인 출력
# =========================

print("\n완료!")
print(f"저장 위치: {OUTPUT_PATH}")
print(f"전체 행 수: {len(result)}")

print("\n미리보기")
print(result.head(30))

print("\n지역 목록")
print(sorted(result["지역"].unique()))

print("\n연도별 행 수")
print(result["기준연도"].value_counts().sort_index())

print("\n결측치 확인")
print(result.isna().sum())