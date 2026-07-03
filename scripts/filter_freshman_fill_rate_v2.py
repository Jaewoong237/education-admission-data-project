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
# 2. 대상 대학 목록 불러오기
# =========================

target = pd.read_csv(TARGET_PATH, encoding="utf-8-sig")
target["대학명"] = target["대학명"].astype(str).str.strip()


# =========================
# 3. 대학알리미 엑셀 읽기 함수
# =========================

def read_academyinfo_excel(file_path: Path) -> pd.DataFrame:
    preview = pd.read_excel(file_path, header=None, nrows=30)

    header_row = None

    for idx, row in preview.iterrows():
        row_values = [str(v).strip() for v in row.tolist()]
        if "기준연도" in row_values and "학교" in row_values:
            header_row = idx
            break

    if header_row is None:
        raise ValueError(f"헤더 행을 찾지 못했습니다: {file_path.name}")

    df = pd.read_excel(file_path, header=header_row)
    df.columns = df.columns.astype(str).str.strip()
    df = df.dropna(how="all")

    return df


# =========================
# 4. 원본 파일 합치기
# =========================

excel_files = sorted(RAW_DIR.glob("academyinfo_freshman_fill_rate_*.xlsx"))

if not excel_files:
    raise FileNotFoundError(f"원본 엑셀 파일을 찾지 못했습니다: {RAW_DIR}")

data_list = []

for file in excel_files:
    print(f"읽는 중: {file.name}")
    df = read_academyinfo_excel(file)
    data_list.append(df)

raw = pd.concat(data_list, ignore_index=True)

raw["학교"] = raw["학교"].astype(str).str.strip()


# =========================
# 5. 학교명 표준화
# =========================

name_map = {
    "국립부경대학교": "부경대학교",
    "영산대학교_제2캠퍼스": "영산대학교",
    "영산대학교(양산)_제2캠퍼스": "영산대학교",
    "영산대학교(해운대)": "영산대학교",
}

raw["학교"] = raw["학교"].replace(name_map)


# =========================
# 6. 22개 대학만 필터링
# =========================

filtered = raw[raw["학교"].isin(target["대학명"])].copy()


# =========================
# 7. 컬럼 찾기 함수
# =========================

def normalize_text(text):
    return (
        str(text)
        .strip()
        .replace(" ", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace("\t", "")
    )


def find_column(df, keyword):
    for col in df.columns:
        if keyword in normalize_text(col):
            return col

    print("\n[현재 컬럼 목록]")
    for c in df.columns:
        print(repr(c))

    raise KeyError(f"'{keyword}'에 해당하는 컬럼을 찾지 못했습니다.")


def to_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("-", "0", regex=False),
        errors="coerce"
    ).fillna(0)


# =========================
# 8. 필요한 숫자 컬럼 표준화
# =========================

column_keywords = {
    "입학정원": "입학정원",
    "모집인원": "모집인원",
    "지원자": "지원자",
    "입학자": "입학자",
}

print("\n[컬럼 매칭 확인]")

for new_col, keyword in column_keywords.items():
    source_col = find_column(filtered, keyword)
    print(f"{new_col} ← {source_col}")
    filtered[new_col] = to_number(filtered[source_col])


# =========================
# 9. 캠퍼스 분리 대학 합산
# =========================

numeric_columns = ["입학정원", "모집인원", "지원자", "입학자"]

grouped = (
    filtered
    .groupby(["기준연도", "학교", "설립구분"], as_index=False)[numeric_columns]
    .sum()
)

grouped["정원내 신입생 충원율(%)"] = (
    grouped["입학자"] / grouped["모집인원"].replace(0, pd.NA) * 100
).round(1)

grouped["경쟁률"] = (
    grouped["지원자"] / grouped["모집인원"].replace(0, pd.NA)
).round(2)


# =========================
# 10. 대상 대학 정보 붙이기
# =========================

merged = grouped.merge(
    target,
    left_on="학교",
    right_on="대학명",
    how="left"
)


# =========================
# 11. 최종 컬럼 정리
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
        "모집인원",
        "지원자",
        "입학자",
        "정원내 신입생 충원율(%)",
        "경쟁률",
    ]
].copy()

result["기준연도"] = result["기준연도"].astype(int)
result = result.sort_values(["대학명", "기준연도"])


# =========================
# 12. 저장 및 확인
# =========================

result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n완료!")
print(f"저장 위치: {OUTPUT_PATH}")
print(f"전체 행 수: {len(result)}")

print("\n대학별 행 수:")
print(result["대학명"].value_counts().sort_index())

missing = sorted(set(target["대학명"]) - set(result["대학명"]))
print("\n누락 대학:")
print(missing)