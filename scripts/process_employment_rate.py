import re
import pandas as pd
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

TARGET_PATH = BASE_DIR / "data" / "interim" / "target_universities.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "academyinfo"

OUTPUT_PATH = BASE_DIR / "data" / "processed" / "employment_rate_target_universities.csv"


# =========================
# 2. 대상 대학 목록 불러오기
# =========================

target = pd.read_csv(TARGET_PATH, encoding="utf-8-sig")
target["대학명"] = target["대학명"].astype(str).str.strip()


# =========================
# 3. 보조 함수
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


def find_column_optional(df, keyword):
    for col in df.columns:
        if keyword in normalize_text(col):
            return col
    return None


def to_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("-", "0", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def get_public_year_from_filename(file_path: Path) -> int:
    match = re.search(r"(\d{4})", file_path.name)
    if match:
        return int(match.group(1))
    raise ValueError(f"파일명에서 공시연도를 찾지 못했습니다: {file_path.name}")


def read_academyinfo_excel(file_path: Path) -> pd.DataFrame:
    preview = pd.read_excel(file_path, header=None, nrows=30)

    header_row = None

    for idx, row in preview.iterrows():
        row_values = [normalize_text(v) for v in row.tolist()]
        has_year = "연도" in row_values or "기준연도" in row_values
        has_school = any("학교" in v for v in row_values)

        if has_year and has_school:
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

excel_files = sorted(RAW_DIR.glob("academyinfo_employment_rate_*.xlsx"))

if not excel_files:
    raise FileNotFoundError(f"취업현황 원본 엑셀 파일을 찾지 못했습니다: {RAW_DIR}")

data_list = []

for file in excel_files:
    public_year = get_public_year_from_filename(file)
    print(f"읽는 중: {file.name} / 공시연도: {public_year}")

    df = read_academyinfo_excel(file)
    df["공시연도"] = public_year

    data_list.append(df)

raw = pd.concat(data_list, ignore_index=True)


# =========================
# 5. 필요한 컬럼 표준화
# =========================

col_year = find_column(raw, "연도")
col_school = find_column(raw, "학교명")
col_region = find_column(raw, "지역")
col_type = find_column(raw, "설립구분")

col_graduate = find_column(raw, "졸업자")
col_employed = find_column(raw, "취업자")
col_advancement = find_column(raw, "진학자")
col_military = find_column(raw, "입대자")
col_unavailable = find_column(raw, "취업불가능자")
col_foreign = find_column(raw, "외국인유학생")
col_excluded = find_column(raw, "제외인정자")
col_employment_rate = find_column(raw, "취업률")

col_retention_rate = find_column_optional(raw, "유지취업률")

employment = pd.DataFrame()

employment["공시연도"] = raw["공시연도"]
employment["자료연도"] = to_number(raw[col_year]).astype("Int64")
employment["대학명"] = raw[col_school].astype(str).str.strip()
employment["지역"] = raw[col_region].astype(str).str.strip()
employment["설립구분"] = raw[col_type].astype(str).str.strip()

employment["졸업자"] = to_number(raw[col_graduate])
employment["취업자"] = to_number(raw[col_employed])
employment["진학자"] = to_number(raw[col_advancement])
employment["입대자"] = to_number(raw[col_military])
employment["취업불가능자"] = to_number(raw[col_unavailable])
employment["외국인유학생"] = to_number(raw[col_foreign])
employment["제외인정자"] = to_number(raw[col_excluded])
employment["취업률(%)_원자료"] = to_number(raw[col_employment_rate])

if col_retention_rate is not None:
    employment["유지취업률(%)"] = to_number(raw[col_retention_rate])
else:
    employment["유지취업률(%)"] = pd.NA


# =========================
# 6. 학교명 표준화
# =========================

name_map = {
    "국립부경대학교": "부경대학교",
    "영산대학교_제2캠퍼스": "영산대학교",
    "영산대학교(양산)_제2캠퍼스": "영산대학교",
    "영산대학교(해운대)": "영산대학교",
}

employment["대학명"] = employment["대학명"].replace(name_map)


# =========================
# 7. 22개 대학만 필터링
# =========================

filtered = employment[employment["대학명"].isin(target["대학명"])].copy()


# =========================
# 8. 캠퍼스 분리 대학 합산
# =========================

sum_cols = [
    "졸업자",
    "취업자",
    "진학자",
    "입대자",
    "취업불가능자",
    "외국인유학생",
    "제외인정자",
]

grouped = (
    filtered
    .groupby(["공시연도", "자료연도", "대학명", "설립구분"], as_index=False)
    .agg(
        졸업자=("졸업자", "sum"),
        취업자=("취업자", "sum"),
        진학자=("진학자", "sum"),
        입대자=("입대자", "sum"),
        취업불가능자=("취업불가능자", "sum"),
        외국인유학생=("외국인유학생", "sum"),
        제외인정자=("제외인정자", "sum"),
        취업률_원자료_평균=("취업률(%)_원자료", "mean"),
        유지취업률_평균=("유지취업률(%)", "mean"),
    )
)

# 취업률 계산 기준:
# 취업률 = 취업자 / (졸업자 - 진학자 - 입대자 - 취업불가능자 - 외국인유학생 - 제외인정자) * 100

grouped["취업률_계산분모"] = (
    grouped["졸업자"]
    - grouped["진학자"]
    - grouped["입대자"]
    - grouped["취업불가능자"]
    - grouped["외국인유학생"]
    - grouped["제외인정자"]
)

grouped["취업률(%)"] = (
    grouped["취업자"] / grouped["취업률_계산분모"].replace(0, pd.NA) * 100
).round(2)

grouped["진학률(%)"] = (
    grouped["진학자"] / grouped["졸업자"].replace(0, pd.NA) * 100
).round(2)

grouped["유지취업률_평균"] = grouped["유지취업률_평균"].round(2)
grouped["취업률_원자료_평균"] = grouped["취업률_원자료_평균"].round(2)


# =========================
# 9. 대상 대학 정보 붙이기
# =========================

merged = grouped.merge(
    target,
    on="대학명",
    how="left"
)


# =========================
# 10. 최종 컬럼 정리
# =========================

result = merged[
    [
        "공시연도",
        "자료연도",
        "대학명",
        "지역",
        "설립구분",
        "수도권/지방",
        "대표/비교",
        "분석그룹",
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
    ]
].copy()

result = result.sort_values(["대학명", "공시연도"])


# =========================
# 11. 저장 및 확인
# =========================

result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n완료!")
print(f"저장 위치: {OUTPUT_PATH}")
print(f"전체 행 수: {len(result)}")

print("\n대학별 행 수")
print(result["대학명"].value_counts().sort_index())

print("\n공시연도별 행 수")
print(result["공시연도"].value_counts().sort_index())

print("\n자료연도 확인")
print(result[["공시연도", "자료연도"]].drop_duplicates().sort_values(["공시연도", "자료연도"]))

missing = sorted(set(target["대학명"]) - set(result["대학명"]))
print("\n누락 대학")
print(missing)

print("\n미리보기")
print(result.head())