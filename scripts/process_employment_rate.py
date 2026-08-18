import re
from pathlib import Path

import pandas as pd


# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
TARGET_PATH = BASE_DIR / "data" / "interim" / "target_universities.csv"
RAW_DIR = BASE_DIR / "data" / "raw" / "academyinfo"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "employment_rate_target_universities.csv"


# =========================
# 2. 대상 대학 목록
# =========================

target = pd.read_csv(TARGET_PATH, encoding="utf-8-sig")
target["대학명"] = target["대학명"].astype(str).str.strip()


# =========================
# 3. 보조 함수
# =========================

def normalize_text(value):
    if pd.isna(value):
        return ""
    return (
        str(value)
        .strip()
        .replace(" ", "")
        .replace("\n", "")
        .replace("\r", "")
        .replace("\t", "")
    )


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
    if not match:
        raise ValueError(f"파일명에서 공시연도를 찾지 못했습니다: {file_path.name}")
    return int(match.group(1))


def find_header_row(raw: pd.DataFrame) -> int:
    for idx in range(min(30, len(raw))):
        values = [normalize_text(v) for v in raw.iloc[idx].tolist()]
        has_year = "연도" in values or "기준연도" in values
        has_school = any("학교명" in v for v in values)
        if has_year and has_school:
            return idx
    raise ValueError("대학알리미 취업현황 파일에서 헤더 행을 찾지 못했습니다.")


def build_top_header(raw: pd.DataFrame, header_row: int) -> list[str]:
    """병합셀로 된 최상위 헤더를 오른쪽으로 전파한다."""
    result = []
    current = ""
    for value in raw.iloc[header_row].tolist():
        text = normalize_text(value)
        if text:
            current = text
        result.append(current)
    return result


def find_data_start_row(raw: pd.DataFrame, header_row: int, year_col: int) -> int:
    for idx in range(header_row + 1, min(header_row + 20, len(raw))):
        year_value = pd.to_numeric(raw.iloc[idx, year_col], errors="coerce")
        if pd.notna(year_value):
            return idx
    raise ValueError("대학알리미 취업현황 파일에서 데이터 시작 행을 찾지 못했습니다.")


def find_single_column(top_header: list[str], keyword: str, required=True):
    matches = [i for i, value in enumerate(top_header) if keyword in value]
    if not matches:
        if required:
            raise KeyError(f"'{keyword}'에 해당하는 열을 찾지 못했습니다.")
        return None
    return matches[0]


def find_group_columns(top_header: list[str], keyword: str) -> list[int]:
    matches = [i for i, value in enumerate(top_header) if keyword in value]
    if not matches:
        raise KeyError(f"'{keyword}'에 해당하는 열 묶음을 찾지 못했습니다.")
    return matches


def sum_numeric_columns(df: pd.DataFrame, indices: list[int]) -> pd.Series:
    numeric = pd.concat([to_number(df.iloc[:, i]) for i in indices], axis=1)
    return numeric.sum(axis=1, min_count=1)


def mean_numeric_columns(df: pd.DataFrame, indices: list[int]) -> pd.Series:
    numeric = pd.concat([to_number(df.iloc[:, i]) for i in indices], axis=1)
    return numeric.mean(axis=1)


def read_academyinfo_employment(file_path: Path, public_year: int) -> pd.DataFrame:
    """
    대학알리미 '졸업생의 취업 현황' 원자료의 다단 헤더를 읽는다.

    핵심:
    졸업자/진학자 등은 남+여를 합산하고,
    취업자(B)는 건강보험 직장가입자, 해외취업자, 농림어업 종사자,
    개인창작활동 종사자, 1인 창(사)업자, 프리랜서의 남+여 전부를 합산한다.
    """
    raw = pd.read_excel(file_path, header=None)
    raw = raw.dropna(how="all").reset_index(drop=True)

    header_row = find_header_row(raw)
    top_header = build_top_header(raw, header_row)

    col_year = find_single_column(top_header, "연도")
    col_school = find_single_column(top_header, "학교명")
    col_region = find_single_column(top_header, "지역")
    col_type = find_single_column(top_header, "설립구분")

    data_start_row = find_data_start_row(raw, header_row, col_year)
    data = raw.iloc[data_start_row:].copy().reset_index(drop=True)

    graduate_cols = find_group_columns(top_header, "졸업자(A)")
    employed_cols = find_group_columns(top_header, "취업자(B)")
    advancement_cols = find_group_columns(top_header, "진학자(C)")
    military_cols = find_group_columns(top_header, "입대자(D)")
    unavailable_cols = find_group_columns(top_header, "취업불가능자(E)")
    foreign_cols = find_group_columns(top_header, "외국인유학생(F)")
    excluded_cols = find_group_columns(top_header, "제외인정자(G)")

    employment_rate_col = find_single_column(top_header, "취업률(%)")
    retention_cols = [i for i, value in enumerate(top_header) if "유지취업률" in value]

    # 중요: index를 먼저 만들어야 공시연도 스칼라 값이 전 행에 채워진다.
    # 빈 DataFrame에 공시연도를 먼저 넣으면 뒤에서 Series가 추가될 때
    # 공시연도가 NaN으로 남아 groupby 단계에서 모든 행이 사라질 수 있다.
    result = pd.DataFrame(index=data.index)
    result["공시연도"] = public_year
    result["자료연도"] = to_number(data.iloc[:, col_year]).astype("Int64")
    result["대학명"] = data.iloc[:, col_school].astype(str).str.strip()
    result["지역"] = data.iloc[:, col_region].astype(str).str.strip()
    result["설립구분"] = data.iloc[:, col_type].astype(str).str.strip()

    result["졸업자"] = sum_numeric_columns(data, graduate_cols)
    result["취업자"] = sum_numeric_columns(data, employed_cols)
    result["진학자"] = sum_numeric_columns(data, advancement_cols)
    result["입대자"] = sum_numeric_columns(data, military_cols)
    result["취업불가능자"] = sum_numeric_columns(data, unavailable_cols)
    result["외국인유학생"] = sum_numeric_columns(data, foreign_cols)
    result["제외인정자"] = sum_numeric_columns(data, excluded_cols)
    result["취업률(%)_원자료"] = to_number(data.iloc[:, employment_rate_col])

    if retention_cols:
        result["유지취업률(%)"] = mean_numeric_columns(data, retention_cols)
    else:
        result["유지취업률(%)"] = pd.NA

    result = result[
        result["자료연도"].notna()
        & result["대학명"].notna()
        & ~result["대학명"].isin(["", "nan", "None"])
    ].copy()

    return result


# =========================
# 4. 원자료 읽기
# =========================

excel_files = sorted(RAW_DIR.glob("academyinfo_employment_rate_*.xlsx"))
if not excel_files:
    raise FileNotFoundError(f"취업현황 원본 엑셀 파일을 찾지 못했습니다: {RAW_DIR}")

data_list = []
for file in excel_files:
    public_year = get_public_year_from_filename(file)
    print(f"읽는 중: {file.name} / 공시연도: {public_year}")
    part = read_academyinfo_employment(file, public_year)
    print(f"  원자료 유효 행: {len(part)} / 대학 예시: {part['대학명'].head(3).tolist()}")
    data_list.append(part)

raw = pd.concat(data_list, ignore_index=True)


# =========================
# 5. 학교명 표준화
# =========================

name_map = {
    "국립부경대학교": "부경대학교",
    "영산대학교_제2캠퍼스": "영산대학교",
    "영산대학교(양산)_제2캠퍼스": "영산대학교",
    "영산대학교(해운대)": "영산대학교",
}
raw["대학명"] = raw["대학명"].replace(name_map)


# =========================
# 6. 22개 대학 필터링
# =========================

filtered = raw[raw["대학명"].isin(target["대학명"])].copy()

print(f"\n22개 대학 필터 전 행 수: {len(raw)}")
print(f"22개 대학 필터 후 행 수: {len(filtered)}")

if filtered.empty:
    sample_raw_names = raw["대학명"].dropna().astype(str).drop_duplicates().head(30).tolist()
    raise ValueError(
        "22개 대상 대학이 한 건도 매칭되지 않았습니다. 학교명 파싱을 확인하세요.\n"
        f"원자료 대학명 예시: {sample_raw_names}\n"
        f"대상 대학명: {target['대학명'].tolist()}"
    )


# =========================
# 7. 캠퍼스 분리 대학 합산
# =========================

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
        원자료_행수=("대학명", "size"),
    )
)

# 대학알리미 취업률 공식
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

grouped["취업률_원자료_평균"] = grouped["취업률_원자료_평균"].round(2)
grouped["유지취업률_평균"] = grouped["유지취업률_평균"].round(2)
grouped["취업률_원자료차이(%p)"] = (
    grouped["취업률(%)"] - grouped["취업률_원자료_평균"]
).abs().round(2)

# 단일 원자료 행 대학은 공시 취업률과 재계산값이 반올림 오차 범위 내에서 일치해야 한다.
single_row_check = grouped["원자료_행수"].eq(1) & grouped["취업률_원자료_평균"].notna()
qa_fail = grouped[single_row_check & grouped["취업률_원자료차이(%p)"].gt(0.11)]
if not qa_fail.empty:
    raise ValueError(
        "취업률 원자료 대조에 실패했습니다. 다단 헤더 집계 구조를 확인하세요.\n"
        + qa_fail[
            [
                "공시연도",
                "자료연도",
                "대학명",
                "취업률(%)",
                "취업률_원자료_평균",
                "취업률_원자료차이(%p)",
            ]
        ].to_string(index=False)
    )


# =========================
# 8. 대상 대학 정보 결합
# =========================

merged = grouped.merge(target, on="대학명", how="left")

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
        "취업률_원자료차이(%p)",
        "진학률(%)",
        "유지취업률_평균",
        "원자료_행수",
    ]
].copy()

result = result.sort_values(["대학명", "공시연도"])


# =========================
# 9. 구조 검증
# =========================

expected_rows = len(target) * len(excel_files)
if len(result) != expected_rows:
    missing = sorted(set(target["대학명"]) - set(result["대학명"]))
    counts = result["대학명"].value_counts().sort_index().to_dict()
    raise ValueError(
        f"예상 행 수와 다릅니다. 예상={expected_rows}, 실제={len(result)}\n"
        f"누락 대학={missing}\n"
        f"대학별 행 수={counts}"
    )

missing = sorted(set(target["대학명"]) - set(result["대학명"]))
if missing:
    raise ValueError(f"누락 대학이 있습니다: {missing}")

per_university = result["대학명"].value_counts()
if not per_university.eq(len(excel_files)).all():
    raise ValueError(
        "대학별 연도 수가 일치하지 않습니다.\n"
        + per_university.sort_index().to_string()
    )


# =========================
# 10. 저장 및 확인
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
print(
    result[["공시연도", "자료연도"]]
    .drop_duplicates()
    .sort_values(["공시연도", "자료연도"])
)

single_diff = result.loc[result["원자료_행수"].eq(1), "취업률_원자료차이(%p)"]
print("\n단일 원자료 행 기준 취업률 원자료 대조 최대 차이(%p)")
print(single_diff.max())

print("\n미리보기")
print(result.head())
