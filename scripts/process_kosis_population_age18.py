import pandas as pd
from pathlib import Path

# =========================
# 1. 경로 설정
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "raw" / "kosis" / "kosis_population_age18_2023_2025.xlsx"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "population_age18_by_region.csv"

# =========================
# 2. KOSIS 원본 읽기
# =========================

df = pd.read_excel(INPUT_PATH, sheet_name="데이터")

print("원본 데이터 미리보기")
print(df.head())

# =========================
# 3. 필요한 컬럼만 정리
# =========================

# KOSIS 파일 구조:
# 시도별(1) = 지역
# 성별(1) = 성별
# 연령별(2) = 18세
# 2023, 2024, 2025 = 연도별 인구

year_cols = ["2023", "2024", "2025"]

keep_cols = ["시도별(1)", "성별(1)", "연령별(2)"] + year_cols

df = df[keep_cols].copy()

df = df.rename(columns={
    "시도별(1)": "지역",
    "성별(1)": "성별",
    "연령별(2)": "연령",
})

# =========================
# 4. 지역명 표준화
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
# 5. wide 형식 → long 형식 변환
# =========================

population_long = df.melt(
    id_vars=["지역", "성별", "연령"],
    value_vars=year_cols,
    var_name="기준연도",
    value_name="18세_인구",
)

population_long["기준연도"] = population_long["기준연도"].astype(int)
population_long["18세_인구"] = pd.to_numeric(population_long["18세_인구"], errors="coerce").astype(int)

# 필요한 컬럼만 남김
population_long = population_long[
    ["기준연도", "지역", "18세_인구"]
].copy()

population_long = population_long.sort_values(["지역", "기준연도"])

# =========================
# 6. 저장
# =========================

population_long.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print("\n완료!")
print(f"저장 위치: {OUTPUT_PATH}")
print(f"전체 행 수: {len(population_long)}")

print("\n미리보기")
print(population_long.head(20))