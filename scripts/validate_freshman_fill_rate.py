import pandas as pd
from pathlib import Path

# ============================================================
# 1. 경로 설정
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw" / "academyinfo"

TARGET_PATH = (
    BASE_DIR
    / "data"
    / "interim"
    / "target_universities.csv"
)

PROCESSED_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "freshman_fill_rate_target_universities.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "reports"
    / "data_validation"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_DETAIL_PATH = (
    OUTPUT_DIR
    / "freshman_fill_rate_raw_validation_detail.csv"
)

FINAL_DETAIL_PATH = (
    OUTPUT_DIR
    / "freshman_fill_rate_validation_detail.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "freshman_fill_rate_validation_summary.csv"
)


# ============================================================
# 2. 숫자 변환 함수
# ============================================================

def to_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace("-", "0", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip(),
        errors="coerce"
    )


# ============================================================
# 3. 대상 대학 불러오기
# ============================================================

target = pd.read_csv(
    TARGET_PATH,
    encoding="utf-8-sig"
)

target["대학명"] = (
    target["대학명"]
    .astype(str)
    .str.strip()
)


# ============================================================
# 4. 학교명 표준화
# ============================================================

name_map = {
    "국립부경대학교": "부경대학교",

    "영산대학교_제2캠퍼스": "영산대학교",
    "영산대학교(양산)_제2캠퍼스": "영산대학교",
    "영산대학교(해운대)": "영산대학교",
}


# ============================================================
# 5. 원자료 개별 행 검증
# ============================================================
#
# 원자료 구조
#
# 6  입학정원(A)
# 7  모집인원 계
# 8  정원내 모집인원(B)
# 10 지원자 계
# 11 정원내 지원자(C)
# 13 입학자 계
# 14 정원내 입학자 남
# 15 정원내 입학자 여
# 18 공시 정원내 신입생 충원율
# 19 공시 경쟁률
#
# 정원내 입학자(D) = 남 + 여
#
# 충원율 = D / B × 100
# 경쟁률 = C / B
# ============================================================

raw_rows = []

for year in [2023, 2024, 2025]:

    file_path = (
        RAW_DIR
        / f"academyinfo_freshman_fill_rate_{year}.xlsx"
    )

    raw = pd.read_excel(
        file_path,
        header=None
    )

    # 실제 데이터 시작 행
    df = raw.iloc[6:].copy()

    temp = pd.DataFrame({
        "기준연도": pd.to_numeric(
            df.iloc[:, 0],
            errors="coerce"
        ),

        "원자료_학교명": (
            df.iloc[:, 5]
            .astype(str)
            .str.strip()
        ),

        "입학정원": to_number(
            df.iloc[:, 6]
        ),

        "모집인원_전체": to_number(
            df.iloc[:, 7]
        ),

        "정원내_모집인원": to_number(
            df.iloc[:, 8]
        ),

        "지원자_전체": to_number(
            df.iloc[:, 10]
        ),

        "정원내_지원자": to_number(
            df.iloc[:, 11]
        ),

        "입학자_전체": to_number(
            df.iloc[:, 13]
        ),

        "정원내_입학자_남": to_number(
            df.iloc[:, 14]
        ),

        "정원내_입학자_여": to_number(
            df.iloc[:, 15]
        ),

        "공시_충원율": to_number(
            df.iloc[:, 18]
        ),

        "공시_경쟁률": to_number(
            df.iloc[:, 19]
        ),
    })

    temp = temp.dropna(
        subset=[
            "기준연도",
            "원자료_학교명"
        ]
    ).copy()

    temp["기준연도"] = (
        temp["기준연도"]
        .astype(int)
    )

    # 분석용 대학명 생성
    temp["대학명"] = (
        temp["원자료_학교명"]
        .replace(name_map)
    )

    # 22개 대상 대학만 유지
    temp = temp[
        temp["대학명"].isin(
            target["대학명"]
        )
    ].copy()

    # 정원내 입학자 D
    temp["정원내_입학자"] = (
        temp["정원내_입학자_남"]
        + temp["정원내_입학자_여"]
    )

    # 원자료 산식 재계산
    temp["재계산_충원율"] = (
        temp["정원내_입학자"]
        / temp["정원내_모집인원"]
        * 100
    ).round(1)

    temp["재계산_경쟁률"] = (
        temp["정원내_지원자"]
        / temp["정원내_모집인원"]
    ).round(1)

    # 공시값과 비교
    temp["충원율_PASS"] = (
        temp["공시_충원율"].round(1)
        == temp["재계산_충원율"].round(1)
    )

    temp["경쟁률_PASS"] = (
        temp["공시_경쟁률"].round(1)
        == temp["재계산_경쟁률"].round(1)
    )

    temp["원자료_PASS"] = (
        temp["충원율_PASS"]
        & temp["경쟁률_PASS"]
    )

    raw_rows.append(temp)


raw_validation = pd.concat(
    raw_rows,
    ignore_index=True
)

raw_validation = raw_validation.sort_values(
    [
        "대학명",
        "기준연도",
        "원자료_학교명"
    ]
)

raw_validation.to_csv(
    RAW_DETAIL_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 6. 캠퍼스 통합 → 대학-연도 66행 구축
# ============================================================

grouped = (
    raw_validation
    .groupby(
        [
            "기준연도",
            "대학명"
        ],
        as_index=False
    )
    .agg({
        "입학정원": "sum",

        "모집인원_전체": "sum",
        "지원자_전체": "sum",
        "입학자_전체": "sum",

        "정원내_모집인원": "sum",
        "정원내_지원자": "sum",
        "정원내_입학자": "sum",
    })
)


# ============================================================
# 7. 통합 대학의 정원내 지표 재계산
# ============================================================

grouped["재계산_충원율"] = (
    grouped["정원내_입학자"]
    / grouped["정원내_모집인원"]
    * 100
).round(1)

grouped["재계산_경쟁률"] = (
    grouped["정원내_지원자"]
    / grouped["정원내_모집인원"]
).round(1)


# ============================================================
# 8. 현재 processed CSV 불러오기
# ============================================================

processed = pd.read_csv(
    PROCESSED_PATH,
    encoding="utf-8-sig"
)

processed["기준연도"] = (
    processed["기준연도"]
    .astype(int)
)

processed["대학명"] = (
    processed["대학명"]
    .astype(str)
    .str.strip()
)

processed_compare = processed[
    [
        "기준연도",
        "대학명",
        "입학정원",
        "모집인원",
        "지원자",
        "입학자",
        "정원내 신입생 충원율(%)",
        "경쟁률",
        "모집인원_전체",
        "지원자_전체",
        "입학자_전체",
    ]
].copy()

processed_compare = (
    processed_compare.rename(
        columns={
            "입학정원":
                "CSV_입학정원",

            "모집인원":
                "CSV_정원내_모집인원",

            "지원자":
                "CSV_정원내_지원자",

            "입학자":
                "CSV_정원내_입학자",

            "정원내 신입생 충원율(%)":
                "CSV_충원율",

            "경쟁률":
                "CSV_경쟁률",

            "모집인원_전체":
                "CSV_모집인원_전체",

            "지원자_전체":
                "CSV_지원자_전체",

            "입학자_전체":
                "CSV_입학자_전체",
        }
    )
)


# ============================================================
# 9. 원자료 통합값 ↔ processed CSV 비교
# ============================================================

final = grouped.merge(
    processed_compare,
    on=[
        "기준연도",
        "대학명"
    ],
    how="outer",
    indicator=True
)

final["행존재_PASS"] = (
    final["_merge"] == "both"
)

final["입학정원_PASS"] = (
    final["입학정원"]
    == final["CSV_입학정원"]
)

final["정원내_모집인원_PASS"] = (
    final["정원내_모집인원"]
    == final["CSV_정원내_모집인원"]
)

final["정원내_지원자_PASS"] = (
    final["정원내_지원자"]
    == final["CSV_정원내_지원자"]
)

final["정원내_입학자_PASS"] = (
    final["정원내_입학자"]
    == final["CSV_정원내_입학자"]
)

final["충원율_PASS"] = (
    final["재계산_충원율"].round(1)
    == final["CSV_충원율"].round(1)
)

final["경쟁률_PASS"] = (
    final["재계산_경쟁률"].round(1)
    == final["CSV_경쟁률"].round(1)
)

final["전체_모집인원_PASS"] = (
    final["모집인원_전체"]
    == final["CSV_모집인원_전체"]
)

final["전체_지원자_PASS"] = (
    final["지원자_전체"]
    == final["CSV_지원자_전체"]
)

final["전체_입학자_PASS"] = (
    final["입학자_전체"]
    == final["CSV_입학자_전체"]
)

final["최종_PASS"] = (
    final["행존재_PASS"]
    & final["입학정원_PASS"]
    & final["정원내_모집인원_PASS"]
    & final["정원내_지원자_PASS"]
    & final["정원내_입학자_PASS"]
    & final["충원율_PASS"]
    & final["경쟁률_PASS"]
    & final["전체_모집인원_PASS"]
    & final["전체_지원자_PASS"]
    & final["전체_입학자_PASS"]
)

final = final.sort_values(
    [
        "대학명",
        "기준연도"
    ]
)

final.to_csv(
    FINAL_DETAIL_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 검증 요약
# ============================================================

summary = pd.DataFrame({
    "항목": [
        "원자료 대상 행",
        "원자료 충원율 PASS",
        "원자료 경쟁률 PASS",
        "원자료 최종 PASS",
        "원자료 FAIL",

        "최종 대학-연도 행",
        "입학정원 PASS",
        "정원내 모집인원 PASS",
        "정원내 지원자 PASS",
        "정원내 입학자 PASS",
        "충원율 PASS",
        "경쟁률 PASS",
        "전체 모집인원 PASS",
        "전체 지원자 PASS",
        "전체 입학자 PASS",
        "최종 PASS",
        "최종 FAIL",
    ],

    "건수": [
        len(raw_validation),

        int(
            raw_validation[
                "충원율_PASS"
            ].sum()
        ),

        int(
            raw_validation[
                "경쟁률_PASS"
            ].sum()
        ),

        int(
            raw_validation[
                "원자료_PASS"
            ].sum()
        ),

        int(
            (~raw_validation[
                "원자료_PASS"
            ]).sum()
        ),

        len(final),

        int(
            final[
                "입학정원_PASS"
            ].sum()
        ),

        int(
            final[
                "정원내_모집인원_PASS"
            ].sum()
        ),

        int(
            final[
                "정원내_지원자_PASS"
            ].sum()
        ),

        int(
            final[
                "정원내_입학자_PASS"
            ].sum()
        ),

        int(
            final[
                "충원율_PASS"
            ].sum()
        ),

        int(
            final[
                "경쟁률_PASS"
            ].sum()
        ),

        int(
            final[
                "전체_모집인원_PASS"
            ].sum()
        ),

        int(
            final[
                "전체_지원자_PASS"
            ].sum()
        ),

        int(
            final[
                "전체_입학자_PASS"
            ].sum()
        ),

        int(
            final[
                "최종_PASS"
            ].sum()
        ),

        int(
            (~final[
                "최종_PASS"
            ]).sum()
        ),
    ]
})

summary.to_csv(
    SUMMARY_PATH,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 결과 출력
# ============================================================

print(
    "\n=== 1. 원자료 캠퍼스 행 검증 ==="
)

print(
    f"대상 원자료 행 수: "
    f"{len(raw_validation)}"
)

print(
    f"충원율 PASS: "
    f"{raw_validation['충원율_PASS'].sum()}"
)

print(
    f"경쟁률 PASS: "
    f"{raw_validation['경쟁률_PASS'].sum()}"
)

print(
    f"원자료 최종 PASS: "
    f"{raw_validation['원자료_PASS'].sum()}"
)

print(
    f"원자료 FAIL: "
    f"{(~raw_validation['원자료_PASS']).sum()}"
)


raw_fail = raw_validation[
    ~raw_validation["원자료_PASS"]
]

if len(raw_fail) == 0:

    print(
        "\n원자료 개별 행 FAIL 없음"
    )

else:

    print(
        "\n[원자료 FAIL 행]"
    )

    print(
        raw_fail[
            [
                "기준연도",
                "원자료_학교명",
                "대학명",

                "정원내_모집인원",
                "정원내_지원자",
                "정원내_입학자",

                "공시_충원율",
                "재계산_충원율",

                "공시_경쟁률",
                "재계산_경쟁률",
            ]
        ].to_string(
            index=False
        )
    )


print(
    "\n=== 2. 최종 66행 CSV 검증 ==="
)

print(
    summary.to_string(
        index=False
    )
)


final_fail = final[
    ~final["최종_PASS"]
]

if len(final_fail) == 0:

    print(
        "\n최종 CSV FAIL 없음"
    )

else:

    print(
        "\n[최종 CSV FAIL 행]"
    )

    print(
        final_fail[
            [
                "기준연도",
                "대학명",

                "정원내_모집인원",
                "CSV_정원내_모집인원",

                "정원내_지원자",
                "CSV_정원내_지원자",

                "정원내_입학자",
                "CSV_정원내_입학자",

                "재계산_충원율",
                "CSV_충원율",

                "재계산_경쟁률",
                "CSV_경쟁률",

                "최종_PASS",
            ]
        ].to_string(
            index=False
        )
    )


print(
    f"\n원자료 상세:"
    f"\n{RAW_DETAIL_PATH}"
)

print(
    f"\n최종 상세:"
    f"\n{FINAL_DETAIL_PATH}"
)

print(
    f"\n요약:"
    f"\n{SUMMARY_PATH}"
)