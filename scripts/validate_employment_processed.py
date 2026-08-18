from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
TARGET_PATH = BASE_DIR / "data" / "interim" / "target_universities.csv"
PROCESSED_PATH = BASE_DIR / "data" / "processed" / "employment_rate_target_universities.csv"
REPORT_DIR = BASE_DIR / "reports" / "data_validation"
DETAIL_PATH = REPORT_DIR / "employment_rate_validation_detail.csv"
SUMMARY_PATH = REPORT_DIR / "employment_rate_validation_summary.txt"

EXPECTED_PUBLIC_DATA_YEAR = {
    2023: 2022,
    2024: 2023,
    2025: 2024,
}

RATE_TOLERANCE = 0.01
OFFICIAL_RATE_TOLERANCE = 0.11

REQUIRED_COLUMNS = {
    "공시연도",
    "자료연도",
    "대학명",
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
    "원자료_행수",
}


def close_enough(left, right, tolerance=RATE_TOLERANCE):
    return np.isclose(left, right, atol=tolerance, rtol=0, equal_nan=False)


def main():
    if not PROCESSED_PATH.exists():
        raise SystemExit(
            "가공 CSV가 없습니다. 먼저 다음 명령을 실행하세요:\n"
            "python scripts/process_employment_rate.py"
        )

    target = pd.read_csv(TARGET_PATH, encoding="utf-8-sig")
    data = pd.read_csv(PROCESSED_PATH, encoding="utf-8-sig")

    missing_columns = sorted(REQUIRED_COLUMNS - set(data.columns))
    if missing_columns:
        raise SystemExit(
            "현재 employment_rate_target_universities.csv는 수정 전 구버전입니다.\n"
            f"누락 컬럼: {missing_columns}\n\n"
            "먼저 아래 명령으로 가공 CSV를 새로 생성한 뒤 다시 검증하세요:\n"
            "python scripts/process_employment_rate.py\n"
            "python scripts/validate_employment_processed.py"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    expected_rows = len(target) * len(EXPECTED_PUBLIC_DATA_YEAR)

    # =========================
    # 1. 구조 검증
    # =========================
    structure_checks = []

    structure_checks.append(
        ("전체 행 수 66개", len(data) == expected_rows, f"expected={expected_rows}, actual={len(data)}")
    )

    missing_universities = sorted(set(target["대학명"]) - set(data["대학명"]))
    structure_checks.append(
        ("22개 대학 누락 없음", len(missing_universities) == 0, f"missing={missing_universities}")
    )

    duplicated_keys = data.duplicated(["공시연도", "대학명"], keep=False)
    duplicate_rows = data.loc[duplicated_keys, ["공시연도", "대학명"]]
    structure_checks.append(
        ("대학-공시연도 중복 없음", duplicate_rows.empty, f"duplicates={len(duplicate_rows)}")
    )

    per_university = data["대학명"].value_counts()
    bad_university_counts = per_university[per_university != len(EXPECTED_PUBLIC_DATA_YEAR)]
    structure_checks.append(
        ("대학별 3개년 존재", bad_university_counts.empty, bad_university_counts.to_dict())
    )

    year_pairs = set(
        tuple(x)
        for x in data[["공시연도", "자료연도"]].drop_duplicates().astype(int).to_numpy()
    )
    expected_pairs = set(EXPECTED_PUBLIC_DATA_YEAR.items())
    structure_checks.append(
        (
            "공시연도-자료연도 매핑 정확",
            year_pairs == expected_pairs,
            f"expected={sorted(expected_pairs)}, actual={sorted(year_pairs)}",
        )
    )

    # =========================
    # 2. 계산식 재검증
    # =========================
    expected_denominator = (
        data["졸업자"]
        - data["진학자"]
        - data["입대자"]
        - data["취업불가능자"]
        - data["외국인유학생"]
        - data["제외인정자"]
    )

    expected_employment_rate = (
        data["취업자"] / expected_denominator.replace(0, np.nan) * 100
    ).round(2)

    expected_advancement_rate = (
        data["진학자"] / data["졸업자"].replace(0, np.nan) * 100
    ).round(2)

    data["검증_분모일치"] = close_enough(data["취업률_계산분모"], expected_denominator)
    data["검증_취업률계산일치"] = close_enough(data["취업률(%)"], expected_employment_rate)
    data["검증_진학률계산일치"] = close_enough(data["진학률(%)"], expected_advancement_rate)

    # 단일 원자료 행 대학은 대학알리미 공시 취업률과 직접 비교.
    single_raw_row = data["원자료_행수"].eq(1) & data["취업률_원자료_평균"].notna()
    official_diff = (data["취업률(%)"] - data["취업률_원자료_평균"]).abs()

    data["검증_원자료취업률일치"] = True
    data.loc[single_raw_row, "검증_원자료취업률일치"] = (
        official_diff.loc[single_raw_row] <= OFFICIAL_RATE_TOLERANCE
    )

    # 복수 캠퍼스는 인원 합산 후 재계산하므로 공시 취업률 단순평균과 직접 비교하지 않음.
    data["검증_비고"] = ""
    data.loc[data["원자료_행수"].gt(1), "검증_비고"] = (
        "캠퍼스/원자료 복수행: 인원 합산 후 취업률 재계산, 공시율 단순평균 직접 비교 제외"
    )

    validation_columns = [
        "검증_분모일치",
        "검증_취업률계산일치",
        "검증_진학률계산일치",
        "검증_원자료취업률일치",
    ]

    data["검증_최종"] = np.where(data[validation_columns].all(axis=1), "PASS", "FAIL")

    detail_columns = [
        "공시연도",
        "자료연도",
        "대학명",
        "졸업자",
        "취업자",
        "진학자",
        "취업률_계산분모",
        "취업률(%)",
        "취업률_원자료_평균",
        "취업률_원자료차이(%p)",
        "진학률(%)",
        "원자료_행수",
        *validation_columns,
        "검증_최종",
        "검증_비고",
    ]

    data[detail_columns].to_csv(DETAIL_PATH, index=False, encoding="utf-8-sig")

    # =========================
    # 3. 최종 요약
    # =========================
    structure_pass = all(check[1] for check in structure_checks)
    row_pass_count = int(data["검증_최종"].eq("PASS").sum())
    row_fail_count = int(data["검증_최종"].eq("FAIL").sum())
    overall_pass = structure_pass and row_fail_count == 0 and len(data) == expected_rows

    single_diff = official_diff.loc[single_raw_row]
    max_official_diff = float(single_diff.max()) if not single_diff.empty else float("nan")

    summary_lines = [
        "대학 입학성과 프로젝트 - 취업률·진학률 원자료/가공자료 검증",
        "=" * 70,
        f"전체 판정: {'PASS' if overall_pass else 'FAIL'}",
        f"검증 대상: {len(target)}개 대학 × {len(EXPECTED_PUBLIC_DATA_YEAR)}개 공시연도 = {expected_rows}개 대학-연도",
        f"실제 가공자료 행 수: {len(data)}",
        f"행 단위 PASS: {row_pass_count}",
        f"행 단위 FAIL: {row_fail_count}",
        f"단일 원자료 행 기준 공시 취업률 최대 차이: {max_official_diff:.2f}%p",
        "",
        "[구조 검증]",
    ]

    for name, passed, detail in structure_checks:
        summary_lines.append(f"- [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    summary_lines += [
        "",
        "[계산 검증]",
        f"- 취업률 분모 일치: {int(data['검증_분모일치'].sum())}/{len(data)}",
        f"- 취업률 재계산 일치: {int(data['검증_취업률계산일치'].sum())}/{len(data)}",
        f"- 진학률 재계산 일치: {int(data['검증_진학률계산일치'].sum())}/{len(data)}",
        f"- 원자료 공시 취업률 대조 통과: {int(data['검증_원자료취업률일치'].sum())}/{len(data)}",
        "",
        f"상세 결과: {DETAIL_PATH}",
    ]

    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    print("\n".join(summary_lines))

    if not overall_pass:
        failed = data.loc[data["검증_최종"].eq("FAIL"), detail_columns]
        if not failed.empty:
            print("\n[FAIL 행]")
            print(failed.to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
