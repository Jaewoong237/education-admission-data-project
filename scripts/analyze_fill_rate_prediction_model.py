# scripts/analyze_fill_rate_prediction_model.py

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# =========================
# 1. 기본 설정
# =========================

warnings.filterwarnings("ignore")

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parents[1]

INPUT_PATH = BASE_DIR / "data" / "processed" / "freshman_fill_rate_with_employment.csv"

OUTPUT_DIR = BASE_DIR / "reports" / "fill_rate_prediction_model"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# 2. 컬럼명 후보
# =========================

COLUMN_ALIASES = {
    "year": [
        "year", "연도", "기준연도"
    ],
    "university": [
        "university", "university_name", "대학명", "학교명"
    ],
    "region": [
        "region", "지역"
    ],
    "foundation_type": [
        "foundation_type", "설립구분"
    ],
    "metro_local": [
        "metro_local", "수도권/지방"
    ],
    "representative_group": [
        "representative_group", "대표/비교"
    ],
    "analysis_group": [
        "analysis_group", "분석그룹"
    ],
    "fill_rate": [
        "fill_rate",
        "freshman_fill_rate",
        "신입생충원율",
        "신입생 충원율",
        "충원율",
        "충원율(%)",
        "신입생 충원율(%)",
        "정원내 신입생 충원율(%)"
    ],
    "employment_rate": [
        "employment_rate",
        "취업률",
        "취업률(%)",
        "졸업생취업률",
        "졸업생 취업률",
        "취업률_원자료_평균"
    ],
    "advancement_rate": [
        "advancement_rate",
        "high_school_advancement_rate",
        "진학률",
        "진학률(%)",
        "고등학교진학률",
        "고등학교 진학률",
        "졸업자_대비_진학자비율"
    ],
    "competition_rate": [
        "competition_rate",
        "admission_competition_rate",
        "경쟁률",
        "입학경쟁률",
        "입학 경쟁률"
    ],
    "admission_quota": [
        "admission_quota", "입학정원"
    ],
    "recruitment_quota": [
        "recruitment_quota", "모집인원"
    ],
    "applicants": [
        "applicants", "지원자"
    ],
    "entrants": [
        "entrants", "입학자"
    ],
}


# =========================
# 3. 유틸 함수
# =========================

def read_csv_safely(path: Path) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "cp949"]

    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일을 읽을 수 없습니다. 인코딩을 확인하세요.")


def find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized_columns = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()
        if key in normalized_columns:
            return normalized_columns[key]

    return None


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}

    for standard_name, aliases in COLUMN_ALIASES.items():
        found_col = find_column(df, aliases)
        if found_col is not None:
            rename_map[found_col] = standard_name

    df = df.rename(columns=rename_map)

    required_columns = [
        "year",
        "university",
        "fill_rate",
        "employment_rate",
        "advancement_rate",
        "competition_rate",
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"필수 컬럼이 없습니다: {missing_columns}\n"
            f"현재 컬럼 목록: {list(df.columns)}"
        )

    return df


def to_numeric(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace(["-", "nan", "None", ""], np.nan)
        .astype(float)
    )


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    numeric_columns = [
        "fill_rate",
        "employment_rate",
        "advancement_rate",
        "competition_rate",
        "admission_quota",
        "recruitment_quota",
        "applicants",
        "entrants",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    categorical_columns = [
        "region",
        "foundation_type",
        "metro_local",
        "representative_group",
        "analysis_group",
    ]

    for col in categorical_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).replace("nan", "미상")

    df = df.dropna(
        subset=[
            "university",
            "fill_rate",
            "employment_rate",
            "advancement_rate",
            "competition_rate",
        ]
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_candidates = [
        "competition_rate",
        "employment_rate",
        "advancement_rate",
        "year",
        "admission_quota",
        "recruitment_quota",
        "applicants",
    ]

    categorical_candidates = [
        "region",
        "foundation_type",
        "metro_local",
        "representative_group",
        "analysis_group",
    ]

    numeric_features = [
        col for col in numeric_candidates
        if col in df.columns and col != "fill_rate"
    ]

    categorical_features = [
        col for col in categorical_candidates
        if col in df.columns
    ]

    return numeric_features, categorical_features


# =========================
# 4. 모델 구성
# =========================

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool
) -> ColumnTransformer:
    if scale_numeric:
        numeric_transformer = Pipeline(
            steps=[
                ("scaler", StandardScaler())
            ]
        )
    else:
        numeric_transformer = "passthrough"

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", make_one_hot_encoder())
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    return ColumnTransformer(transformers=transformers)


def build_models(
    numeric_features: list[str],
    categorical_features: list[str]
) -> dict[str, Pipeline]:
    models = {
        "Baseline_Mean": Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=False
                    )
                ),
                ("model", DummyRegressor(strategy="mean"))
            ]
        ),
        "Linear_Regression": Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=True
                    )
                ),
                ("model", LinearRegression())
            ]
        ),
        "Ridge_Regression": Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=True
                    )
                ),
                ("model", Ridge(alpha=1.0))
            ]
        ),
        "Random_Forest": Pipeline(
            steps=[
                (
                    "preprocess",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                        scale_numeric=False
                    )
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=42,
                        max_depth=4,
                        min_samples_leaf=2
                    )
                )
            ]
        ),
    }

    return models


# =========================
# 5. 평가 함수
# =========================

def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def evaluate_train_test(
    models: dict[str, Pipeline],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series
) -> tuple[pd.DataFrame, dict[str, Pipeline], pd.DataFrame]:
    rows = []
    fitted_models = {}

    prediction_df = pd.DataFrame({
        "actual_fill_rate": y_test.values,
    }, index=X_test.index)

    for model_name, model in models.items():
        fitted_model = clone(model)
        fitted_model.fit(X_train, y_train)

        train_pred = fitted_model.predict(X_train)
        test_pred = fitted_model.predict(X_test)

        train_metrics = calculate_metrics(y_train, train_pred)
        test_metrics = calculate_metrics(y_test, test_pred)

        rows.append({
            "model": model_name,
            "train_MAE": train_metrics["MAE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
        })

        prediction_df[f"pred_{model_name}"] = test_pred
        fitted_models[model_name] = fitted_model

    result = pd.DataFrame(rows).sort_values("test_RMSE")

    return result, fitted_models, prediction_df


def evaluate_cross_validation(
    models: dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> pd.DataFrame:
    rows = []

    kfold = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42
    )

    for model_name, model in models.items():
        fold_metrics = []

        for fold, (train_idx, test_idx) in enumerate(kfold.split(X), start=1):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)
            y_pred = fitted_model.predict(X_test)

            metrics = calculate_metrics(y_test, y_pred)
            metrics["fold"] = fold
            fold_metrics.append(metrics)

        fold_df = pd.DataFrame(fold_metrics)

        rows.append({
            "model": model_name,
            "cv_MAE_mean": fold_df["MAE"].mean(),
            "cv_MAE_std": fold_df["MAE"].std(),
            "cv_RMSE_mean": fold_df["RMSE"].mean(),
            "cv_RMSE_std": fold_df["RMSE"].std(),
            "cv_R2_mean": fold_df["R2"].mean(),
            "cv_R2_std": fold_df["R2"].std(),
        })

    result = pd.DataFrame(rows).sort_values("cv_RMSE_mean")

    return result


# =========================
# 6. 변수 중요도 / 계수 추출
# =========================

def get_feature_names(
    fitted_model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str]
) -> list[str]:
    preprocessor = fitted_model.named_steps["preprocess"]

    try:
        feature_names = preprocessor.get_feature_names_out()
        feature_names = [
            str(name)
            .replace("num__", "")
            .replace("cat__", "")
            for name in feature_names
        ]
        return feature_names
    except Exception:
        return numeric_features + categorical_features


def save_random_forest_importance(
    fitted_model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str]
) -> pd.DataFrame:
    feature_names = get_feature_names(
        fitted_model,
        numeric_features,
        categorical_features
    )

    importances = fitted_model.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv(
        TABLE_DIR / "random_forest_feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    top_importance = importance_df.head(15).sort_values("importance")

    plt.figure(figsize=(8, 6))
    plt.barh(top_importance["feature"], top_importance["importance"])
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "random_forest_feature_importance.png", dpi=300)
    plt.close()

    return importance_df


def save_linear_coefficients(
    fitted_model: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
    model_name: str
) -> pd.DataFrame:
    feature_names = get_feature_names(
        fitted_model,
        numeric_features,
        categorical_features
    )

    coefficients = fitted_model.named_steps["model"].coef_

    coef_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients),
    }).sort_values("abs_coefficient", ascending=False)

    file_name = f"{model_name.lower()}_coefficients.csv"

    coef_df.to_csv(
        TABLE_DIR / file_name,
        index=False,
        encoding="utf-8-sig"
    )

    top_coef = coef_df.head(15).copy()
    top_coef = top_coef.sort_values("coefficient")

    plt.figure(figsize=(8, 6))
    plt.barh(top_coef["feature"], top_coef["coefficient"])
    plt.title(f"{model_name} Coefficients")
    plt.xlabel("Coefficient")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"{model_name.lower()}_coefficients.png", dpi=300)
    plt.close()

    return coef_df


# =========================
# 7. 시각화 저장
# =========================

def save_actual_vs_predicted_plot(
    prediction_df: pd.DataFrame,
    model_name: str
) -> None:
    pred_col = f"pred_{model_name}"

    if pred_col not in prediction_df.columns:
        return

    actual = prediction_df["actual_fill_rate"]
    predicted = prediction_df[pred_col]

    min_value = min(actual.min(), predicted.min())
    max_value = max(actual.max(), predicted.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(actual, predicted, alpha=0.8)
    plt.plot([min_value, max_value], [min_value, max_value])
    plt.title(f"Actual vs Predicted Fill Rate - {model_name}")
    plt.xlabel("Actual Fill Rate")
    plt.ylabel("Predicted Fill Rate")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"actual_vs_predicted_{model_name}.png", dpi=300)
    plt.close()


def save_residual_plot(
    prediction_df: pd.DataFrame,
    model_name: str
) -> None:
    pred_col = f"pred_{model_name}"

    if pred_col not in prediction_df.columns:
        return

    predicted = prediction_df[pred_col]
    residual = prediction_df["actual_fill_rate"] - predicted

    plt.figure(figsize=(7, 5))
    plt.scatter(predicted, residual, alpha=0.8)
    plt.axhline(0)
    plt.title(f"Residual Plot - {model_name}")
    plt.xlabel("Predicted Fill Rate")
    plt.ylabel("Residual")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / f"residual_plot_{model_name}.png", dpi=300)
    plt.close()


# =========================
# 8. 분석 노트 저장
# =========================

def save_analysis_note(
    train_test_result: pd.DataFrame,
    cv_result: pd.DataFrame,
    rf_importance: pd.DataFrame | None,
    numeric_features: list[str],
    categorical_features: list[str],
    best_model_name: str
) -> None:
    best_test_row = train_test_result.loc[
        train_test_result["model"] == best_model_name
    ].iloc[0]

    best_cv_row = cv_result.loc[
        cv_result["model"] == best_model_name
    ].iloc[0]

    lines = []

    lines.append("# 신입생 충원율 예측모형 분석")
    lines.append("")
    lines.append("## 분석 개요")
    lines.append("본 분석은 2023~2025년 22개 대학 자료를 활용하여 신입생 충원율을 설명하거나 예측하는 주요 요인을 탐색하기 위해 수행하였다.")
    lines.append("종속변수는 신입생 충원율이며, 설명변수로 경쟁률, 취업률, 진학률, 연도, 지역, 설립구분, 수도권/지방, 분석그룹 등을 활용하였다.")
    lines.append("")
    lines.append("## 사용 변수")
    lines.append("")
    lines.append("### 수치형 변수")
    for col in numeric_features:
        lines.append(f"- {col}")

    lines.append("")
    lines.append("### 범주형 변수")
    for col in categorical_features:
        lines.append(f"- {col}")

    lines.append("")
    lines.append("## 모형 비교 결과")
    lines.append("")
    lines.append("훈련/검증 데이터 분할 결과 기준으로 가장 낮은 테스트 RMSE를 보인 모형은 다음과 같다.")
    lines.append("")
    lines.append(f"- 최적 모형: {best_model_name}")
    lines.append(f"- Test MAE: {best_test_row['test_MAE']:.3f}")
    lines.append(f"- Test RMSE: {best_test_row['test_RMSE']:.3f}")
    lines.append(f"- Test R-squared: {best_test_row['test_R2']:.3f}")
    lines.append("")
    lines.append("5-fold 교차검증 기준 결과는 다음과 같다.")
    lines.append("")
    lines.append(f"- CV MAE 평균: {best_cv_row['cv_MAE_mean']:.3f}")
    lines.append(f"- CV RMSE 평균: {best_cv_row['cv_RMSE_mean']:.3f}")
    lines.append(f"- CV R-squared 평균: {best_cv_row['cv_R2_mean']:.3f}")
    lines.append("")

    if rf_importance is not None and len(rf_importance) > 0:
        lines.append("## 랜덤포레스트 변수 중요도 상위 변수")
        lines.append("")
        for _, row in rf_importance.head(10).iterrows():
            lines.append(f"- {row['feature']}: {row['importance']:.4f}")
        lines.append("")

    lines.append("## 해석")
    lines.append("본 분석은 신입생 충원율을 단순히 관찰하는 데서 나아가, 경쟁률·취업률·진학률·지역 특성 등이 충원율 예측에 어느 정도 기여하는지 확인하기 위한 탐색적 예측모형 분석이다.")
    lines.append("다만 분석 대상이 22개 대학의 3개년 자료, 총 66개 관측치로 제한되어 있으므로 예측 성능은 일반화보다는 변수 간 관계와 설명 가능성을 확인하는 데 초점을 두어 해석해야 한다.")
    lines.append("따라서 본 결과는 대학 입학성과를 예측하는 최종 모형이라기보다, 입시 전략 수립을 위한 주요 지표의 상대적 중요도를 확인하는 보조 분석으로 활용하는 것이 적절하다.")

    with open(OUTPUT_DIR / "prediction_model_note.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =========================
# 9. 메인 실행
# =========================

def main() -> None:
    print("=" * 60)
    print("신입생 충원율 예측모형 분석 시작")
    print("=" * 60)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {INPUT_PATH}")

    df = read_csv_safely(INPUT_PATH)

    print(f"원본 데이터 행 수: {len(df)}")
    print(f"원본 컬럼 목록: {list(df.columns)}")

    df = standardize_columns(df)
    df = clean_data(df)

    print(f"정제 후 데이터 행 수: {len(df)}")

    numeric_features, categorical_features = get_feature_columns(df)

    print(f"수치형 설명변수: {numeric_features}")
    print(f"범주형 설명변수: {categorical_features}")

    feature_columns = numeric_features + categorical_features

    X = df[feature_columns].copy()
    y = df["fill_rate"].copy()

    metadata_columns = [
        col for col in [
            "year",
            "university",
            "region",
            "foundation_type",
            "metro_local",
            "representative_group",
            "analysis_group",
        ]
        if col in df.columns
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42
    )

    models = build_models(numeric_features, categorical_features)

    train_test_result, fitted_models, prediction_df = evaluate_train_test(
        models,
        X_train,
        X_test,
        y_train,
        y_test
    )

    cv_result = evaluate_cross_validation(
        models,
        X,
        y,
        n_splits=5
    )

    train_test_result.to_csv(
        TABLE_DIR / "model_performance_train_test.csv",
        index=False,
        encoding="utf-8-sig"
    )

    cv_result.to_csv(
        TABLE_DIR / "model_performance_cross_validation.csv",
        index=False,
        encoding="utf-8-sig"
    )

    test_metadata = df.loc[X_test.index, metadata_columns].copy()
    prediction_output = pd.concat(
        [
            test_metadata.reset_index(drop=True),
            prediction_df.reset_index(drop=True)
        ],
        axis=1
    )

    prediction_output.to_csv(
        TABLE_DIR / "test_prediction_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    real_models = train_test_result[
        train_test_result["model"] != "Baseline_Mean"
    ]

    best_model_name = real_models.sort_values("test_RMSE").iloc[0]["model"]

    save_actual_vs_predicted_plot(prediction_df, best_model_name)
    save_residual_plot(prediction_df, best_model_name)

    rf_importance = None

    if "Random_Forest" in fitted_models:
        rf_importance = save_random_forest_importance(
            fitted_models["Random_Forest"],
            numeric_features,
            categorical_features
        )

    if "Linear_Regression" in fitted_models:
        save_linear_coefficients(
            fitted_models["Linear_Regression"],
            numeric_features,
            categorical_features,
            "Linear_Regression"
        )

    if "Ridge_Regression" in fitted_models:
        save_linear_coefficients(
            fitted_models["Ridge_Regression"],
            numeric_features,
            categorical_features,
            "Ridge_Regression"
        )

    save_analysis_note(
        train_test_result=train_test_result,
        cv_result=cv_result,
        rf_importance=rf_importance,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        best_model_name=best_model_name
    )

    print("")
    print("[훈련/검증 분할 성능]")
    print(train_test_result)

    print("")
    print("[교차검증 성능]")
    print(cv_result)

    print("")
    print(f"최적 모형: {best_model_name}")

    print("=" * 60)
    print("분석 완료")
    print(f"결과 저장 위치: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()