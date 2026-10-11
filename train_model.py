"""Train and evaluate the Student Performance Linear Regression pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "student_performance.csv"
MODELS_DIR = PROJECT_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "student_score_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"
COEFFICIENTS_PATH = MODELS_DIR / "feature_coefficients.csv"
DIAGNOSTICS_PATH = MODELS_DIR / "training_diagnostics.png"

TARGET = "Final_Exam_Score"
IDENTIFIER = "Student_ID"
NUMERIC_FEATURES = [
    "Age",
    "Study_Hours",
    "Attendance",
    "Previous_Score",
    "Sleep_Hours",
    "Internet_Hours",
    "Assignment_Score",
]
CATEGORICAL_FEATURES = ["Gender", "Extra_Classes", "Family_Support"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
REQUIRED_COLUMNS = [IDENTIFIER, *FEATURES, TARGET]


def read_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Read the CSV and validate its contract before any processing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}\n"
            "Place student_performance.csv next to train_model.py."
        )

    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise ValueError(f"Could not read the dataset at {path}: {exc}") from exc

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            "The dataset is missing required column(s): "
            f"{', '.join(missing_columns)}. Expected columns include: "
            f"{', '.join(REQUIRED_COLUMNS)}."
        )
    if frame.empty:
        raise ValueError("The dataset contains no student records.")
    return frame


def _normalize_category_values(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim category whitespace and standardize the Yes/No and Gender labels."""
    cleaned = frame.copy()
    for column in CATEGORICAL_FEATURES:
        values = cleaned[column].astype("string").str.strip()
        values = values.replace("", pd.NA)
        if column in {"Extra_Classes", "Family_Support"}:
            values = values.map(
                lambda value: (
                    {"yes": "Yes", "no": "No"}.get(str(value).casefold(), str(value))
                    if not pd.isna(value)
                    else pd.NA
                )
            ).astype("string")
        elif column == "Gender":
            values = values.map(
                lambda value: str(value).title() if not pd.isna(value) else pd.NA
            ).astype("string")
        cleaned[column] = values
    return cleaned


def build_pipeline() -> Pipeline:
    """Build all preprocessing inside the estimator to prevent data leakage."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
            (
                "one_hot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )


def _write_diagnostic_figure(
    actual: pd.Series,
    predicted: np.ndarray,
    output_path: Path,
) -> None:
    residuals = actual.to_numpy(dtype=float) - predicted
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(actual, predicted, color="#168c8c", alpha=0.82, edgecolor="white")
    low = min(float(actual.min()), float(predicted.min()))
    high = max(float(actual.max()), float(predicted.max()))
    axes[0].plot([low, high], [low, high], linestyle="--", color="#e07a42")
    axes[0].set(title="Actual vs. predicted scores", xlabel="Actual score", ylabel="Predicted score")

    axes[1].hist(residuals, bins=12, color="#3678a8", edgecolor="white")
    axes[1].axvline(0, linestyle="--", color="#e07a42")
    axes[1].set(title="Test-set residuals", xlabel="Actual − predicted", ylabel="Students")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def train_and_evaluate(
    data_path: Path = DATA_PATH,
    models_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    """Clean, split, train, evaluate and persist the full prediction bundle."""
    original = read_dataset(data_path)
    duplicate_count = int(original.duplicated().sum())
    frame = original.drop_duplicates().copy()
    frame = _normalize_category_values(frame)

    # The identifier is retained for traceability only. It never enters X.
    for column in [*NUMERIC_FEATURES, TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    missing_target_count = int(frame[TARGET].isna().sum())
    frame = frame.dropna(subset=[TARGET]).copy()
    if len(frame) < 5:
        raise ValueError(
            "At least 5 records with a valid Final_Exam_Score are required to train and test."
        )

    features = frame[FEATURES].copy()
    target = frame[TARGET].astype(float)
    row_indices = frame.index.to_numpy()
    x_train, x_test, y_train, y_test, train_indices, test_indices = train_test_split(
        features,
        target,
        row_indices,
        test_size=0.2,
        random_state=42,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    predicted = pipeline.predict(x_test)

    mse = float(mean_squared_error(y_test, predicted))
    metrics: dict[str, Any] = {
        "model": "Linear Regression",
        "target": TARGET,
        "features": FEATURES,
        "excluded_identifier": IDENTIFIER,
        "test_fraction": 0.2,
        "random_state": 42,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "rows_in_source": int(len(original)),
        "duplicate_rows_removed": duplicate_count,
        "rows_without_target_removed": missing_target_count,
        "mae": float(mean_absolute_error(y_test, predicted)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_test, predicted)),
        "sklearn_version": sklearn.__version__,
    }

    models_dir.mkdir(parents=True, exist_ok=True)
    # Resolve all outputs from the requested directory while keeping the default constants usable.
    pipeline_path = models_dir / PIPELINE_PATH.name
    metrics_path = models_dir / METRICS_PATH.name
    predictions_path = models_dir / PREDICTIONS_PATH.name
    coefficients_path = models_dir / COEFFICIENTS_PATH.name
    diagnostics_path = models_dir / DIAGNOSTICS_PATH.name
    joblib.dump(pipeline, pipeline_path)
    with metrics_path.open("w", encoding="utf-8") as output:
        json.dump(metrics, output, indent=2, allow_nan=False)

    pd.DataFrame(
        {
            "source_row_index": test_indices,
            "actual_score": y_test.to_numpy(dtype=float),
            "predicted_score": predicted,
            "residual": y_test.to_numpy(dtype=float) - predicted,
        }
    ).to_csv(predictions_path, index=False)

    feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    coefficients = pipeline.named_steps["model"].coef_
    pd.DataFrame(
        {
            "feature": feature_names,
            "coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False).to_csv(
        coefficients_path, index=False
    )
    _write_diagnostic_figure(y_test, predicted, diagnostics_path)

    print("Training complete.")
    print(f"Source records: {len(original)}")
    print(f"Training records: {metrics['train_rows']} | Test records: {metrics['test_rows']}")
    print(f"Duplicate rows removed: {duplicate_count}")
    print(f"Rows removed for missing target: {missing_target_count}")
    print(f"Student_ID excluded from model inputs: yes")
    print(
        "Metrics — "
        f"MAE: {metrics['mae']:.3f} | "
        f"MSE: {metrics['mse']:.3f} | "
        f"RMSE: {metrics['rmse']:.3f} | "
        f"R²: {metrics['r2']:.3f}"
    )
    print(f"Saved model pipeline: {pipeline_path}")
    print(f"Saved evaluation results: {metrics_path}")
    return metrics


if __name__ == "__main__":
    try:
        train_and_evaluate()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Training stopped: {error}") from error
