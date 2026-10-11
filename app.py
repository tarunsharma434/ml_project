"""Streamlit dashboard for exploring, evaluating and using the score model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from train_model import (
    CATEGORICAL_FEATURES,
    DATA_PATH,
    FEATURES,
    IDENTIFIER,
    NUMERIC_FEATURES,
    TARGET,
    read_dataset,
)


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "student_score_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "evaluation_metrics.json"
PREDICTIONS_PATH = MODELS_DIR / "test_predictions.csv"
COEFFICIENTS_PATH = MODELS_DIR / "feature_coefficients.csv"
DIAGNOSTICS_PATH = MODELS_DIR / "training_diagnostics.png"

INK = "#17324D"
TEAL = "#168C8C"
CORAL = "#E07A42"
BLUE = "#3678A8"

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: linear-gradient(145deg, #f5f8fa 0%, #edf4f5 100%); }
    [data-testid="stSidebar"] { background: #17324d; }
    [data-testid="stSidebar"] * { color: #f5fbff; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #f5fbff; }
    .hero {
        padding: 1.35rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(115deg, #17324d 0%, #176f78 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0.1rem 0 0.35rem 0; }
    .hero p { color: #e2f3f2; margin: 0; font-size: 1.02rem; }
    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.11em;
        font-size: 0.72rem;
        font-weight: 700;
        color: #b8e2dd;
    }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #d9e5e9;
        padding: 0.95rem 1rem;
        border-radius: 14px;
        box-shadow: 0 5px 18px rgba(23, 50, 77, 0.06);
    }
    .note {
        border-left: 4px solid #168c8c;
        background: #e8f4f2;
        padding: 0.85rem 1rem;
        border-radius: 0 10px 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    return read_dataset(DATA_PATH)


@st.cache_resource(show_spinner=False)
def load_model_artifacts() -> tuple[Any, dict[str, Any], pd.DataFrame, pd.DataFrame]:
    required = [PIPELINE_PATH, METRICS_PATH, PREDICTIONS_PATH, COEFFICIENTS_PATH]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved model file(s): {', '.join(missing)}. "
            "Run `python train_model.py` once, then restart the app."
        )
    pipeline = joblib.load(PIPELINE_PATH)
    with METRICS_PATH.open(encoding="utf-8") as file:
        metrics = json.load(file)
    predictions = pd.read_csv(PREDICTIONS_PATH)
    coefficients = pd.read_csv(COEFFICIENTS_PATH)
    return pipeline, metrics, predictions, coefficients


def style_figure(figure: go.Figure, height: int = 390) -> go.Figure:
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="white",
        font={"family": "Arial, sans-serif", "color": INK},
        margin={"l": 16, "r": 16, "t": 52, "b": 18},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    figure.update_xaxes(showgrid=True, gridcolor="#e8eef0", zerolinecolor="#cbd7dc")
    figure.update_yaxes(showgrid=True, gridcolor="#e8eef0", zerolinecolor="#cbd7dc")
    return figure


def show_hero(eyebrow: str, title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_mean(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").mean())


def coefficient_label(name: str) -> str:
    """Make transformed categorical feature names easier to read."""
    for prefix in ("Gender_", "Extra_Classes_", "Family_Support_"):
        if name.startswith(prefix):
            return f"{prefix[:-1]} = {name[len(prefix):]}"
    return name


def render_overview(data: pd.DataFrame, metrics: dict[str, Any]) -> None:
    show_hero(
        "LEARNING ANALYTICS · LINEAR REGRESSION",
        "Student Performance Predictor",
        "Explore the class dataset, understand the model, and estimate a score from a student's academic and study profile.",
    )
    st.subheader("At a glance")
    score = pd.to_numeric(data[TARGET], errors="coerce")
    columns = st.columns(4)
    columns[0].metric("Students in dataset", f"{len(data):,}")
    columns[1].metric("Average final score", f"{score.mean():.1f} / 100")
    columns[2].metric("Highest final score", f"{score.max():.0f} / 100")
    columns[3].metric("Lowest final score", f"{score.min():.0f} / 100")

    left, right = st.columns([1.35, 1])
    with left:
        st.subheader("A practical view of student outcomes")
        st.write(
            "This educational regression project learns patterns between student "
            "characteristics and final exam results. Use the explorer to inspect the "
            "uploaded records, then try a profile in the predictor."
        )
        st.markdown(
            '<div class="note"><strong>Important:</strong> A model estimate is not a '
            "guarantee of a student's result. Scores can also be affected by factors "
            "that are not present in this small dataset.</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.subheader("Held-out model snapshot")
        c1, c2 = st.columns(2)
        c1.metric("Mean absolute error", f"{metrics['mae']:.2f} points")
        c2.metric("R² score", f"{metrics['r2']:.3f}")
        st.caption(
            f"Evaluated on {metrics['test_rows']} test students, held out from "
            f"{metrics['rows_in_source']} source records. Split: 80% train / 20% test."
        )

    st.subheader("What the model uses")
    st.write(
        "Age, study hours, attendance, previous score, sleep, internet hours, "
        "assignment score, gender, extra classes, and family support. "
        "`Student_ID` is intentionally excluded because it identifies a row, "
        "not an academic characteristic."
    )
    with st.expander("How this estimate is produced"):
        st.write(
            "A preprocessing pipeline fills missing numeric values with training-set "
            "medians and missing categories with training-set most-common values. "
            "It one-hot encodes categories, then fits Linear Regression. The complete "
            "pipeline is saved and loaded by this app, so normal app startup does not retrain."
        )


def render_explorer(data: pd.DataFrame) -> None:
    show_hero(
        "SOURCE DATA",
        "Dataset Explorer",
        "Search and filter the uploaded student records, inspect data types, and review data quality.",
    )

    first, second, third = st.columns(3)
    first.metric("Rows × columns", f"{data.shape[0]} × {data.shape[1]}")
    second.metric("Missing cells", f"{int(data.isna().sum().sum())}")
    third.metric("Duplicate records", f"{int(data.duplicated().sum())}")

    with st.expander("Column information and descriptive statistics", expanded=False):
        info = pd.DataFrame(
            {
                "Column": data.columns,
                "Data type": [str(data[column].dtype) for column in data.columns],
                "Non-missing values": [int(data[column].notna().sum()) for column in data.columns],
                "Missing values": [int(data[column].isna().sum()) for column in data.columns],
                "Unique values": [int(data[column].nunique(dropna=True)) for column in data.columns],
            }
        )
        st.dataframe(info, width="stretch", hide_index=True)
        st.caption("Numeric descriptive statistics")
        st.dataframe(data.describe().T, width="stretch")

    st.subheader("Filter records")
    filter_columns = st.columns(3)
    filters: dict[str, list[str]] = {}
    for column, container in zip(CATEGORICAL_FEATURES, filter_columns):
        choices = sorted(data[column].dropna().astype(str).unique().tolist())
        filters[column] = container.multiselect(
            column.replace("_", " "),
            options=choices,
            default=choices,
            key=f"filter_{column}",
        )

    search = st.text_input(
        "Search all visible columns",
        placeholder="Type a name, category, or value to narrow the table",
    )
    filtered = data.copy()
    for column, selected in filters.items():
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]
        else:
            filtered = filtered.iloc[0:0]
    if search.strip():
        query = search.strip().casefold()
        matches = filtered.astype("string").apply(
            lambda column: column.str.casefold().str.contains(query, na=False, regex=False)
        )
        filtered = filtered[matches.any(axis=1)]
    st.caption(f"Showing {len(filtered):,} of {len(data):,} rows")
    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
        height=440,
    )

    quality_left, quality_right = st.columns([1.1, 1])
    with quality_left:
        st.subheader("Missing values by column")
        missing = data.isna().sum().rename("Missing cells").reset_index()
        missing.columns = ["Column", "Missing cells"]
        missing = missing[missing["Missing cells"] > 0]
        if missing.empty:
            st.success("No missing values found in the uploaded data.")
        else:
            chart = px.bar(
                missing,
                x="Missing cells",
                y="Column",
                orientation="h",
                color="Missing cells",
                color_continuous_scale=["#b9ded8", TEAL],
                title="Fields with missing data",
            )
            st.plotly_chart(style_figure(chart, 300), width="stretch")
    with quality_right:
        st.subheader("Cleaning used for training")
        st.write(
            f"The training script removed **{int(data.duplicated().sum())}** duplicate "
            "rows before splitting. Missing predictor values are imputed inside the "
            "pipeline using statistics learned from the training split only."
        )
        st.write(
            "Records with a missing target cannot be used to train or score the model; "
            "the training report counts any such removals."
        )


def render_visualizations(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
) -> None:
    show_hero(
        "EXPLORE PATTERNS",
        "Data Visualization",
        "Compare academic and lifestyle features with final exam outcomes. Patterns are descriptive, not proof of cause and effect.",
    )
    scatter_specs = [
        ("Study_Hours", "Study hours per day"),
        ("Attendance", "Attendance (%)"),
        ("Previous_Score", "Previous score"),
        ("Assignment_Score", "Assignment score"),
    ]
    for row_start in (0, 2):
        col1, col2 = st.columns(2)
        for spec, container in zip(scatter_specs[row_start : row_start + 2], (col1, col2)):
            feature, label = spec
            plot_data = data[[feature, TARGET, "Gender"]].dropna()
            chart = px.scatter(
                plot_data,
                x=feature,
                y=TARGET,
                color="Gender",
                color_discrete_sequence=[TEAL, CORAL, BLUE],
                labels={feature: label, TARGET: "Final exam score"},
                title=f"{label} vs. final exam score",
                hover_data={"Gender": True},
            )
            container.plotly_chart(style_figure(chart), width="stretch")

    left, right = st.columns(2)
    with left:
        chart = px.histogram(
            data,
            x=TARGET,
            nbins=16,
            color_discrete_sequence=[TEAL],
            labels={TARGET: "Final exam score", "count": "Number of students"},
            title="Final exam score distribution",
        )
        st.plotly_chart(style_figure(chart), width="stretch")
    with right:
        numeric = data.select_dtypes(include=np.number).drop(columns=[IDENTIFIER], errors="ignore")
        corr = numeric.corr()
        chart = px.imshow(
            corr,
            text_auto=".2f",
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu",
            origin="lower",
            title="Numeric feature correlation",
            labels={"color": "Correlation"},
        )
        st.plotly_chart(style_figure(chart, 420), width="stretch")
        st.caption(
            "Correlation describes a linear association between numeric columns; "
            "it does not show that one feature causes another."
        )

    st.subheader("Actual vs. predicted scores")
    chart = px.scatter(
        predictions,
        x="actual_score",
        y="predicted_score",
        labels={"actual_score": "Actual score", "predicted_score": "Predicted score"},
        title="Held-out test students",
        color_discrete_sequence=[BLUE],
        hover_data={"residual": ":.2f"},
    )
    low = float(min(predictions["actual_score"].min(), predictions["predicted_score"].min()))
    high = float(max(predictions["actual_score"].max(), predictions["predicted_score"].max()))
    chart.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            name="Perfect prediction",
            line={"color": CORAL, "dash": "dash"},
        )
    )
    st.plotly_chart(style_figure(chart, 440), width="stretch")


def make_suggestions(values: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []
    if values["Study_Hours"] < 3:
        suggestions.append(
            "Build a steady routine: add one focused study block at a time and review it weekly."
        )
    else:
        suggestions.append(
            "Keep study sessions consistent and use short practice questions to check understanding."
        )
    if values["Attendance"] < 75:
        suggestions.append(
            "Prioritize attending upcoming classes and catch up on missed topics with a teacher or classmate."
        )
    if values["Previous_Score"] < 60:
        suggestions.append(
            "Revisit foundational topics from earlier assessments before moving to harder material."
        )
    if values["Assignment_Score"] < 70:
        suggestions.append(
            "Break assignments into smaller checkpoints and ask for feedback on the first attempt."
        )
    if values["Sleep_Hours"] < 7:
        suggestions.append(
            "Protect a regular sleep schedule; adequate rest can support attention and learning."
        )
    if values["Internet_Hours"] > 5:
        suggestions.append(
            "Set clear study and leisure boundaries for internet time, while keeping learning resources accessible."
        )
    if values["Extra_Classes"] == "No":
        suggestions.append(
            "If a topic feels difficult, consider office hours, peer tutoring, or an extra class."
        )
    if values["Family_Support"] == "No":
        suggestions.append(
            "A trusted teacher, mentor, or study group can provide another source of encouragement."
        )
    return suggestions[:6]


def score_interpretation(score: float) -> str:
    if score >= 80:
        return "A strong estimate on the 0–100 score scale. Keep reinforcing the habits that are working."
    if score >= 70:
        return "A promising estimate. Consistent practice and feedback may help strengthen learning."
    if score >= 60:
        return "A developing estimate. Focused revision in weaker topics could be useful."
    return "This estimate suggests extra academic support and a clear revision plan may be helpful."


def render_predictor(data: pd.DataFrame, pipeline: Any) -> None:
    show_hero(
        "TRY A STUDENT PROFILE",
        "Predict a Final Score",
        "Enter a profile within reasonable ranges and receive a model estimate plus practical, rule-based study suggestions.",
    )
    st.markdown(
        '<div class="note">This estimate reflects patterns in the uploaded dataset. '
        "It is not a promise or a substitute for a teacher's guidance.</div>",
        unsafe_allow_html=True,
    )

    gender_options = sorted(data["Gender"].dropna().astype(str).unique().tolist())
    gender_options = gender_options or ["Female", "Male"]
    with st.form("score_prediction_form"):
        st.subheader("Academic and learning profile")
        row1 = st.columns(3)
        gender = row1[0].selectbox("Gender", gender_options)
        age = row1[1].number_input("Age (years)", min_value=14, max_value=30, value=20, step=1)
        study_hours = row1[2].number_input(
            "Study hours per day", min_value=0.0, max_value=24.0, value=4.5, step=0.5
        )

        row2 = st.columns(3)
        attendance = row2[0].number_input(
            "Attendance (%)", min_value=0.0, max_value=100.0, value=80.0, step=1.0
        )
        previous_score = row2[1].number_input(
            "Previous score", min_value=0.0, max_value=100.0, value=70.0, step=1.0
        )
        assignment_score = row2[2].number_input(
            "Assignment score", min_value=0.0, max_value=100.0, value=72.0, step=1.0
        )

        row3 = st.columns(3)
        sleep_hours = row3[0].number_input(
            "Sleep hours per day", min_value=0.0, max_value=24.0, value=7.0, step=0.5
        )
        internet_hours = row3[1].number_input(
            "Internet hours per day", min_value=0.0, max_value=24.0, value=4.0, step=0.5
        )
        extra_classes = row3[2].selectbox("Extra classes", ["Yes", "No"])

        family_support = st.selectbox("Family support", ["Yes", "No"])
        submitted = st.form_submit_button(
            "Predict Final Score",
            type="primary",
            width="stretch",
        )

    if not submitted:
        st.info("Complete the profile and select **Predict Final Score** to view an estimate.")
        return

    values: dict[str, Any] = {
        "Gender": gender,
        "Age": int(age),
        "Study_Hours": float(study_hours),
        "Attendance": float(attendance),
        "Previous_Score": float(previous_score),
        "Sleep_Hours": float(sleep_hours),
        "Internet_Hours": float(internet_hours),
        "Assignment_Score": float(assignment_score),
        "Extra_Classes": extra_classes,
        "Family_Support": family_support,
    }
    # Keep a stable feature order and names identical to training.
    input_frame = pd.DataFrame([{feature: values[feature] for feature in FEATURES}])
    predicted_score = float(pipeline.predict(input_frame)[0])
    gauge_value = float(np.clip(predicted_score, 0, 100))
    st.divider()
    score_col, gauge_col = st.columns([0.85, 1.15])
    with score_col:
        st.metric("Predicted final exam score", f"{predicted_score:.1f} / 100")
        st.write(score_interpretation(predicted_score))
        st.caption(
            "The gauge is displayed on a 0–100 scale. The numeric model estimate is shown as calculated."
        )
    with gauge_col:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=gauge_value,
                number={"suffix": " / 100", "valueformat": ".1f", "font": {"color": INK}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": INK},
                    "bar": {"color": TEAL},
                    "bgcolor": "white",
                    "steps": [
                        {"range": [0, 60], "color": "#f7dfd5"},
                        {"range": [60, 75], "color": "#f7edcf"},
                        {"range": [75, 100], "color": "#d8eee8"},
                    ],
                    "threshold": {
                        "line": {"color": CORAL, "width": 4},
                        "thickness": 0.8,
                        "value": gauge_value,
                    },
                },
                title={"text": "Estimated score range"},
            )
        )
        gauge.update_layout(height=250, margin={"l": 30, "r": 30, "t": 60, "b": 15})
        st.plotly_chart(gauge, width="stretch", config={"displayModeBar": False})

    st.subheader("Personalized study suggestions")
    for suggestion in make_suggestions(values):
        st.markdown(f"- {suggestion}")


def render_performance(
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    show_hero(
        "MODEL CHECK",
        "Model Performance",
        "Every score below is calculated from the held-out 20% test split, not from training records.",
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric("MAE", f"{metrics['mae']:.3f}")
    metric_columns[1].metric("MSE", f"{metrics['mse']:.3f}")
    metric_columns[2].metric("RMSE", f"{metrics['rmse']:.3f}")
    metric_columns[3].metric("R²", f"{metrics['r2']:.3f}")

    with st.expander("What do these numbers mean?", expanded=True):
        st.markdown(
            """
            - **MAE (Mean Absolute Error):** the average size of the prediction error, in score points. Smaller is generally better.
            - **MSE (Mean Squared Error):** squares each error before averaging, so larger misses count more. Its unit is points squared.
            - **RMSE (Root Mean Squared Error):** the square root of MSE, returned to score points. Smaller is generally better.
            - **R²:** compares the model with predicting the test-set average. Near 1 means the model explains more of the test variation; 0 is about the average baseline; a negative value is worse than that baseline.

            These metrics describe this one held-out split of a small dataset. They are not a guarantee of performance on other students.
            """
        )
    st.caption(
        f"Train/test: {metrics['train_rows']} / {metrics['test_rows']} students · "
        f"Random state: {metrics['random_state']} · "
        f"Scikit-learn: {metrics.get('sklearn_version', 'not recorded')}"
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        chart = px.scatter(
            predictions,
            x="actual_score",
            y="predicted_score",
            labels={"actual_score": "Actual score", "predicted_score": "Predicted score"},
            title="Actual vs. predicted test scores",
            color_discrete_sequence=[TEAL],
            hover_data={"residual": ":.2f"},
        )
        low = float(min(predictions["actual_score"].min(), predictions["predicted_score"].min()))
        high = float(max(predictions["actual_score"].max(), predictions["predicted_score"].max()))
        chart.add_trace(
            go.Scatter(
                x=[low, high],
                y=[low, high],
                mode="lines",
                name="Perfect prediction",
                line={"color": CORAL, "dash": "dash"},
            )
        )
        st.plotly_chart(style_figure(chart), width="stretch")

    with chart_right:
        chart = px.scatter(
            predictions,
            x="predicted_score",
            y="residual",
            labels={"predicted_score": "Predicted score", "residual": "Residual (actual − predicted)"},
            title="Residual plot",
            color_discrete_sequence=[BLUE],
        )
        chart.add_hline(y=0, line_dash="dash", line_color=CORAL)
        st.plotly_chart(style_figure(chart), width="stretch")

    st.subheader("Linear Regression coefficients")
    display = coefficients.copy()
    display["display_feature"] = display["feature"].map(coefficient_label)
    display = display.sort_values("coefficient")
    chart = px.bar(
        display,
        x="coefficient",
        y="display_feature",
        orientation="h",
        color="coefficient",
        color_continuous_scale=["#3678a8", "#f5f8fa", "#168c8c"],
        labels={"coefficient": "Coefficient (score points per feature unit)", "display_feature": "Model feature"},
        title="Learned coefficients",
    )
    st.plotly_chart(style_figure(chart, max(380, 34 * len(display))), width="stretch")
    st.write(
        "A positive numeric coefficient means the estimate rises by that amount for a "
        "one-unit increase in that feature, with the other model inputs held constant. "
        "A negative coefficient points the other way. For one-hot categories, the "
        "coefficient is relative to the dropped reference category learned by the pipeline. "
        "Coefficient size should be read in each feature's own units, not as a direct ranking of importance."
    )
    if DIAGNOSTICS_PATH.exists():
        with st.expander("Open the exported training diagnostic image"):
            st.image(
                str(DIAGNOSTICS_PATH),
                caption="Static Matplotlib diagnostic generated by train_model.py",
                width="stretch",
            )


def main() -> None:
    st.sidebar.markdown("## STUDENT INSIGHTS")
    st.sidebar.caption("Performance analysis · score prediction")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Home dashboard",
            "Dataset Explorer",
            "Data Visualization",
            "Predict Student Score",
            "Model Performance",
        ],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.caption("Primary model: Linear Regression")

    try:
        data = load_data()
    except (FileNotFoundError, ValueError) as error:
        st.error(str(error))
        st.stop()

    try:
        pipeline, metrics, predictions, coefficients = load_model_artifacts()
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as error:
        st.error(str(error))
        st.code("python train_model.py", language="bash")
        st.stop()

    if page == "Home dashboard":
        render_overview(data, metrics)
    elif page == "Dataset Explorer":
        render_explorer(data)
    elif page == "Data Visualization":
        render_visualizations(data, predictions)
    elif page == "Predict Student Score":
        render_predictor(data, pipeline)
    else:
        render_performance(metrics, predictions, coefficients)

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Source dataset: {DATA_PATH.name} · Target: {TARGET} · "
        f"Model inputs: {len(FEATURES)} (ID excluded)"
    )


if __name__ == "__main__":
    main()
