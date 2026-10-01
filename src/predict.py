"""Inference utilities: model loading, single-customer prediction and SHAP.

The saved artifact ``models/churn_model.joblib`` contains a dictionary with

* ``pipeline``   : full sklearn Pipeline (FeatureEngineer -> preprocessor -> model)
* ``background`` : a small transformed training sample (used for linear SHAP)
* ``threshold``  : the tuned decision threshold (from ``src.train``)
* ``model_name`` : human-readable model name

SHAP values are computed on the model's raw output (log-odds margin for
tree models) and then aggregated back to the **original features**: the
contributions of one-hot encoded categories are summed so the explanation
reads in business terms ("Contract = Month-to-month increases risk").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_model() -> tuple[object, dict]:
    """Load the trained pipeline and its metadata.

    Returns
    -------
    (pipeline, artifact)
        The fitted sklearn Pipeline and the metadata dict saved alongside it.

    Raises
    ------
    FileNotFoundError
        If the model has not been trained yet.
    """
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found at {config.MODEL_PATH}. "
            "Train it first with: python -m src.train"
        )
    artifact = joblib.load(config.MODEL_PATH)
    if isinstance(artifact, dict) and "pipeline" in artifact:
        return artifact["pipeline"], artifact
    # Legacy bare pipeline
    return artifact, {"pipeline": artifact}


def load_metrics() -> dict:
    """Load the saved evaluation metrics (empty dict if unavailable)."""
    if config.METRICS_PATH.exists():
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    return {}


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
def predict_customer(
    customer: dict, pipeline=None, threshold: float | None = None
) -> dict:
    """Predict churn for a single customer given as a feature dictionary.

    Parameters
    ----------
    customer : dict with one value per raw feature (see config.RAW_FEATURES).
    pipeline : optional pre-loaded pipeline (loads the saved one if None).
    threshold : optional decision threshold (defaults to the tuned value).

    Returns
    -------
    dict with churn_probability, prediction_label, prediction
    ("HIGH CHURN RISK" / "LOW CHURN RISK") and confidence.
    """
    if pipeline is None:
        pipeline, artifact = load_model()
        if threshold is None:
            threshold = float(artifact.get("threshold", 0.5))
    elif threshold is None:
        threshold = 0.5

    df = pd.DataFrame([customer])
    proba = float(pipeline.predict_proba(df)[0, 1])
    label = int(proba >= threshold)

    return {
        "churn_probability": proba,
        "prediction_label": label,
        "prediction": "HIGH CHURN RISK" if label == 1 else "LOW CHURN RISK",
        "confidence": float(max(proba, 1.0 - proba)),
    }


# ---------------------------------------------------------------------------
# SHAP machinery
# ---------------------------------------------------------------------------
def _sigmoid(z: float) -> float:
    """Logistic function used to convert log-odds into probabilities."""
    return 1.0 / (1.0 + np.exp(-z))


def _make_explainer(model):
    """Return a SHAP explainer suited to the final estimator."""
    import shap

    if hasattr(model, "get_booster"):  # XGBoost (sklearn API)
        return shap.TreeExplainer(model)
    from sklearn.ensemble import RandomForestClassifier

    if isinstance(model, RandomForestClassifier):
        return shap.TreeExplainer(model)
    raise ValueError(
        f"SHAP explanation is not supported for {type(model).__name__}. "
        "The pipeline expects XGBoost, Random Forest or Logistic Regression."
    )


def _original_feature_name(output_name: str) -> str:
    """Map a preprocessor output column back to its original feature."""
    if output_name in config.NUMERICAL_FEATURES:
        return output_name
    if output_name in config.CATEGORICAL_FEATURES:
        return output_name
    # one-hot column "<feature>_<category>" -> "<feature>"
    for feature in config.CATEGORICAL_FEATURES:
        if output_name.startswith(feature + "_"):
            return feature
    return output_name


def _feature_display_value(output_name: str, original_name: str, df: pd.DataFrame):
    """Human-readable value of an original feature for one customer.

    Uses the customer's actual value from the (engineered) input row. The
    one-hot category suffix is only a fallback if the column is unavailable.
    """
    if original_name in df.columns:
        return df[original_name].iloc[0]
    return output_name[len(original_name) + 1:]  # one-hot category fallback


def _linear_contributions(pipeline, X_t):
    """Exact, coefficient-based contributions for a LogisticRegression.

    The pipeline standardises numerical features before the linear model,
    so each coefficient is converted back to "log-odds per raw unit" with
    ``coef / scale`` and multiplied by the centred raw value. One-hot
    indicators are multiplied by their coefficient directly. The sum of all
    contributions plus the intercept equals the model's log-odds output.
    """
    from sklearn.linear_model import LogisticRegression

    model = pipeline.named_steps["model"]
    if not isinstance(model, LogisticRegression):
        raise ValueError("_linear_contributions expects a LogisticRegression")

    preprocessor = pipeline.named_steps["preprocessor"]
    scaler = preprocessor.named_transformers_["numeric"].named_steps["scaler"]
    coefs = model.coef_[0]
    intercept = float(model.intercept_[0])
    X_t = np.asarray(X_t)

    contributions = np.zeros_like(X_t, dtype=float)
    for j, name in enumerate(list(preprocessor.get_feature_names_out())):
        if name in config.NUMERICAL_FEATURES:
            k = config.NUMERICAL_FEATURES.index(name)
            # undo the standardisation: log-odds per raw unit
            contributions[:, j] = (
                (X_t[:, j] - scaler.mean_[k]) * (coefs[j] / scaler.scale_[k])
            )
        else:
            contributions[:, j] = coefs[j] * X_t[:, j]  # one-hot indicator
    return contributions, intercept


def shap_values_many(pipeline, X: pd.DataFrame):
    """Compute SHAP values for many rows (global explanations).

    Returns (feature_names, shap_matrix, base_value) in the preprocessor's
    output space (log-odds margin). For tree models a SHAP TreeExplainer is
    used; for LogisticRegression the mathematically equivalent coefficient
    decomposition is used.
    """
    from sklearn.linear_model import LogisticRegression

    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Route the raw input through the feature-engineering step first
    # (idempotent if the columns already exist).
    features_step = pipeline.named_steps.get("features")
    if features_step is not None:
        X = features_step.transform(X)

    X_t = preprocessor.transform(X)
    names = list(preprocessor.get_feature_names_out())

    if isinstance(model, LogisticRegression):
        sv, base = _linear_contributions(pipeline, X_t)
        return names, sv, base

    explainer = _make_explainer(model)
    raw = explainer.shap_values(X_t)

    if isinstance(raw, list):  # RandomForest binary -> [class0, class1]
        sv = np.asarray(raw[1], dtype=float)
        base = float(np.asarray(explainer.expected_value).ravel()[1])
    else:
        sv = np.asarray(raw, dtype=float)
        base = float(np.asarray(explainer.expected_value).ravel()[0])
    return names, sv, base


def aggregate_shap(names: list[str], shap_row, X_row: pd.DataFrame) -> pd.DataFrame:
    """Aggregate one-hot SHAP contributions back to the original features."""
    shap_row = np.asarray(shap_row, dtype=float).ravel()
    contrib: dict[str, float] = {}
    value: dict[str, object] = {}

    for name, v in zip(names, shap_row):
        orig = _original_feature_name(name)
        contrib[orig] = contrib.get(orig, 0.0) + float(v)
        value[orig] = _feature_display_value(name, orig, X_row)

    table = pd.DataFrame(
        [{"feature": f, "value": value[f], "shap_value": contrib[f]} for f in contrib]
    )
    table["abs_shap"] = table["shap_value"].abs()
    table = table.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    table["direction"] = np.where(table["shap_value"] > 0, "increases", "decreases")
    return table


def explain_prediction(customer: dict, pipeline=None, top_n: int | None = None) -> dict:
    """Explain a single customer's prediction with SHAP.

    Returns
    -------
    dict with
    * ``base_value`` : the model's expected output (log-odds baseline),
    * ``table``      : one row per original feature, sorted by |SHAP|,
                       containing the feature value and impact direction.
    """
    if pipeline is None:
        pipeline, _ = load_model()

    df = pd.DataFrame([customer])
    # Engineered row used for human-readable feature values.
    features_step = pipeline.named_steps.get("features")
    engineered = features_step.transform(df) if features_step is not None else df

    names, sv, base = shap_values_many(pipeline, df)
    table = aggregate_shap(names, sv, engineered)
    if top_n:
        table = table.head(top_n).reset_index(drop=True)
    return {"base_value": base, "table": table}


def probability_shifts(explanation: dict) -> pd.DataFrame:
    """Approximate each feature's isolated impact on churn probability.

    For every feature, the model's output is re-evaluated with only that
    feature's SHAP contribution on top of the baseline, then mapped from
    log-odds to probability. This is the standard per-feature "impact in
    percentage points" approximation.
    """
    base = explanation["base_value"]
    p_base = _sigmoid(base)
    table = explanation["table"].copy()
    table["prob_shift"] = table["shap_value"].apply(
        lambda v: _sigmoid(base + v) - p_base
    )
    return table


# ---------------------------------------------------------------------------
# Explanation plots
# ---------------------------------------------------------------------------
def plot_waterfall(explanation: dict, probability: float) -> "matplotlib.figure.Figure":
    """Plot the top SHAP contributions as a probability-impact waterfall."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shifts = probability_shifts(explanation)
    top = shifts.head(10).iloc[::-1]  # largest at the top of the chart

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d62728" if s > 0 else "#2ca02c" for s in top["prob_shift"]]
    labels = [
        f"{row['feature']} = {row['value']}"
        for _, row in top.iterrows()
    ]
    ax.barh(labels, top["prob_shift"] * 100, color=colors, alpha=0.85)
    ax.axvline(0, color="grey", linewidth=0.8)
    ax.set_xlabel("Impact on churn probability (percentage points, approx.)")
    ax.set_title(
        f"SHAP waterfall — predicted churn probability {probability:.1%}"
    )
    for i, (v, name) in enumerate(zip(top["prob_shift"] * 100, labels)):
        offset = 0.15 if v >= 0 else -0.15
        ax.text(v + offset, i, f"{v:+.1f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_global_importance(pipeline, X: pd.DataFrame, top_n: int = 20,
                           save_path=None) -> "matplotlib.figure.Figure":
    """Bar chart of mean |SHAP| per original feature (global importance)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names, sv, _ = shap_values_many(pipeline, X)
    contrib = pd.DataFrame(sv, columns=names).abs().mean().reset_index()
    contrib.columns = ["output_feature", "mean_abs_shap"]
    contrib["feature"] = contrib["output_feature"].apply(_original_feature_name)
    importance = (
        contrib.groupby("feature")["mean_abs_shap"].sum()
        .sort_values(ascending=False).head(top_n)
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(importance.index[::-1], importance.values[::-1], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP value| (average impact on model output)")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_global_summary(pipeline, X: pd.DataFrame, top_n: int = 20,
                        save_path=None) -> "matplotlib.figure.Figure":
    """SHAP beeswarm summary plot on the original (aggregated) features."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    names, sv, _ = shap_values_many(pipeline, X)

    # Aggregate per original feature for a business-readable chart.
    agg = np.zeros((sv.shape[0], len(config.NUMERICAL_FEATURES) + len(config.CATEGORICAL_FEATURES)))
    all_features = config.NUMERICAL_FEATURES + config.CATEGORICAL_FEATURES
    for j, name in enumerate(names):
        orig = _original_feature_name(name)
        agg[:, all_features.index(orig)] += sv[:, j]

    display = pd.DataFrame(
        agg,
        columns=[f.replace("_", " ").title() for f in all_features],
    )
    importance = np.abs(agg).mean(axis=0)
    top_idx = np.argsort(importance)[::-1][:top_n]
    display = display.iloc[:, top_idx]

    fig = plt.figure(figsize=(9, 8))
    shap.summary_plot(agg[:, top_idx], display, show=False, max_display=top_n)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample = {
        "gender": "Male", "SeniorCitizen": "No", "Partner": "No",
        "Dependents": "No", "tenure": 3, "PhoneService": "Yes",
        "MultipleLines": "No", "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.5, "TotalCharges": 268.5,
    }
    result = predict_customer(sample)
    print(f"Prediction: {result['prediction']}")
    print(f"Churn probability: {result['churn_probability']:.1%} "
          f"(confidence {result['confidence']:.1%})")
    explanation = explain_prediction(sample)
    print("\nTop factors:")
    for _, row in explanation["table"].head(8).iterrows():
        print(f"  {row['feature']:>20} = {str(row['value']):<25} "
              f"-> {row['direction']} risk (SHAP {row['shap_value']:+.3f})")


