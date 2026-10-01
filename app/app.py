"""Streamlit web application for interactive churn prediction.

Run from the project root with::

    streamlit run app/app.py

The app loads the trained pipeline (``models/churn_model.joblib``) and
explains every prediction with the model's own feature contributions
(SHAP / exact linear decomposition), so the "why" behind each prediction
comes from the model — never invented.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless matplotlib backend for st.pyplot / SHAP figures.
os.environ.setdefault("MPLBACKEND", "Agg")

# Make the project root importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src import config  # noqa: E402
from src.predict import (  # noqa: E402
    explain_prediction,
    load_metrics,
    load_model,
    plot_waterfall,
    predict_customer,
    probability_shifts,
)

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="wide",
)

HIGH_RISK_COLOR = "#d62728"
LOW_RISK_COLOR = "#2ca02c"

CONTRACTS = ["Month-to-month", "One year", "Two year"]
PAYMENT_METHODS = [
    "Electronic check", "Mailed check",
    "Bank transfer (automatic)", "Credit card (automatic)",
]
INTERNET_SERVICES = ["DSL", "Fiber optic", "No"]
YES_NO = ["No", "Yes"]


@st.cache_resource
def get_model():
    """Load the trained pipeline once per Streamlit session."""
    return load_model()


def build_sidebar_inputs() -> dict:
    """Render the customer form in the sidebar and return the input values."""
    with st.sidebar:
        st.header("👤 Customer information")
        with st.form("customer_form"):
            st.subheader("Demographics")
            col_a, col_b = st.columns(2)
            with col_a:
                gender = st.selectbox("Gender", ["Female", "Male"])
                senior = st.selectbox("Senior citizen", YES_NO)
            with col_b:
                partner = st.selectbox("Partner", YES_NO)
                dependents = st.selectbox("Dependents", YES_NO)

            st.subheader("Account")
            tenure = st.slider("Tenure (months)", 0, 72, 1)
            contract = st.selectbox("Contract type", CONTRACTS)
            paperless = st.selectbox("Paperless billing", YES_NO)
            payment = st.selectbox("Payment method", PAYMENT_METHODS)

            st.subheader("Services")
            col_a, col_b = st.columns(2)
            with col_a:
                phone = st.selectbox("Phone service", YES_NO)
                multiple_lines = st.selectbox(
                    "Multiple lines", ["No", "Yes", "No phone service"]
                )
            with col_b:
                internet = st.selectbox("Internet service", INTERNET_SERVICES)
            service_values = {}
            for service in config.SERVICE_COLUMNS:
                options = (
                    ["No", "Yes", "No internet service"]
                    if internet == "No"
                    else ["No", "Yes"]
                )
                service_values[service] = st.selectbox(service, options)

            st.subheader("Charges")
            monthly = st.slider(
                "Monthly charges ($)", 18.0, 119.0, 70.0, step=0.5
            )
            total = st.number_input(
                "Total charges ($)", min_value=0.0,
                value=round(monthly * tenure, 2), step=1.0,
            )

            submitted = st.form_submit_button(
                "🔮 Predict churn", use_container_width=True, type="primary"
            )

    customer = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        **service_values,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    return customer, submitted


def render_prediction_banner(prediction: str) -> None:
    """Big, coloured banner with the binary outcome."""
    is_high = prediction == "HIGH CHURN RISK"
    background = HIGH_RISK_COLOR if is_high else LOW_RISK_COLOR
    st.markdown(
        f"""
        <div style="background-color:{background}; padding:18px;
                    border-radius:12px; text-align:center;">
            <h2 style="color:white; margin:0;">{prediction}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_gauge(probability: float, threshold: float) -> None:
    """Plotly gauge for the churn probability."""
    color = HIGH_RISK_COLOR if probability >= threshold else LOW_RISK_COLOR
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Churn probability"},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 100], "color": "rgba(0,0,0,0.05)"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=260, margin=dict(l=30, r=30, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_top_drivers(explanation: dict) -> None:
    """List the strongest feature contributions with direction arrows."""
    shifts = probability_shifts(explanation)
    st.markdown("#### 🔍 What drives this prediction?")
    for _, row in shifts.head(8).iterrows():
        increases = row["shap_value"] > 0
        arrow = "🔺" if increases else "🔻"
        color = HIGH_RISK_COLOR if increases else LOW_RISK_COLOR
        direction = "increases" if increases else "decreases"
        st.markdown(
            f"- {arrow} **{row['feature']}** = `{row['value']}` "
            f"→ *{direction}* churn risk "
            f"({row['prob_shift']:+.1f} pp, :{color}[approx.])"
        )


def main() -> None:
    """Render the application."""
    st.title("📉 Customer Churn Prediction")
    st.caption(
        "End-to-end demo — Logistic Regression pipeline with leak-free "
        "preprocessing, tuned decision threshold and SHAP explanations. "
        "Trained on the IBM Telco Customer Churn dataset."
    )

    try:
        pipeline, artifact = get_model()
    except FileNotFoundError:
        st.error(
            "⚠️ The trained model is missing (`models/churn_model.joblib`). "
            "Train it once with:"
        )
        st.code("python -m src.train", language="bash")
        st.stop()

    customer, submitted = build_sidebar_inputs()

    threshold = float(artifact.get("threshold", 0.5))

    if not submitted:
        st.info("👈 Fill in the customer profile in the sidebar, then click "
                "**Predict churn**.")
        return

    with st.spinner("Computing prediction and explanations ..."):
        result = predict_customer(
            customer, pipeline=pipeline, threshold=threshold
        )
        explanation = explain_prediction(customer, pipeline=pipeline)

    render_prediction_banner(result["prediction"])

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Churn probability", f"{result['churn_probability']:.1%}")
    col2.metric("Confidence", f"{result['confidence']:.1%}")
    col3.metric("Decision threshold", f"{threshold:.0%}")
    col4.metric("Model", artifact.get("model_name", "—"))

    left, right = st.columns([1, 1.1])
    with left:
        render_gauge(result["churn_probability"], threshold)
        render_top_drivers(explanation)
    with right:
        st.markdown("#### 🌊 SHAP waterfall")
        fig = plot_waterfall(explanation, result["churn_probability"])
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Waterfall: approximate impact of each feature on the predicted "
            "churn probability (percentage points). Positive = pushes the "
            "probability up, negative = pushes it down."
        )

    with st.expander("ℹ️ Model details & test performance"):
        metrics = load_metrics()
        if metrics:
            st.markdown(f"**Model:** {metrics.get('model_name')}")
            st.markdown(f"**Trained at:** {metrics.get('trained_at')}")
            st.markdown(f"**Selection rule:** {metrics.get('selection_rule')}")
            st.markdown("**Held-out test set metrics:**")
            rows = []
            for scenario, key in [
                ("threshold = 0.50", "test_metrics_threshold_050"),
                (f"tuned threshold = {metrics.get('threshold')}",
                 "test_metrics_tuned_threshold"),
            ]:
                m = metrics.get(key, {})
                rows.append({
                    "scenario": scenario,
                    "accuracy": round(m.get("accuracy", 0), 4),
                    "precision": round(m.get("precision", 0), 4),
                    "recall": round(m.get("recall", 0), 4),
                    "f1": round(m.get("f1", 0), 4),
                    "roc_auc": round(m.get("roc_auc", 0), 4),
                })
            st.dataframe(pd.DataFrame(rows).set_index("scenario"))
            st.caption(
                "Recall measures the share of real churners that were caught "
                "— the metric that matters most for a retention campaign."
            )

    st.divider()
    st.caption(
        "Predictions are probabilities, not certainties. This demo is for "
        "educational purposes and uses the IBM Telco sample dataset."
    )


if __name__ == "__main__":
    main()
