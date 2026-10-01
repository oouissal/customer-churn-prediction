"""Generate the churn model monitoring report (simulated production traffic).

**This is a reproducible, simulated monitoring scenario — not live
production monitoring.** The project does not run a production stream, so the
"recent prediction data" is played by the hold-out test split:

* **reference**  = training data (``data/processed/train.csv``)
* **current**    = hold-out test set scored by the deployed model
                  (``data/processed/test.csv``)

The report compares the two windows with Evidently and covers:

* data quality (missing values, distributions),
* feature drift (univariate + multivariate),
* target drift (churn rate),
* prediction drift (predicted churn probability distribution).

Run it with::

    python monitoring/generate_report.py

The HTML report is written to ``monitoring/reports/churn_monitoring_report.html``.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
from evidently.legacy.pipeline.column_mapping import ColumnMapping  # noqa: E402
from evidently.legacy.metric_preset import (  # noqa: E402
    DataDriftPreset,
    DataQualityPreset,
    TargetDriftPreset,
)
from evidently.legacy.metrics import ColumnDriftMetric  # noqa: E402
from evidently.legacy.report import Report  # noqa: E402
from src import config  # noqa: E402
from src.predict import load_model  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "monitoring" / "reports"
REPORT_PATH = REPORTS_DIR / "churn_monitoring_report.html"


def load_and_score() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train/test sets and add the model's churn probability column."""
    if not config.TRAIN_DATA_PATH.exists() or not config.TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run `python -m src.train` first."
        )
    pipeline, _ = load_model()
    reference = pd.read_csv(config.TRAIN_DATA_PATH)
    current = pd.read_csv(config.TEST_DATA_PATH)

    for frame in (reference, current):
        features = frame.drop(columns=[config.TARGET])
        frame["churn_probability"] = pipeline.predict_proba(features)[:, 1]

    return reference, current


def build_report(reference: pd.DataFrame, current: pd.DataFrame) -> Report:
    """Build the Evidently report comparing the two data windows."""
    column_mapping = ColumnMapping(
        target=config.TARGET,
        prediction="churn_probability",
        numerical_features=config.NUMERICAL_FEATURES,
        categorical_features=config.CATEGORICAL_FEATURES,
    )
    report = Report(
        metrics=[
            DataQualityPreset(),
            DataDriftPreset(),
            TargetDriftPreset(),
            ColumnDriftMetric(column_name="churn_probability"),
        ],
    )
    report.run(
        reference_data=reference,
        current_data=current,
        column_mapping=column_mapping,
    )
    return report


def main() -> Path:
    """Generate and persist the monitoring report."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Loading and scoring train/test windows ...")
    reference, current = load_and_score()
    print(f"  reference: {reference.shape[0]} rows (training window)")
    print(f"  current  : {current.shape[0]} rows (simulated recent traffic)")

    report = build_report(reference, current)
    report.save_html(str(REPORT_PATH))
    print(f"\nMonitoring report written to: {REPORT_PATH}")
    print("Note: this is a *simulated* monitoring scenario — the 'current'")
    print("window is the hold-out test set, not live production traffic.")
    return REPORT_PATH


if __name__ == "__main__":
    main()
