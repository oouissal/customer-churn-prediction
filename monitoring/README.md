# Model Monitoring

Lightweight monitoring module built with
[Evidently](https://www.evidentlyai.com/).

## ⚠️ Honest scope note

This is a **simulated / reproducible monitoring scenario** — the project does
**not** have a live production stream, and this module does not claim to be
real production monitoring. The "recent prediction data" window is played by
the hold-out **test set**, scored by the deployed model. The point is to
demonstrate *how* production monitoring would work: reference vs. current
comparison, drift detection, and a shareable HTML report.

## What the report covers

| Concern | How it is measured |
| --- | --- |
| Data quality | missing values, share of missing cells, distribution stats |
| Feature drift | univariate drift per feature + multivariate drift (PCA/domain) |
| Target drift | churn rate in reference vs. current window |
| Prediction drift | drift of the predicted churn probability distribution |

Reference window = training data (`data/processed/train.csv`, 4,225 rows).
Current window = hold-out test set (`data/processed/test.csv`, 1,409 rows).

## Usage

```bash
# from the repository root, with the virtualenv active
python monitoring/generate_report.py
```

The HTML report is written to `monitoring/reports/churn_monitoring_report.html`
(open it in any browser).

In a real production system the same script would read the current window from
a feature store / prediction log instead of the test set.
