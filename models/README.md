# Model Registry

This directory holds the trained artefacts. Models are **not** managed by
hard-coded paths — the lifecycle is owned by the **MLflow Model Registry**,
with the local files as the deployment copies consumed by the API.

## Artefacts

| File | Purpose |
| --- | --- |
| `churn_model.joblib` | The fitted pipeline (`FeatureEngineer → ColumnTransformer → LogisticRegression`), decision threshold, training date, MLflow run id and registry version. |
| `metrics.json` | Full evaluation report: CV table, threshold-tuning results, test metrics at 0.50 and at the tuned threshold, and the MLflow run/version references. |

Both files are produced by `dvc repro` (or `python -m src.train`) and are
tracked by **DVC**, not Git — they are reproducible from the raw dataset.

## Lifecycle

```
Development ──► Validation ──► Production
   (train)        (metrics)      (served by the API)
```

1. **Development** — `python -m src.train` trains the three candidates,
   logs them to the MLflow experiment `customer-churn-prediction` and
   registers the winner as `customer-churn-model` (version N).
2. **Validation** — the test metrics in `metrics.json` are compared against
   the previous version; promotion to production is a documented decision.
3. **Production** — the API loads `churn_model.joblib` and reports the
   version from the artefact (`GET /model/info`), so clients always know
   which model is answering their requests.

## Versioning

* Each training run registers a new version in the MLflow Model Registry
  (version 1, 2, 3, …).
* The deployed version string is `<registry-version>.0.0` (e.g. `1.0.0`).
* The currently deployed version is visible in the dashboard, in
  `GET /model/info` and in every prediction response.

## Reproducing

```bash
# regenerate everything from the raw dataset (validation -> training ->
# evaluation -> MLflow registration)
dvc repro
```
