# Architecture & Technology Choices

This document explains **why** each technology was selected for this project.
It is written to be read before (or during) a technical interview.

---

## 1. System architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  React + TypeScript (Vite)  —  professional dashboard UI       │
│  Dashboard · Prediction · Model info · Batch prediction        │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTPS/JSON (REST)
┌──────────────────────────▼──────────────────────────────────────┐
│  FastAPI (Uvicorn)  —  REST API + Pydantic validation           │
│  GET /health · GET /model/info · POST /predict ·                │
│  POST /predict/batch · GET /dashboard/*                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │ in-process service layer
┌──────────────────────────▼──────────────────────────────────────┐
│  Prediction service → preprocessing pipeline → trained model    │
│  (scikit-learn Pipeline: FeatureEngineer → ColumnTransformer    │
│   → LogisticRegression)                                         │
│  + SHAP explanation service                                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  Model artefacts: models/churn_model.joblib + models/metrics.json│
│  Registered in the MLflow Model Registry (customer-churn-model) │
└─────────────────────────────────────────────────────────────────┘
```

The ML logic lives in `src/` and is consumed by the API; the React frontend
never touches the model directly — every call goes through the REST API.

---

## 2. Technology choices

### React + TypeScript + Vite
**Why:** the task was to replace Streamlit with a professional frontend.
React is the industry-standard UI library; TypeScript adds compile-time
safety for the API contracts; Vite provides a fast, modern build with a
built-in dev proxy. Tailwind CSS keeps the UI minimal without hand-written
CSS, and Recharts renders the dashboard charts from API data only.

### FastAPI + Pydantic + Uvicorn
**Why:** FastAPI is the de-facto standard for Python ML APIs. It gives
automatic OpenAPI docs (`/docs`), async support and first-class Pydantic
integration. Pydantic validates every request **before** any ML code runs —
invalid input is rejected with a structured 422 instead of crashing the
pipeline. Uvicorn is the recommended ASGI server for FastAPI.

### scikit-learn + XGBoost + SHAP
**Why:** these were already the core of the project and deliver the complete
modelling workflow: leak-free `Pipeline`/`ColumnTransformer` preprocessing,
three candidate models compared with stratified cross-validation, threshold
tuning, and SHAP explanations that read in business terms. Nothing here was
changed for the sake of change — the ML results are bit-identical to the
original baseline.


### MLflow
**Why:** experiment tracking + model registry in one tool. Every candidate
model logs its hyperparameters, CV metrics and training duration; the final
model is registered as `customer-churn-model` with a version number that the
API reports to its clients. This gives the project a real model lifecycle
(Development → Validation → Production) instead of a hard-coded `.pkl` path.

### Pandera
**Why:** lightweight, code-first data validation for pandas. The raw CSV is
validated (columns, dtypes, allowed values, ranges, missing-value rules,
duplicates) **before** training; failures stop the pipeline with a readable
error. Pandera keeps the project lightweight compared to Great Expectations,
which would add a heavier metadata server + configuration surface.

### DVC
**Why:** separates code from data in the same git repository. The raw
dataset and the generated model/processed-data artefacts are tracked by DVC
(content-addressed cache) while only code and pipeline definitions stay in
Git. `dvc repro` reproduces the whole training pipeline from the raw CSV.
No remote storage is configured — the local cache is the source of truth and
the README documents how to add one.

### Evidently
**Why:** a purpose-built library for ML monitoring reports (data quality,
feature drift, target drift, prediction drift). The monitoring module
generates a self-contained HTML report comparing the training window with a
"recent predictions" window — clearly labelled as a **simulated** scenario
because this project has no live production traffic.

### Docker + docker-compose
**Why:** reproducible environments for the API, the UI and MLflow. The
frontend uses a multi-stage build (Node build → nginx), the backend image
contains the ML package + trained artefacts, and neither image contains the
dataset, `.venv` or caches. Compose wires the three services together with
health checks.

### GitHub Actions
**Why:** continuous integration that runs on every push/PR: lint (ruff),
the full test suite with coverage, a repository-structure check and Docker
image builds. Deployment is deliberately **not** configured — there is
nothing to deploy to, and pretending otherwise would be dishonest.

### pytest + pytest-cov + httpx
**Why:** the existing suite (34 tests) is preserved and extended with API
tests. `TestClient` (httpx-based) exercises the real FastAPI app with the
real model, which catches integration bugs that pure unit tests miss.

### ruff
**Why:** one fast tool for linting + formatting with sensible defaults,
replacing the flake8/black/isort zoo. Configured in `pyproject.toml`.

---

## 3. Key design decisions

1. **The API owns the risk levels.** `LOW / MEDIUM / HIGH` is a business rule
   (`< 40%`, `40–70%`, `> 70%`) layered on top of the probability, separate
   from the 0.61 decision threshold that produces the CHURN/NO_CHURN label.
2. **The dashboard computes everything server-side.** The frontend renders
   numbers from `GET /dashboard/summary` and `GET /dashboard/drivers`, which
   are computed from the model's predictions on the hold-out test set — real
   numbers, clearly labelled as such.
3. **No fake production claims.** There is no cloud deployment, no
   authentication, no live monitoring. Everything that exists in the repo
   actually runs; everything else is documented as a future improvement.
4. **Deterministic pipeline.** Fixed `random_state`, fixed splits and
   deterministic cleaning mean re-running `dvc repro` reproduces the exact
   baseline metrics (verified against the original `metrics.json`).
