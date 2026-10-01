"""End-to-end training and evaluation entry point.

Run from the project root with::

    python -m src.train

The script performs the complete modelling workflow:

1. load + clean the raw data (deterministic business rules only),
2. engineer features (row-wise, no learned statistics -> no leakage),
3. stratified 60/20/20 train/validation/test split,
4. 5-fold stratified cross-validation of Logistic Regression, Random
   Forest and XGBoost (class imbalance handled with class weights /
   ``scale_pos_weight`` — SMOTE is deliberately not used, see README),
5. model selection (ROC-AUC first, F1 tie-break, interpretability last),
6. decision-threshold tuning on the validation set,
7. final training on train+validation and honest evaluation on the test set,
8. SHAP global explanations,
9. persistence of the pipeline, metrics and every report figure.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import xgboost as xgb  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, evaluate  # noqa: E402
from src.data_preprocessing import (  # noqa: E402
    build_preprocessor,
    clean_data,
    encode_target,
    load_raw_data,
    split_data,
)
from src.feature_engineering import FeatureEngineer, add_features  # noqa: E402
from src.predict import plot_global_importance, plot_global_summary  # noqa: E402

CV_SCORING = {
    "roc_auc": "roc_auc",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
}


# ---------------------------------------------------------------------------
# Candidate models
# ---------------------------------------------------------------------------
def build_model_pipelines(scale_pos_weight: float | None = None) -> dict:
    """Build the three candidate pipelines (feature engineering included).

    Class imbalance is handled natively:
    * Logistic Regression and Random Forest use balanced class weights,
    * XGBoost uses ``scale_pos_weight`` (ratio of negatives to positives),
    computed on the training set to avoid leakage.
    """
    def make_pipeline(estimator) -> Pipeline:
        return Pipeline(
            steps=[
                ("features", FeatureEngineer()),
                ("preprocessor", build_preprocessor()),
                ("model", estimator),
            ]
        )

    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=config.RANDOM_STATE,
            )
        ),
        "Random Forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced",
                min_samples_leaf=2,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "XGBoost": make_pipeline(
            xgb.XGBClassifier(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
        ),
    }


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
def select_model(cv_summary: pd.DataFrame) -> str:
    """Select the final model with a documented, business-aware rule.

    1. Primary criterion : highest mean cross-validation ROC-AUC.
    2. If several models are within 0.005 AUC, keep the one(s) with the
       highest mean F1 for the churn class.
    3. Final tie-break: prefer the more interpretable / cheaper model
       (Logistic Regression > XGBoost > Random Forest).
    """
    best_auc = cv_summary["roc_auc_mean"].max()
    candidates = cv_summary[cv_summary["roc_auc_mean"] >= best_auc - 0.005]

    if len(candidates) > 1:
        best_f1 = candidates["f1_mean"].max()
        candidates = candidates[candidates["f1_mean"] >= best_f1 - 0.005]

    interpretability = {
        "Logistic Regression": 0,
        "XGBoost": 1,
        "Random Forest": 2,
    }
    winner = candidates.assign(
        rank=candidates["model"].map(interpretability)
    ).sort_values("rank").iloc[0]
    return str(winner["model"])


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------
def main() -> dict:
    """Run the full training/evaluation workflow and persist all artefacts."""
    config.ensure_directories()
    print("=" * 72)
    print(" CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 72)

    # 1. Data ---------------------------------------------------------------
    print("\n[1/7] Loading and cleaning data ...")
    raw = load_raw_data()
    df = clean_data(raw)
    y = encode_target(df)
    X = add_features(df.drop(columns=[config.TARGET, config.ID_COLUMN]))
    print(f"       {raw.shape[0]} rows loaded, "
          f"{X.shape[1]} feature columns after engineering, "
          f"churn rate {y.mean():.1%}")

    # 2. Train / validation / test split -------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"[2/7] Stratified split: "
          f"train={len(X_train)}  validation={len(X_val)}  test={len(X_test)}")
    print(f"       scale_pos_weight (XGBoost) = {scale_pos_weight:.2f}")

    # 3. Cross-validation ----------------------------------------------------
    print(f"[3/7] {config.CV_FOLDS}-fold stratified cross-validation "
          f"(random_state={config.RANDOM_STATE}) ...")
    models = build_model_pipelines(scale_pos_weight)
    cv = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    cv_rows = []
    fitted_on_train = {}
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring=CV_SCORING, n_jobs=1, return_train_score=False,
        )
        # cross_validate fits clones; refit once for the validation plots.
        pipeline.fit(X_train, y_train)
        fitted_on_train[name] = pipeline
        cv_rows.append({
            "model": name,
            "roc_auc_mean": np.mean(scores["test_roc_auc"]),
            "roc_auc_std": np.std(scores["test_roc_auc"]),
            "f1_mean": np.mean(scores["test_f1"]),
            "f1_std": np.std(scores["test_f1"]),
            "recall_mean": np.mean(scores["test_recall"]),
            "recall_std": np.std(scores["test_recall"]),
            "precision_mean": np.mean(scores["test_precision"]),
            "precision_std": np.std(scores["test_precision"]),
        })
        print(f"       {name:<20} ROC-AUC={np.mean(scores['test_roc_auc']):.4f} "
              f"±{np.std(scores['test_roc_auc']):.4f} | "
              f"F1={np.mean(scores['test_f1']):.4f} | "
              f"Recall={np.mean(scores['test_recall']):.4f}")

    cv_summary = pd.DataFrame(cv_rows)
    selected_name = select_model(cv_summary)
    selected = fitted_on_train[selected_name]
    print(f"\n       Selected model: {selected_name}")

    # 4. Threshold tuning -----------------------------------------------------
    print(f"[4/7] Tuning decision threshold on the validation set "
          f"(maximising {config.THRESHOLD_METRIC}) ...")
    val_proba = selected.predict_proba(X_val)[:, 1]
    best_threshold, tune_metrics = evaluate.find_best_threshold(
        y_val, val_proba, metric=config.THRESHOLD_METRIC
    )
    print(f"       Best threshold = {best_threshold:.2f} "
          f"-> precision={tune_metrics['precision']:.3f}, "
          f"recall={tune_metrics['recall']:.3f}, f1={tune_metrics['f1']:.3f}")

    # 5. Final training on train+validation -----------------------------------
    print("[5/7] Refitting the selected model on train + validation ...")
    X_train_full = pd.concat([X_train, X_val], axis=0)
    y_train_full = pd.concat([y_train, y_val], axis=0)
    final_pipeline = Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("preprocessor", build_preprocessor()),
            ("model", models[selected_name].named_steps["model"]),
        ]
    )
    final_pipeline.fit(X_train_full, y_train_full)

    # 6. Honest evaluation on the test set -------------------------------------
    print("[6/7] Evaluating on the held-out test set ...")
    test_metrics_050 = evaluate.evaluate_model(
        final_pipeline, X_test, y_test, threshold=0.5
    )
    test_metrics_tuned = evaluate.evaluate_model(
        final_pipeline, X_test, y_test, threshold=best_threshold
    )
    print("       @ threshold 0.50   : "
          f"acc={test_metrics_050['accuracy']:.4f} | "
          f"prec={test_metrics_050['precision']:.4f} | "
          f"recall={test_metrics_050['recall']:.4f} | "
          f"f1={test_metrics_050['f1']:.4f} | auc={test_metrics_050['roc_auc']:.4f}")
    print("       @ tuned threshold  : "
          f"acc={test_metrics_tuned['accuracy']:.4f} | "
          f"prec={test_metrics_tuned['precision']:.4f} | "
          f"recall={test_metrics_tuned['recall']:.4f} | "
          f"f1={test_metrics_tuned['f1']:.4f} | auc={test_metrics_tuned['roc_auc']:.4f}")

    # 7. Reports, figures and persistence --------------------------------------
    print("[7/7] Generating figures and persisting artefacts ...")
    y_pred_test = (
        final_pipeline.predict_proba(X_test)[:, 1] >= best_threshold
    ).astype(int)

    evaluate.plot_roc_curves(
        fitted_on_train, X_val, y_val,
        save_path=config.FIGURES_DIR / "roc_curves.png",
    )
    evaluate.plot_precision_recall_curve(
        selected, X_val, y_val,
        save_path=config.FIGURES_DIR / "pr_curve.png",
    )
    evaluate.plot_threshold_curve(
        y_val, val_proba, best_threshold,
        save_path=config.FIGURES_DIR / "threshold_curve.png",
    )
    evaluate.plot_confusion_matrix(
        y_test, y_pred_test,
        title=f"Test set ({selected_name}, threshold={best_threshold:.2f})",
        save_path=config.FIGURES_DIR / "confusion_matrix_test.png",
    )
    plot_global_importance(
        final_pipeline, X_test,
        save_path=config.FIGURES_DIR / "shap_importance.png",
    )
    plot_global_summary(
        final_pipeline, X_test,
        save_path=config.FIGURES_DIR / "shap_summary.png",
    )

    preprocessor = final_pipeline.named_steps["preprocessor"]
    background = preprocessor.transform(X_train_full.iloc[:100])

    metrics = {
        "model_name": selected_name,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selection_rule": "ROC-AUC first; F1 (churn) tie-break within 0.005; "
                          "interpretability as final tie-break",
        "threshold": best_threshold,
        "threshold_metric": config.THRESHOLD_METRIC,
        "cv_table": cv_summary.to_dict(orient="records"),
        "validation_metrics_at_tuned_threshold": tune_metrics,
        "test_metrics_threshold_050": test_metrics_050,
        "test_metrics_tuned_threshold": test_metrics_tuned,
        "churn_rate_train": float(y_train_full.mean()),
        "churn_rate_test": float(y_test.mean()),
        "config": {
            "random_state": config.RANDOM_STATE,
            "test_size": config.TEST_SIZE,
            "validation_size": config.VALIDATION_SIZE,
            "cv_folds": config.CV_FOLDS,
        },
    }

    artifact = {
        "pipeline": final_pipeline,
        "background": background,
        "threshold": best_threshold,
        "model_name": selected_name,
        "trained_at": metrics["trained_at"],
    }
    joblib.dump(artifact, config.MODEL_PATH)
    config.METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, default=str), encoding="utf-8"
    )

    cv_summary.round(4).to_csv(config.COMPARISON_CSV_PATH, index=False)

    X_train_full.assign(**{config.TARGET: y_train_full}).to_csv(
        config.TRAIN_DATA_PATH, index=False
    )
    X_test.assign(**{config.TARGET: y_test}).to_csv(
        config.TEST_DATA_PATH, index=False
    )

    print(f"\nSaved model   : {config.MODEL_PATH}")
    print(f"Saved metrics : {config.METRICS_PATH}")
    print(f"Saved figures : {config.FIGURES_DIR}")
    print("\nTraining completed successfully.")
    return metrics


if __name__ == "__main__":
    main()
