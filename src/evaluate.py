"""Model evaluation utilities: metrics, threshold tuning and diagnostic plots.

For churn prediction, accuracy alone is misleading: with a ~26.5% churn rate,
a model that always predicts "no churn" reaches ~73.5% accuracy while being
completely useless. The metrics reported here — especially **recall for the
churn class**, precision, F1 and ROC-AUC — reflect the real business
trade-off between catching churners (recall) and not wasting retention
effort on loyal customers (precision).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend (works in CI / notebooks)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


sns.set_theme(style="whitegrid", palette="muted")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Compute the full metric suite for a binary churn model.

    Returns accuracy, precision, recall, F1 (for the churn class) and
    ROC-AUC, plus the raw confusion matrix values (tn, fp, fn, tp).
    """
    y_pred = (np.asarray(y_proba)[:, 1] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]  # keep the positive-class probabilities only

    return {
        "threshold": threshold,
        "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def evaluate_model(
    model, X: pd.DataFrame, y: pd.Series, threshold: float = 0.5
) -> dict:
    """Evaluate a fitted sklearn pipeline on a dataset at a given threshold."""
    return compute_metrics(y, model.predict_proba(X), threshold=threshold)


def metrics_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Turn a list of metric dictionaries into a clean comparison table."""
    table = pd.DataFrame(metrics_list)
    keep = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    table = table[[c for c in keep if c in table.columns]]
    table[["accuracy", "precision", "recall", "f1", "roc_auc"]] = table[
        ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ].round(4)
    return table


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------
def find_best_threshold(
    y_true, y_proba, metric: str = "f1", step: float = 0.01
) -> tuple[float, dict]:
    """Grid-search the decision threshold that maximises the chosen metric.

    Default is F1: it balances recall (catching churners) against precision
    (not spamming loyal customers with retention offers). Use
    ``metric="recall"`` if the business prioritises catching churners.

    ``y_proba`` may be a 1-D array of positive-class probabilities or an
    ``(n, 2)`` predict_proba matrix.
    """
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]
    thresholds = np.arange(0.01, 1.0, step)
    scores = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        p = precision_score(y_true, y_pred, zero_division=0)
        r = recall_score(y_true, y_pred, zero_division=0)
        f = f1_score(y_true, y_pred, zero_division=0)
        scores.append({"threshold": t, "precision": p, "recall": r, "f1": f})

    curve = pd.DataFrame(scores)
    best_row = curve.loc[curve[metric].idxmax()]
    return float(best_row["threshold"]), best_row.to_dict()



# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    y_true, y_pred, title: str = "Confusion Matrix", save_path=None
) -> None:
    """Plot an absolute + row-normalised confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    labels = ["No churn", "Churn"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, fmt, cbar in zip(
        axes, [cm, cm_norm], ["d", ".1%"], [True, False], strict=True
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues", cbar=cbar,
            xticklabels=labels, yticklabels=labels, ax=ax,
            annot_kws={"size": 12},
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
    axes[0].set_title(f"{title} — counts")
    axes[1].set_title(f"{title} — row-normalised")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    models: dict[str, object], X, y, save_path=None
) -> pd.DataFrame:
    """Plot ROC curves for several fitted models and return AUC values."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey",
            label="Random (AUC=0.50)")
    rows = []
    for name, model in models.items():
        proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, proba)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)
        rows.append({"model": name, "auc": auc_val})
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves — validation set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pd.DataFrame(rows)


def plot_precision_recall_curve(model, X, y, save_path=None) -> float:
    """Plot the precision-recall curve of a model; return average precision."""
    proba = model.predict_proba(X)[:, 1]
    precision, recall, _ = precision_recall_curve(y, proba)
    ap = average_precision_score(y, proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, linewidth=2, label=f"Model (AP={ap:.3f})")
    ax.axhline(y.mean(), linestyle="--", color="grey",
               label=f"Baseline (churn rate={y.mean():.2f})")
    ax.set_xlabel("Recall (churn class)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return float(ap)


def plot_threshold_curve(
    y_true, y_proba, best_threshold: float | None = None, save_path=None
) -> pd.DataFrame:
    """Plot precision / recall / F1 against the decision threshold."""
    proba = np.asarray(y_proba)
    if proba.ndim == 2:
        proba = proba[:, 1]
    thresholds = np.arange(0.01, 1.0, 0.01)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        })
    curve = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for col, color in [("precision", "tab:blue"),
                       ("recall", "tab:orange"),
                       ("f1", "tab:green")]:
        ax.plot(curve["threshold"], curve[col],
                label=col.capitalize(), color=color, linewidth=2)
    if best_threshold is not None:
        ax.axvline(best_threshold, linestyle="--", color="red",
                   label=f"Selected threshold = {best_threshold:.2f}")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score")
    ax.set_title("Precision / Recall / F1 vs decision threshold")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return curve
