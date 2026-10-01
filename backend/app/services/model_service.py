"""Model service: loading, metadata and lifecycle of the deployed model.

The model is loaded **once** and cached for the lifetime of the process
(thread-safe ``functools.lru_cache``). The artefact is the joblib file
produced by ``python -m src.train``; it contains the fitted pipeline, the
tuned decision threshold, the training date and the MLflow registry version
recorded at training time.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

import joblib
from src import config as src_config

from app.core import config

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """Raised when the trained model artefact cannot be found or loaded."""


class ModelService:
    """Loads the trained pipeline once and exposes its metadata."""

    def __init__(
        self,
        model_path=None,
        metrics_path=None,
    ) -> None:
        self.model_path = model_path or config.MODEL_PATH
        self.metrics_path = metrics_path or config.METRICS_PATH
        self._pipeline: Any | None = None
        self._artifact: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the pipeline and its metadata (idempotent)."""
        if self._pipeline is not None:
            return
        if not self.model_path.exists():
            raise ModelNotAvailableError(
                f"Trained model not found at {self.model_path}. "
                "Train it first with: python -m src.train"
            )
        try:
            self._artifact = joblib.load(self.model_path)
        except Exception as exc:  # pragma: no cover - corrupted artefact
            raise ModelNotAvailableError(
                f"Could not load the model artefact: {exc}"
            ) from exc
        if isinstance(self._artifact, dict) and "pipeline" in self._artifact:
            self._pipeline = self._artifact["pipeline"]
        else:  # legacy bare pipeline
            self._pipeline = self._artifact
            self._artifact = {"pipeline": self._artifact}
        logger.info("Model loaded from %s", self.model_path)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    @property
    def pipeline(self) -> Any:
        """The fitted scikit-learn pipeline (loads lazily)."""
        self.load()
        return self._pipeline

    @property
    def artifact(self) -> dict[str, Any]:
        """The metadata saved alongside the pipeline."""
        self.load()
        return self._artifact or {}

    @property
    def threshold(self) -> float:
        """The tuned decision threshold."""
        return float(self.artifact.get("threshold", 0.5))

    @property
    def model_version(self) -> str:
        """The model version recorded at training time (MLflow registry)."""
        return str(
            self.artifact.get("model_version", src_config.FALLBACK_MODEL_VERSION)
        )

    @property
    def model_name(self) -> str:
        """Human-readable model name."""
        return str(self.artifact.get("model_name", "unknown"))

    @property
    def metrics(self) -> dict:
        """The evaluation metrics stored in ``models/metrics.json``."""
        if not self.metrics_path.exists():
            return {}
        try:
            return json.loads(self.metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.exception("Could not read metrics file %s", self.metrics_path)
            return {}

    def feature_count(self) -> int:
        """Number of features the pipeline was trained on."""
        try:
            return int(
                self.pipeline.named_steps["preprocessor"]
                .get_feature_names_out()
                .shape[0]
            )
        except Exception:
            return len(src_config.RAW_FEATURES) + len(
                src_config.ENGINEERED_FEATURES
            )

    def is_loaded(self) -> bool:
        """Whether the model artefact was successfully loaded."""
        try:
            self.load()
            return True
        except ModelNotAvailableError:
            return False


@lru_cache(maxsize=1)
def get_model_service() -> ModelService:
    """Return the process-wide singleton model service."""
    return ModelService()
