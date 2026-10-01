"""Validate the expected monorepo structure (used by CI and locally).

Checks that every file/directory a contributor or the pipeline depends on
exists, and fails with a clear message otherwise. Run with::

    python scripts/validate_structure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES: list[str] = [
    # Core ML package
    "src/config.py",
    "src/data_preprocessing.py",
    "src/feature_engineering.py",
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",
    "src/validation.py",
    # Backend
    "backend/app/main.py",
    "backend/app/core/config.py",
    "backend/app/api/routes/health.py",
    "backend/app/api/routes/prediction.py",
    "backend/app/api/routes/model.py",
    "backend/app/services/model_service.py",
    "backend/app/services/prediction_service.py",
    "backend/app/services/explanation_service.py",
    "backend/requirements.txt",
    "backend/Dockerfile",
    # Frontend
    "frontend/package.json",
    "frontend/vite.config.ts",
    "frontend/src/App.tsx",
    "frontend/src/pages/DashboardPage.tsx",
    "frontend/src/pages/PredictionPage.tsx",
    "frontend/src/pages/ModelInfoPage.tsx",
    "frontend/src/pages/BatchPredictionPage.tsx",
    "frontend/Dockerfile",
    # MLOps & infrastructure
    "dvc.yaml",
    "monitoring/generate_report.py",
    "monitoring/README.md",
    "docker-compose.yml",
    ".github/workflows/ci.yml",
    # Documentation & configuration
    "README.md",
    "docs/architecture.md",
    "models/README.md",
    "pyproject.toml",
    "requirements.txt",
    ".env.example",
    ".gitignore",
    "LICENSE",
]

REQUIRED_DIRS: list[str] = [
    "data",
    "models",
    "notebooks",
    "reports",
    "scripts",
    "tests",
    "backend/tests",
    "mlruns",
]


def main() -> int:
    """Return 0 when the structure is valid, 1 otherwise."""
    missing = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).is_file()]
    missing_dirs = [
        path for path in REQUIRED_DIRS if not (PROJECT_ROOT / path).is_dir()
    ]

    if missing or missing_dirs:
        for path in missing:
            print(f"MISSING FILE : {path}", file=sys.stderr)
        for path in missing_dirs:
            print(f"MISSING DIR  : {path}", file=sys.stderr)
        return 1

    print(f"Structure OK: {len(REQUIRED_FILES)} files and "
          f"{len(REQUIRED_DIRS)} directories present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
