"""FastAPI backend for the customer churn prediction platform.

The backend serves the trained ML pipeline (``src`` package) through a REST
API. ``src`` lives in the repository root, so the project root is added to
``sys.path`` here — this keeps the backend runnable from anywhere without
duplicating the ML code.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
