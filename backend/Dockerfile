# Backend image — FastAPI + trained ML pipeline.
# Build context is the repository root (needed to copy src/ and models/).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System dependencies for the ML stack (libgomp is required by XGBoost).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for better layer caching. Only the runtime
# dependencies are installed — training-only tools (MLflow, DVC, Evidently,
# Pandera) stay out of the production image.
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Application code: the ML package, the trained artefacts and the backend
# app. Processed data (used by the dashboard) is deliberately NOT baked into
# the image — it is mounted at runtime by docker-compose.
COPY src /app/src
COPY models /app/models
COPY backend/app /app/backend/app
RUN mkdir -p /app/data/processed

ENV PYTHONPATH=/app:/app/backend

EXPOSE 8000

# Healthcheck: the API exposes /api/health and the container needs curl.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
