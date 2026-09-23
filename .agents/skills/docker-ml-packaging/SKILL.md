---
name: docker-ml-packaging
description: Procedures and templates for containerizing a graduated project's serving API with Docker - multi-stage builds, monorepo build context, non-root user, health checks, and compose for a local MLflow server. Docker must be installed first (it is not installed on this machine yet).
---

# Docker ML Packaging & Containerization Skill

Containerization packages the Python runtime, dependencies, system libraries, and application code into an immutable, portable artifact that runs identically on local machines, cloud VMs, and Kubernetes clusters.

---

## 1. Production `Dockerfile` for FastAPI Serving

```dockerfile
# Stage 1: Base image
FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies (requirements live at the REPO ROOT)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the graduated project. The build context is the REPO ROOT, so the project
# folder is preserved and only that project is copied into the image.
ARG PROJECT=projects/01_diabetes_regression
COPY ${PROJECT}/src/ ./src/
COPY ${PROJECT}/configs/ ./configs/

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 2. Docker Compose for Local Multi-Service Development (`docker-compose.yml`)

```yaml
version: '3.8'

services:
  mlflow-server:
    image: python:3.11-slim
    command: >
      sh -c "pip install mlflow && mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root /mlruns --host 0.0.0.0 --port 5000"
    ports:
      - "5000:5000"
    volumes:
      - ./mlruns:/mlruns
      - ./mlflow.db:/mlflow.db

  model-api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow-server:5000
    depends_on:
      - mlflow-server
```

---

## 3. Essential Docker CLI Commands

**Prerequisite — verified:** `docker` is **not** installed on this machine
(`docker --version` → `The term 'docker' is not recognized`), so this milestone is
blocked until Docker Desktop is installed. Confirm with `docker --version` first.

```powershell
# Build from the REPO ROOT - the build context must include requirements.txt
docker build -t mlops-service:v1 --build-arg PROJECT=projects/01_diabetes_regression .

# Run, mapping port 8000
docker run -p 8000:8000 --name mlops-api mlops-service:v1

# Test the endpoint (PowerShell, not curl - curl.exe aliases differ on Windows)
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get

# Inspect a failing container - the fastest way to see a lifespan/loading error
docker logs mlops-api

# Stop and clean up
docker stop mlops-api; docker rm mlops-api
```

### The model-inside-the-container problem

`models:/<name>@champion` cannot resolve inside a container that has no tracking
store or registry. Choose deliberately:

| Approach | Trade-off |
| :--- | :--- |
| Mount the store: `-v ${PWD}/mlflow.db:/app/mlflow.db -v ${PWD}/mlruns:/app/mlruns` | simplest locally; image is not self-contained |
| Point at an MLflow server: `-e MLFLOW_TRACKING_URI=http://mlflow:5000` | realistic production shape (see the compose file in §2) |
| Bake the model into the image at build time via `mlflow.artifacts.download_artifacts` | self-contained, but now promotions require an image rebuild |

Only bake the model in if you accept rebuilding on every promotion.

### `.dockerignore` (keeps the build context small)

```
.venv/
.git/
.dvc/cache/
mlruns/
mlflow.db
.pytest_cache/
**/__pycache__/
projects/*/data/
projects/*/notebooks/
*.md
```

Without this, `docker build .` sends the entire virtualenv (hundreds of MB) to the
daemon on every build.
