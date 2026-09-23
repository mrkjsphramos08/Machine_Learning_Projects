---
name: docker-ml-packaging
description: Guidelines and production templates for containerizing ML training pipelines and FastAPI serving microservices with Docker and multi-stage builds.
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

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and configs
COPY src/ ./src/
COPY configs/ ./configs/

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

```powershell
# Build Docker image
docker build -t mlops-serving:v1 .

# Run container locally mapping port 8000
docker run -p 8000:8000 --name mlops-api mlops-serving:v1

# Test endpoint
curl http://localhost:8000/health

# Stop and clean up container
docker stop mlops-api && docker rm mlops-api
```
