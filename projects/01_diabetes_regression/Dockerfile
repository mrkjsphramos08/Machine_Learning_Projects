# Multi-stage production Dockerfile for Diabetes Serving API
FROM python:3.11-slim-bookworm as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install minimal system dependencies & apply security updates
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies from repo root requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source and configuration
COPY projects/01_diabetes_regression/src/ ./src/
COPY projects/01_diabetes_regression/configs/ ./configs/

# Create a non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
