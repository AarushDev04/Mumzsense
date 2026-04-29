# Multi-stage Docker build for backend
# Base: python:3.11-slim
# Includes HuggingFace model cache
# MumzSense v1 — Multi-stage Docker build for backend
# Base: python:3.11-slim
# Stage 1: deps builder; Stage 2: lean runtime image

FROM python:3.11-slim AS builder

# Install build tools needed for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY backend/requirements.runtime.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.runtime.txt

# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS runtime

# System deps for scientific Python wheels used by runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy repo contents needed at runtime
COPY backend ./backend
COPY frontend ./frontend
COPY evals ./evals
COPY README.md ./README.md
COPY PRD.md ./PRD.md
COPY TRADEOFFS.md ./TRADEOFFS.md
COPY EVALS.md ./EVALS.md

# Create models directory (populated by training or mounted volume)
RUN mkdir -p /app/backend/models

# HuggingFace cache — pre-baked or mounted at runtime
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
# Enable real semantic embeddings (bge-large + multilingual-mpnet)
ENV USE_REAL_EMBEDDINGS=true
# Classifier confidence threshold (lowered from 0.60 while BiLSTM unavailable)
ENV CONFIDENCE_THRESHOLD=0.45
RUN mkdir -p /app/.cache/huggingface

# Non-root user for security
RUN useradd -m -u 1001 appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI port
EXPOSE 8000

# Health-check (Railway / Docker)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Entrypoint: uvicorn with sensible production defaults
WORKDIR /app/backend

CMD ["uvicorn", "main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--log-level", "info", \
    "--proxy-headers", \
    "--forwarded-allow-ips", "*"]