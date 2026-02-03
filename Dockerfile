# ==============================================================================
# ESM-2 Multi-GPU Inference Service Dockerfile
# ==============================================================================
# Multi-stage build for production-ready container
# Targets: NVIDIA CUDA 12.1 + PyTorch 2.1 + FastAPI
# ==============================================================================

# ------------------------------------------------------------------------------
# Stage 1: Base image with CUDA and Python
# ------------------------------------------------------------------------------
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

# ------------------------------------------------------------------------------
# Stage 2: Builder - Install Python dependencies
# ------------------------------------------------------------------------------
FROM base AS builder

WORKDIR /build

# Copy requirements first for better layer caching
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support first (large download)
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ------------------------------------------------------------------------------
# Stage 3: Production image
# ------------------------------------------------------------------------------
FROM base AS production

# Labels
LABEL maintainer="MLOps Team"
LABEL description="ESM-2 Multi-GPU Inference Service"
LABEL version="1.0.0"

# Create non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Set up application directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY app/ ./app/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Application settings
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=1 \
    # CUDA settings
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    # HuggingFace cache
    HF_HOME=/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create cache directory with proper permissions
RUN mkdir -p /app/.cache/huggingface && \
    chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ------------------------------------------------------------------------------
# Stage 4: Development image (optional)
# ------------------------------------------------------------------------------
FROM production AS development

USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Copy test files
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY pyproject.toml .

USER appuser

# Override CMD for development
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
