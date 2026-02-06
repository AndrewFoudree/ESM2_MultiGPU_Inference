"""
Configuration settings for ESM-2 Multi-GPU Inference Service
Uses Pydantic Settings for environment variable management.
"""

from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Environment Variables:
    - MODEL_NAME: HuggingFace model identifier
    - MAX_BATCH_SIZE: Maximum sequences per batch request
    - MAX_SEQUENCE_LENGTH: Maximum protein sequence length
    - CORS_ORIGINS: Allowed CORS origins (comma-separated)
    - LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)
    - WORKERS: Number of Uvicorn workers (for non-GPU deployment)
    - BATCH_QUEUE_ENABLED: Enable request batching for /predict endpoint
    - BATCH_QUEUE_MAX_WAIT_MS: Max time to wait for more requests
    - BATCH_QUEUE_MAX_SIZE: Max requests to batch together
    """

    # Model settings
    MODEL_NAME: str = "facebook/esm2_t33_650M_UR50D"
    MAX_BATCH_SIZE: int = 64
    MAX_SEQUENCE_LENGTH: int = 1024

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1  # Keep at 1 for GPU - each worker loads models

    # CORS settings
    CORS_ORIGINS: List[str] = ["*"]

    # Logging
    LOG_LEVEL: str = "INFO"

    # Health check settings
    STARTUP_TIMEOUT_SECONDS: int = 300  # 5 minutes for model loading

    # GPU settings
    CUDA_VISIBLE_DEVICES: str = ""  # Empty means use all available

    # Batch queue settings (for combining concurrent /predict requests)
    BATCH_QUEUE_ENABLED: bool = True
    BATCH_QUEUE_MAX_WAIT_MS: int = 50  # Wait up to 50ms for more requests
    BATCH_QUEUE_MAX_SIZE: int = 64  # Max requests to combine

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Global settings instance
settings = Settings()
