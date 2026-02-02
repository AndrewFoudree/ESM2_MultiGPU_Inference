"""
ESM-2 Multi-GPU FastAPI Inference Service

This service provides a production-ready API for protein sequence inference
using the ESM-2 model with automatic multi-GPU scaling.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model_manager import GPUDistributor, ModelManager
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global model manager instance
model_manager: ModelManager = None
gpu_distributor: GPUDistributor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI application.
    Handles model loading on startup and cleanup on shutdown.
    """
    global model_manager, gpu_distributor

    logger.info("Starting ESM-2 Multi-GPU Inference Service...")

    # Initialize GPU distributor and model manager
    gpu_distributor = GPUDistributor()
    model_manager = ModelManager(
        model_name=settings.MODEL_NAME, gpu_distributor=gpu_distributor
    )

    # Load models onto available GPUs
    await model_manager.load_models()

    logger.info(f"Service ready with {gpu_distributor.gpu_count} GPU(s) available")

    yield

    # Cleanup on shutdown
    logger.info("Shutting down service...")
    await model_manager.cleanup()
    logger.info("Service shutdown complete")


app = FastAPI(
    title="ESM-2 Multi-GPU Inference Service",
    description=(
        "Production-ready FastAPI service for ESM-2 protein language model "
        "inference with automatic multi-GPU scaling."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint",
    description="Returns service health status and GPU availability information.",
)
async def health_check() -> HealthResponse:
    """
    Health check endpoint reporting GPU availability and service status.

    Returns:
        HealthResponse: Service health status including GPU count and model status.
    """
    if model_manager is None:
        return HealthResponse(
            status="initializing",
            gpu_count=0,
            gpu_ids=[],
            model_loaded=False,
            model_name=settings.MODEL_NAME,
            message="Service is starting up",
        )

    return HealthResponse(
        status="healthy",
        gpu_count=gpu_distributor.gpu_count,
        gpu_ids=gpu_distributor.gpu_ids,
        model_loaded=model_manager.is_loaded,
        model_name=settings.MODEL_NAME,
        message=f"Service operational with {gpu_distributor.gpu_count} GPU(s)",
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Inference"],
    summary="Single sequence prediction",
    description="Perform inference on a single protein sequence.",
)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Single sequence prediction endpoint.

    Args:
        request: PredictionRequest containing the protein sequence.

    Returns:
        PredictionResponse: Model embeddings and metadata.

    Raises:
        HTTPException: If model is not loaded or inference fails.
    """
    if model_manager is None or not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is still initializing.",
        )

    try:
        result = await model_manager.predict_single(
            sequence=request.sequence,
            include_embeddings=request.include_embeddings,
            include_attention=request.include_attention,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Inference"],
    summary="Batch sequence prediction",
    description="Perform inference up to 64 protein sequences.",
)
async def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch prediction endpoint with multi-GPU distribution.

    Distributes incoming batch across all available GPUs for parallel inference.
    Maximum batch size is 64 sequences.

    Args:
        request: BatchPredictionRequest containing list of protein sequences.

    Returns:
        BatchPredictionResponse: Results for all sequences with GPU distribution info.

    Raises:
        HTTPException: If model not loaded, batch too large, or inference fails.
    """
    if model_manager is None or not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Service is still initializing.",
        )

    if len(request.sequences) > settings.MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Batch size {len(request.sequences)} "
                f"exceeds maximum of {settings.MAX_BATCH_SIZE}"
            ),
        )

    if len(request.sequences) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Empty batch provided"
        )

    try:
        result = await model_manager.predict_batch(
            sequences=request.sequences,
            include_embeddings=request.include_embeddings,
            include_attention=request.include_attention,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference failed: {str(e)}",
        )


@app.get(
    "/",
    tags=["Root"],
    summary="Root endpoint",
    description=(
        "Perform inference on up to 64 protein sequences " "in parallel across GPUs."
    ),
)
async def root():
    """Root endpoint with service information."""
    return {
        "service": "ESM-2 Multi-GPU Inference",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
