"""
Pydantic schemas for ESM-2 Multi-GPU Inference Service

Defines request and response models for all API endpoints.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str = Field(
        ..., description="Service status: 'healthy', 'initializing', or 'unhealthy'"
    )
    gpu_count: int = Field(
        ..., description="Number of available GPUs (0, 1, 4, or 8)", ge=0
    )
    gpu_ids: List[int] = Field(
        default_factory=list, description="List of available GPU device IDs"
    )
    model_loaded: bool = Field(
        ..., description="Whether the ESM-2 model is loaded and ready"
    )
    model_name: str = Field(..., description="Name/identifier of the loaded model")
    message: str = Field(..., description="Human-readable status message")


class PredictionRequest(BaseModel):
    """Request model for single sequence prediction."""

    sequence: str = Field(
        ...,
        description="Protein sequence using standard amino acid codes",
        min_length=1,
        max_length=1024,
    )
    include_embeddings: bool = Field(
        default=True, description="Include sequence embeddings in response"
    )
    include_attention: bool = Field(
        default=False,
        description="Include attention weights in response (increases response size)",
    )

    @field_validator("sequence")
    @classmethod
    def validate_sequence(cls, v: str) -> str:
        """Validate protein sequence characters."""
        v = v.upper().strip()
        valid_chars = set("ACDEFGHIKLMNPQRSTVWXY")
        invalid = set(v) - valid_chars
        if invalid:
            raise ValueError(
                f"Invalid amino acid characters: {invalid}. "
                f"Valid characters are: {sorted(valid_chars)}"
            )
        return v

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQL",
                    "include_embeddings": True,
                    "include_attention": False,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for single sequence prediction."""

    sequence: str = Field(..., description="Input protein sequence")
    sequence_length: int = Field(..., description="Length of the input sequence")
    embedding: Optional[List[float]] = Field(
        default=None, description="Sequence embedding vector (mean-pooled)"
    )
    embedding_dim: Optional[int] = Field(
        default=None, description="Dimension of the embedding vector"
    )
    attention_shape: Optional[List[int]] = Field(
        default=None, description="Shape of attention weights [heads, seq_len, seq_len]"
    )
    gpu_id: int = Field(..., description="GPU device ID used for inference")
    inference_time_ms: float = Field(
        ..., description="Total inference time in milliseconds"
    )
    model_name: str = Field(..., description="Model used for inference")


class BatchPredictionRequest(BaseModel):
    """Request model for batch sequence prediction."""

    sequences: List[str] = Field(
        ...,
        description="List of protein sequences (max 64)",
        min_length=1,
        max_length=64,
    )
    include_embeddings: bool = Field(
        default=True, description="Include sequence embeddings in response"
    )
    include_attention: bool = Field(
        default=False, description="Include attention weights in response"
    )

    @field_validator("sequences")
    @classmethod
    def validate_sequences(cls, v: List[str]) -> List[str]:
        """Validate all protein sequences in batch."""
        valid_chars = set("ACDEFGHIKLMNPQRSTVWXY")
        validated = []

        for i, seq in enumerate(v):
            seq = seq.upper().strip()
            if not seq:
                raise ValueError(f"Empty sequence at index {i}")

            invalid = set(seq) - valid_chars
            if invalid:
                raise ValueError(f"Invalid characters in sequence {i}: {invalid}")

            if len(seq) > 1024:
                raise ValueError(f"Sequence {i} exceeds maximum length of 1024")

            validated.append(seq)

        return validated

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "sequences": [
                        "MKTVRQERLKSIVRILERSKEPVSGAQL",
                        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQ",
                        "MSGSHHHHHHSSGLVPRGSH",
                    ],
                    "include_embeddings": True,
                    "include_attention": False,
                }
            ]
        }
    }


class SequenceResult(BaseModel):
    """Result for a single sequence in batch prediction."""

    sequence: str = Field(..., description="Input protein sequence")
    sequence_length: int = Field(..., description="Length of the sequence")
    embedding: Optional[List[float]] = Field(
        default=None, description="Sequence embedding vector"
    )
    embedding_dim: Optional[int] = Field(
        default=None, description="Embedding dimension"
    )
    attention_shape: Optional[List[int]] = Field(
        default=None, description="Attention weights shape"
    )
    gpu_id: int = Field(..., description="GPU that processed this sequence")


class GPUDistributionInfo(BaseModel):
    """Information about how batch was distributed across GPUs."""

    total_sequences: int = Field(..., description="Total number of sequences in batch")
    num_gpus_used: int = Field(..., description="Number of GPUs used for this batch")
    sequences_per_gpu: Dict[str, int] = Field(
        ..., description="Number of sequences assigned to each GPU"
    )
    gpu_assignments: Dict[str, List[int]] = Field(
        ..., description="Mapping of GPU ID to sequence indices"
    )


class BatchPredictionResponse(BaseModel):
    """Response model for batch sequence prediction."""

    results: List[SequenceResult] = Field(
        ..., description="Prediction results for each sequence"
    )
    total_sequences: int = Field(..., description="Total number of sequences processed")
    total_inference_time_ms: float = Field(
        ..., description="Total inference time for entire batch"
    )
    average_time_per_sequence_ms: float = Field(
        ..., description="Average inference time per sequence"
    )
    distribution_info: GPUDistributionInfo = Field(
        ..., description="Details about GPU distribution"
    )
    model_name: str = Field(..., description="Model used for inference")
