"""
Model Manager and GPU Distributor for ESM-2 Multi-GPU Inference

This module handles:
- Auto-detection of available GPUs
- Loading model replicas onto each GPU
- Distributing batch requests across GPUs for parallel inference
- Resource management and cleanup
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import torch

from app.config import settings
from app.schemas import (
    BatchPredictionResponse,
    GPUDistributionInfo,
    PredictionResponse,
    SequenceResult,
)

logger = logging.getLogger(__name__)


@dataclass
class GPUDistributor:
    """
    Handles GPU detection and batch distribution across available GPUs.

    GPU Distribution Strategy:
    -------------------------
    1. Auto-detect available GPUs using torch.cuda
    2. For batch requests, divide sequences evenly across GPUs
    3. Handle remainder by assigning extra sequences to first N GPUs
    4. Track which GPU processed each sequence for transparency

    Example distribution for 10 sequences across 4 GPUs:
    - GPU 0: sequences [0, 1, 2] (3 sequences)
    - GPU 1: sequences [3, 4, 5] (3 sequences)
    - GPU 2: sequences [6, 7] (2 sequences)
    - GPU 3: sequences [8, 9] (2 sequences)
    """

    _gpu_count: int = field(default=0, init=False)
    _gpu_ids: List[int] = field(default_factory=list, init=False)
    _device_type: str = field(default="cpu", init=False)

    def __post_init__(self):
        """Initialize GPU detection on creation."""
        self._detect_gpus()

    def _detect_gpus(self) -> None:
        """
        Auto-detect available NVIDIA GPUs using PyTorch CUDA.

        Sets:
            _gpu_count: Number of available GPUs (0, 1, 4, or 8 typically)
            _gpu_ids: List of GPU device indices
            _device_type: 'cuda' if GPUs available, 'cpu' otherwise
        """
        if torch.cuda.is_available():
            self._gpu_count = torch.cuda.device_count()
            self._gpu_ids = list(range(self._gpu_count))
            self._device_type = "cuda"

            # Log GPU information
            for i in self._gpu_ids:
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                logger.info(
                    f"Detected GPU {i}: {gpu_name} " f"({gpu_memory:.1f} GB memory)"
                )
        else:
            self._gpu_count = 0
            self._gpu_ids = []
            self._device_type = "cpu"
            logger.warning("No CUDA GPUs detected. Running on CPU.")

    @property
    def gpu_count(self) -> int:
        """Return the number of available GPUs."""
        return self._gpu_count

    @property
    def gpu_ids(self) -> List[int]:
        """Return list of GPU device IDs."""
        return self._gpu_ids.copy()

    @property
    def device_type(self) -> str:
        """Return device type ('cuda' or 'cpu')."""
        return self._device_type

    def get_device(self, gpu_id: int = 0) -> torch.device:
        """
        Get torch device for specified GPU.

        Args:
            gpu_id: GPU index to use (ignored if no GPUs available)

        Returns:
            torch.device: Device object for the specified GPU or CPU
        """
        if self._device_type == "cuda" and gpu_id < self._gpu_count:
            return torch.device(f"cuda:{gpu_id}")
        return torch.device("cpu")

    def distribute_batch(
        self, sequences: List[str]
    ) -> List[Tuple[int, List[int], List[str]]]:
        """
        Distribute a batch of sequences across available GPUs.

        Uses round-robin distribution with even splitting:
        1. Calculate base sequences per GPU: batch_size // num_gpus
        2. Calculate remainder: batch_size % num_gpus
        3. First 'remainder' GPUs get one extra sequence

        Args:
            sequences: List of protein sequences to distribute

        Returns:
            List of tuples: (gpu_id, sequence_indices, sequences_for_gpu)

        Example:
            For 10 sequences across 4 GPUs:
            [
                (0, [0,1,2], ['seq0','seq1','seq2']),
                (1, [3,4,5], ['seq3','seq4','seq5']),
                (2, [6,7], ['seq6','seq7']),
                (3, [8,9], ['seq8','seq9'])
            ]
        """
        num_sequences = len(sequences)

        # Handle case with no GPUs - run everything on CPU
        if self._gpu_count == 0:
            return [(0, list(range(num_sequences)), sequences)]

        distribution = []
        current_idx = 0

        # Calculate even distribution
        base_per_gpu = num_sequences // self._gpu_count
        remainder = num_sequences % self._gpu_count

        for gpu_id in self._gpu_ids:
            # First 'remainder' GPUs get one extra sequence
            count = base_per_gpu + (1 if gpu_id < remainder else 0)

            if count > 0:
                indices = list(range(current_idx, current_idx + count))
                gpu_sequences = [sequences[i] for i in indices]
                distribution.append((gpu_id, indices, gpu_sequences))
                current_idx += count

        return distribution

    def get_distribution_info(self, sequences: List[str]) -> GPUDistributionInfo:
        """
        Get detailed distribution information for logging/response.

        Args:
            sequences: Sequences to be distributed

        Returns:
            GPUDistributionInfo with distribution details
        """
        distribution = self.distribute_batch(sequences)

        gpu_assignments = {}
        for gpu_id, indices, _ in distribution:
            gpu_assignments[f"gpu_{gpu_id}"] = indices

        return GPUDistributionInfo(
            total_sequences=len(sequences),
            num_gpus_used=len(distribution),
            sequences_per_gpu={
                f"gpu_{gpu_id}": len(indices) for gpu_id, indices, _ in distribution
            },
            gpu_assignments=gpu_assignments,
        )


class ModelManager:
    """
    Manages ESM-2 model loading and inference across multiple GPUs.

    Architecture:
    ------------
    - Loads one model replica per GPU for maximum throughput
    - Uses ThreadPoolExecutor for parallel GPU inference
    - Handles model lifecycle (load, predict, cleanup)

    Multi-GPU Inference Flow:
    ------------------------
    1. Receive batch of sequences
    2. GPUDistributor splits batch across GPUs
    3. Each GPU processes its subset in parallel
    4. Results are gathered and reordered
    5. Return combined results with distribution metadata
    """

    def __init__(self, model_name: str, gpu_distributor: GPUDistributor):
        """
        Initialize ModelManager.

        Args:
            model_name: HuggingFace model identifier for ESM-2
            gpu_distributor: GPUDistributor instance for GPU management
        """
        self.model_name = model_name
        self.gpu_distributor = gpu_distributor
        self.models: Dict[int, Any] = {}  # gpu_id -> model
        self.tokenizers: Dict[int, Any] = {}  # gpu_id -> tokenizer
        self._is_loaded = False
        self._executor = ThreadPoolExecutor(
            max_workers=max(gpu_distributor.gpu_count, 1)
        )
        self._lock = asyncio.Lock()

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded and ready."""
        return self._is_loaded

    async def load_models(self) -> None:
        """
        Load ESM-2 model replicas onto all available GPUs.

        For each GPU:
        1. Load model from HuggingFace
        2. Move model to GPU device
        3. Set to evaluation mode
        4. Store reference for inference
        """
        from transformers import AutoModel, AutoTokenizer

        logger.info(f"Loading model: {self.model_name}")

        # Determine devices to load onto
        if self.gpu_distributor.gpu_count > 0:
            devices = [
                self.gpu_distributor.get_device(i) for i in self.gpu_distributor.gpu_ids
            ]
        else:
            devices = [torch.device("cpu")]

        # Load model onto each device
        for device in devices:
            device_id = device.index if device.type == "cuda" else 0

            logger.info(f"Loading model replica onto {device}")

            # Load tokenizer (shared across devices : keep separate for thread safety)
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizers[device_id] = tokenizer

            # Load and move model to device
            model = AutoModel.from_pretrained(self.model_name)
            model = model.to(device)
            model.eval()

            self.models[device_id] = model

            # Clear CUDA cache after loading
            if device.type == "cuda":
                torch.cuda.empty_cache()

        self._is_loaded = True
        logger.info(f"Successfully loaded {len(self.models)} model replica(s)")

    def _run_inference_on_gpu(
        self,
        gpu_id: int,
        sequences: List[str],
        include_embeddings: bool = True,
        include_attention: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Run inference on a specific GPU (synchronous, for thread pool).

        Args:
            gpu_id: GPU device ID to use
            sequences: Protein sequences to process
            include_embeddings: Whether to return embeddings
            include_attention: Whether to return attention weights

        Returns:
            List of result dictionaries for each sequence
        """
        model = self.models[gpu_id]
        tokenizer = self.tokenizers[gpu_id]
        device = self.gpu_distributor.get_device(gpu_id)

        results = []

        with torch.no_grad():
            # Tokenize batch
            inputs = tokenizer(
                sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=settings.MAX_SEQUENCE_LENGTH,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Run inference
            outputs = model(**inputs, output_attentions=include_attention)

            # Process each sequence result
            for i, seq in enumerate(sequences):
                result = {
                    "sequence": seq,
                    "sequence_length": len(seq),
                    "gpu_id": gpu_id,
                }

                if include_embeddings:
                    # Get sequence embedding (mean pooling over sequence length)
                    seq_len = inputs["attention_mask"][i].sum().item()
                    embedding = outputs.last_hidden_state[i, :seq_len].mean(dim=0)
                    result["embedding"] = embedding.cpu().numpy().tolist()
                    result["embedding_dim"] = len(result["embedding"])

                if include_attention:
                    # Get attention from last layer (simplified)
                    attention = outputs.attentions[-1][i].cpu().numpy()
                    result["attention_shape"] = list(attention.shape)

                results.append(result)

        return results

    async def predict_single(
        self,
        sequence: str,
        include_embeddings: bool = True,
        include_attention: bool = False,
    ) -> PredictionResponse:
        """
        Perform inference on a single protein sequence.

        Args:
            sequence: Protein sequence string
            include_embeddings: Whether to include embedding in response
            include_attention: Whether to include attention weights

        Returns:
            PredictionResponse with inference results
        """
        # Validate sequence
        self._validate_sequence(sequence)

        # Use first available GPU
        gpu_id = 0 if self.gpu_distributor.gpu_count > 0 else 0

        start_time = time.time()

        # Run inference in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            self._executor,
            self._run_inference_on_gpu,
            gpu_id,
            [sequence],
            include_embeddings,
            include_attention,
        )

        inference_time = time.time() - start_time
        result = results[0]

        return PredictionResponse(
            sequence=result["sequence"],
            sequence_length=result["sequence_length"],
            embedding=result.get("embedding"),
            embedding_dim=result.get("embedding_dim"),
            attention_shape=result.get("attention_shape"),
            gpu_id=result["gpu_id"],
            inference_time_ms=inference_time * 1000,
            model_name=self.model_name,
        )

    async def predict_batch(
        self,
        sequences: List[str],
        include_embeddings: bool = True,
        include_attention: bool = False,
    ) -> BatchPredictionResponse:
        """
        Perform batch inference with multi-GPU distribution.

        Multi-GPU Distribution Algorithm:
        ---------------------------------
        1. Validate all sequences
        2. Get distribution plan from GPUDistributor
        3. Submit inference tasks to thread pool (one per GPU)
        4. Await all results in parallel
        5. Reorder results to match original sequence order
        6. Return combined results with distribution metadata

        Args:
            sequences: List of protein sequences
            include_embeddings: Whether to include embeddings
            include_attention: Whether to include attention weights

        Returns:
            BatchPredictionResponse with all results and distribution info
        """
        # Validate all sequences
        for seq in sequences:
            self._validate_sequence(seq)

        # Get distribution plan
        distribution = self.gpu_distributor.distribute_batch(sequences)
        distribution_info = self.gpu_distributor.get_distribution_info(sequences)

        start_time = time.time()
        loop = asyncio.get_event_loop()

        # Submit parallel inference tasks
        tasks = []
        for gpu_id, indices, gpu_sequences in distribution:
            task = loop.run_in_executor(
                self._executor,
                self._run_inference_on_gpu,
                gpu_id,
                gpu_sequences,
                include_embeddings,
                include_attention,
            )
            tasks.append((indices, task))

        # Gather all results
        all_results = [None] * len(sequences)

        for indices, task in tasks:
            results = await task
            for idx, result in zip(indices, results):
                all_results[idx] = result

        total_time = time.time() - start_time

        # Convert to SequenceResult objects
        sequence_results = [
            SequenceResult(
                sequence=r["sequence"],
                sequence_length=r["sequence_length"],
                embedding=r.get("embedding"),
                embedding_dim=r.get("embedding_dim"),
                attention_shape=r.get("attention_shape"),
                gpu_id=r["gpu_id"],
            )
            for r in all_results
        ]

        return BatchPredictionResponse(
            results=sequence_results,
            total_sequences=len(sequences),
            total_inference_time_ms=total_time * 1000,
            average_time_per_sequence_ms=(total_time * 1000) / len(sequences),
            distribution_info=distribution_info,
            model_name=self.model_name,
        )

    def _validate_sequence(self, sequence: str) -> None:
        """
        Validate a protein sequence.

        Args:
            sequence: Protein sequence to validate

        Raises:
            ValueError: If sequence is invalid
        """
        if not sequence:
            raise ValueError("Empty sequence provided")

        if len(sequence) > settings.MAX_SEQUENCE_LENGTH:
            raise ValueError(
                f"Sequence length {len(sequence)} exceeds maximum "
                f"of {settings.MAX_SEQUENCE_LENGTH}"
            )

        # Standard amino acid alphabet
        valid_chars = set("ACDEFGHIKLMNPQRSTVWXY")
        invalid_chars = set(sequence.upper()) - valid_chars

        if invalid_chars:
            raise ValueError(f"Invalid characters in sequence: {invalid_chars}")

    async def cleanup(self) -> None:
        """Clean up models and free GPU memory."""
        logger.info("Cleaning up model resources...")

        for gpu_id, model in self.models.items():
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        self.models.clear()
        self.tokenizers.clear()
        self._is_loaded = False
        self._executor.shutdown(wait=True)

        logger.info("Cleanup complete")
