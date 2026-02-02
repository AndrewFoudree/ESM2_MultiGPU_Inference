"""
Tests for ModelManager functionality.

Tests model loading, inference, and multi-GPU coordination.
"""

from unittest.mock import MagicMock

import pytest


class TestModelManager:
    """Test suite for ModelManager class."""

    @pytest.mark.asyncio
    async def test_model_loading_cpu(self, mock_torch_no_cuda, mock_transformers):
        """Test model loads on CPU when no GPUs available."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        await manager.load_models()

        assert manager.is_loaded
        assert 0 in manager.models
        mock_transformers["model"].assert_called_once()
        mock_transformers["tokenizer"].assert_called_once()

    @pytest.mark.asyncio
    async def test_model_loading_4_gpus(self, mock_torch_cuda, mock_transformers):
        """Test model loads replicas on all 4 GPUs."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        await manager.load_models()

        assert manager.is_loaded
        # Should have 4 model replicas
        assert len(manager.models) == 4
        assert len(manager.tokenizers) == 4
        # Model should be called 4 times (once per GPU)
        assert mock_transformers["model"].call_count == 4

    @pytest.mark.asyncio
    async def test_model_loading_8_gpus(self, mock_torch_cuda_8gpu, mock_transformers):
        """Test model loads replicas on all 8 GPUs."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        await manager.load_models()

        assert manager.is_loaded
        assert len(manager.models) == 8
        assert mock_transformers["model"].call_count == 8

    def test_sequence_validation_valid(self, mock_torch_no_cuda):
        """Test valid sequence passes validation."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        # Should not raise
        manager._validate_sequence("MKTVRQERLKSIVRILERSKEPVSGAQL")
        manager._validate_sequence("ACDEFGHIKLMNPQRSTVWXY")

    def test_sequence_validation_empty(self, mock_torch_no_cuda):
        """Test empty sequence fails validation."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        with pytest.raises(ValueError, match="Empty sequence"):
            manager._validate_sequence("")

    def test_sequence_validation_invalid_chars(self, mock_torch_no_cuda):
        """Test invalid characters fail validation."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        with pytest.raises(ValueError, match="Invalid characters"):
            manager._validate_sequence("MKTVRQ123")

        with pytest.raises(ValueError, match="Invalid characters"):
            manager._validate_sequence("MKT!@#VRQ")

    def test_sequence_validation_too_long(self, mock_torch_no_cuda):
        """Test sequence exceeding max length fails validation."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        long_sequence = "M" * 2000  # Exceeds 1024 limit

        with pytest.raises(ValueError, match="exceeds maximum"):
            manager._validate_sequence(long_sequence)

    @pytest.mark.asyncio
    async def test_cleanup(self, mock_torch_cuda, mock_transformers):
        """Test cleanup releases resources."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        await manager.load_models()
        assert manager.is_loaded

        await manager.cleanup()

        assert not manager.is_loaded
        assert len(manager.models) == 0
        assert len(manager.tokenizers) == 0


class TestModelManagerInference:
    """Test suite for ModelManager inference operations."""

    @pytest.mark.asyncio
    async def test_predict_single_returns_embedding(
        self, mock_torch_no_cuda, mock_transformers, mock_model, mock_tokenizer
    ):
        """Test single prediction returns embedding."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        # Manually set up models with mocks
        manager.models = {0: mock_model}
        manager.tokenizers = {0: mock_tokenizer}
        manager._is_loaded = True

        # Mock the _run_inference_on_gpu method
        def mock_inference(gpu_id, sequences, include_embeddings, include_attention):
            return [
                {
                    "sequence": sequences[0],
                    "sequence_length": len(sequences[0]),
                    "embedding": [0.1] * 1280,
                    "embedding_dim": 1280,
                    "gpu_id": gpu_id,
                }
            ]

        manager._run_inference_on_gpu = mock_inference

        result = await manager.predict_single(
            sequence="MKTVRQERLKSIVRILERSKEPVSGAQL", include_embeddings=True
        )

        assert result.sequence == "MKTVRQERLKSIVRILERSKEPVSGAQL"
        assert result.embedding is not None
        assert len(result.embedding) == 1280
        assert result.gpu_id == 0

    @pytest.mark.asyncio
    async def test_predict_batch_distributes_across_gpus(
        self, mock_torch_cuda, mock_transformers, mock_model, mock_tokenizer
    ):
        """Test batch prediction distributes across GPUs."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        # Set up mocked models for each GPU
        manager.models = {i: mock_model for i in range(4)}
        manager.tokenizers = {i: mock_tokenizer for i in range(4)}
        manager._is_loaded = True

        # Track which GPUs were called
        gpu_calls = []

        def mock_inference(gpu_id, sequences, include_embeddings, include_attention):
            gpu_calls.append(gpu_id)
            return [
                {
                    "sequence": seq,
                    "sequence_length": len(seq),
                    "embedding": [0.1] * 1280,
                    "embedding_dim": 1280,
                    "gpu_id": gpu_id,
                }
                for seq in sequences
            ]

        manager._run_inference_on_gpu = mock_inference

        sequences = [f"MKTVRQERL{i}" for i in range(8)]
        result = await manager.predict_batch(sequences=sequences)

        assert result.total_sequences == 8
        assert len(result.results) == 8
        # All 4 GPUs should have been used
        assert set(gpu_calls) == {0, 1, 2, 3}

    @pytest.mark.asyncio
    async def test_predict_batch_results_ordered_correctly(
        self, mock_torch_cuda, mock_transformers
    ):
        """Test batch results maintain original order."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        manager.models = {i: MagicMock() for i in range(4)}
        manager.tokenizers = {i: MagicMock() for i in range(4)}
        manager._is_loaded = True

        def mock_inference(gpu_id, sequences, include_embeddings, include_attention):
            # Return results that we can verify order
            return [
                {
                    "sequence": seq,
                    "sequence_length": len(seq),
                    "embedding": [float(i) for i in range(1280)],
                    "embedding_dim": 1280,
                    "gpu_id": gpu_id,
                }
                for seq in sequences
            ]

        manager._run_inference_on_gpu = mock_inference

        # Create sequences with identifiable content
        sequences = [f"SEQ{i}MKTVRQ" for i in range(10)]
        result = await manager.predict_batch(sequences=sequences)

        # Verify order is preserved
        for i, seq_result in enumerate(result.results):
            assert seq_result.sequence == f"SEQ{i}MKTVRQ"

    @pytest.mark.asyncio
    async def test_predict_batch_distribution_info(
        self, mock_torch_cuda_8gpu, mock_transformers
    ):
        """Test batch returns accurate distribution info."""
        from app.model_manager import GPUDistributor, ModelManager

        distributor = GPUDistributor()
        manager = ModelManager(
            model_name="facebook/esm2_t33_650M_UR50D", gpu_distributor=distributor
        )

        manager.models = {i: MagicMock() for i in range(8)}
        manager.tokenizers = {i: MagicMock() for i in range(8)}
        manager._is_loaded = True

        def mock_inference(gpu_id, sequences, include_embeddings, include_attention):
            return [
                {"sequence": seq, "sequence_length": len(seq), "gpu_id": gpu_id}
                for seq in sequences
            ]

        manager._run_inference_on_gpu = mock_inference

        sequences = ["MKTVRQ"] * 64
        result = await manager.predict_batch(sequences=sequences)

        # Check distribution info
        assert result.distribution_info.total_sequences == 64
        assert result.distribution_info.num_gpus_used == 8
        # 64 / 8 = 8 sequences per GPU
        for gpu_key in result.distribution_info.sequences_per_gpu:
            assert result.distribution_info.sequences_per_gpu[gpu_key] == 8
