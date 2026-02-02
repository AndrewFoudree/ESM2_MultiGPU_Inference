"""
Tests for FastAPI endpoints.

Tests the /health, /predict, and /predict/batch endpoints.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


class TestHealthEndpoint:
    """Test suite for /health endpoint."""

    def test_health_initializing_state(self):
        """Test health returns initializing when model not loaded."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.device_count", return_value=0),
        ):

            from app import main as main_module
            from app.main import app

            # Temporarily set model_manager to None
            original_manager = main_module.model_manager
            main_module.model_manager = None

            try:
                client = TestClient(app, raise_server_exceptions=False)
                response = client.get("/health")

                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "initializing"
                assert data["model_loaded"] is False
            finally:
                main_module.model_manager = original_manager

    @pytest.mark.asyncio
    async def test_health_with_4_gpus(self, mock_torch_cuda, mock_transformers):
        """Test health endpoint reports 4 GPUs correctly."""
        from app import main as main_module
        from app.main import app
        from app.model_manager import GPUDistributor

        # Set up mock model manager
        distributor = GPUDistributor()
        manager = MagicMock()
        manager.is_loaded = True

        original_manager = main_module.model_manager
        original_distributor = main_module.gpu_distributor
        main_module.model_manager = manager
        main_module.gpu_distributor = distributor

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["gpu_count"] == 4
            assert data["gpu_ids"] == [0, 1, 2, 3]
            assert data["model_loaded"] is True
        finally:
            main_module.model_manager = original_manager
            main_module.gpu_distributor = original_distributor

    @pytest.mark.asyncio
    async def test_health_with_8_gpus(self, mock_torch_cuda_8gpu, mock_transformers):
        """Test health endpoint reports 8 GPUs correctly."""
        from app import main as main_module
        from app.main import app
        from app.model_manager import GPUDistributor

        distributor = GPUDistributor()
        manager = MagicMock()
        manager.is_loaded = True

        original_manager = main_module.model_manager
        original_distributor = main_module.gpu_distributor
        main_module.model_manager = manager
        main_module.gpu_distributor = distributor

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["gpu_count"] == 8
            assert data["gpu_ids"] == list(range(8))
        finally:
            main_module.model_manager = original_manager
            main_module.gpu_distributor = original_distributor


class TestPredictEndpoint:
    """Test suite for /predict endpoint."""

    @pytest.mark.asyncio
    async def test_predict_single_sequence(self, mock_torch_no_cuda, mock_transformers):
        """Test single sequence prediction."""
        from app import main as main_module
        from app.main import app
        from app.schemas import PredictionResponse

        # Create mock response
        mock_response = PredictionResponse(
            sequence="MKTVRQERLKSIVRILERSKEPVSGAQL",
            sequence_length=28,
            embedding=[0.1] * 1280,
            embedding_dim=1280,
            gpu_id=0,
            inference_time_ms=50.0,
            model_name="facebook/esm2_t33_650M_UR50D",
        )

        # Set up mocks
        manager = MagicMock()
        manager.is_loaded = True
        manager.predict_single = AsyncMock(return_value=mock_response)

        distributor = MagicMock()
        distributor.gpu_count = 0
        distributor.gpu_ids = []

        original_manager = main_module.model_manager
        original_distributor = main_module.gpu_distributor
        main_module.model_manager = manager
        main_module.gpu_distributor = distributor

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/predict",
                    json={
                        "sequence": "MKTVRQERLKSIVRILERSKEPVSGAQL",
                        "include_embeddings": True,
                        "include_attention": False,
                    },
                )

            assert response.status_code == 200
            data = response.json()
            assert data["sequence"] == "MKTVRQERLKSIVRILERSKEPVSGAQL"
            assert data["sequence_length"] == 28
            assert len(data["embedding"]) == 1280
        finally:
            main_module.model_manager = original_manager
            main_module.gpu_distributor = original_distributor

    @pytest.mark.asyncio
    async def test_predict_model_not_loaded(self, mock_torch_no_cuda):
        """Test predict returns 503 when model not loaded."""
        from app import main as main_module
        from app.main import app

        original_manager = main_module.model_manager
        main_module.model_manager = None

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/predict", json={"sequence": "MKTVRQERLKSIVRILERSKEPVSGAQL"}
                )

            assert response.status_code == 503
            assert "not loaded" in response.json()["detail"].lower()
        finally:
            main_module.model_manager = original_manager

    @pytest.mark.asyncio
    async def test_predict_invalid_sequence(self, mock_torch_no_cuda):
        """Test predict with invalid characters."""
        from app import main as main_module
        from app.main import app

        manager = MagicMock()
        manager.is_loaded = True

        original_manager = main_module.model_manager
        main_module.model_manager = manager

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/predict", json={"sequence": "INVALID123!@#"}
                )

            assert response.status_code == 422  # Validation error
        finally:
            main_module.model_manager = original_manager


class TestBatchPredictEndpoint:
    """Test suite for /predict/batch endpoint."""

    @pytest.mark.asyncio
    async def test_batch_predict_success(self, mock_torch_cuda, mock_transformers):
        """Test batch prediction with multiple sequences."""
        from app import main as main_module
        from app.main import app
        from app.model_manager import GPUDistributor
        from app.schemas import (
            BatchPredictionResponse,
            GPUDistributionInfo,
            SequenceResult,
        )

        sequences = [
            "MKTVRQERLKSIVRILERSKEPVSGAQL",
            "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQ",
            "MSGSHHHHHHSSGLVPRGSH",
        ]

        # Create mock response
        mock_results = [
            SequenceResult(
                sequence=seq,
                sequence_length=len(seq),
                embedding=[0.1] * 1280,
                embedding_dim=1280,
                gpu_id=i % 4,
            )
            for i, seq in enumerate(sequences)
        ]

        mock_distribution = GPUDistributionInfo(
            total_sequences=3,
            num_gpus_used=3,
            sequences_per_gpu={"gpu_0": 1, "gpu_1": 1, "gpu_2": 1},
            gpu_assignments={"gpu_0": [0], "gpu_1": [1], "gpu_2": [2]},
        )

        mock_response = BatchPredictionResponse(
            results=mock_results,
            total_sequences=3,
            total_inference_time_ms=100.0,
            average_time_per_sequence_ms=33.33,
            distribution_info=mock_distribution,
            model_name="facebook/esm2_t33_650M_UR50D",
        )

        manager = MagicMock()
        manager.is_loaded = True
        manager.predict_batch = AsyncMock(return_value=mock_response)

        distributor = GPUDistributor()

        original_manager = main_module.model_manager
        original_distributor = main_module.gpu_distributor
        main_module.model_manager = manager
        main_module.gpu_distributor = distributor

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/predict/batch", json={"sequences": sequences}
                )

            assert response.status_code == 200
            data = response.json()
            assert data["total_sequences"] == 3
            assert len(data["results"]) == 3
            assert "distribution_info" in data
        finally:
            main_module.model_manager = original_manager
            main_module.gpu_distributor = original_distributor

    @pytest.mark.asyncio
    async def test_batch_predict_max_size(self, mock_torch_cuda_8gpu):
        """Test batch prediction at max size (64)."""
        from app import main as main_module
        from app.main import app
        from app.schemas import (
            BatchPredictionResponse,
            GPUDistributionInfo,
            SequenceResult,
        )

        sequences = ["MKTVRQERLKSIVRILERSKEPVSGAQL"] * 64

        mock_results = [
            SequenceResult(
                sequence=seq,
                sequence_length=len(seq),
                embedding=[0.1] * 1280,
                embedding_dim=1280,
                gpu_id=i % 8,
            )
            for i, seq in enumerate(sequences)
        ]

        mock_distribution = GPUDistributionInfo(
            total_sequences=64,
            num_gpus_used=8,
            sequences_per_gpu={f"gpu_{i}": 8 for i in range(8)},
            gpu_assignments={
                f"gpu_{i}": list(range(i * 8, (i + 1) * 8)) for i in range(8)
            },
        )

        mock_response = BatchPredictionResponse(
            results=mock_results,
            total_sequences=64,
            total_inference_time_ms=500.0,
            average_time_per_sequence_ms=7.8,
            distribution_info=mock_distribution,
            model_name="facebook/esm2_t33_650M_UR50D",
        )

        manager = MagicMock()
        manager.is_loaded = True
        manager.predict_batch = AsyncMock(return_value=mock_response)

        distributor = MagicMock()
        distributor.gpu_count = 8

        original_manager = main_module.model_manager
        original_distributor = main_module.gpu_distributor
        main_module.model_manager = manager
        main_module.gpu_distributor = distributor

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/predict/batch", json={"sequences": sequences}
                )

            assert response.status_code == 200
            data = response.json()
            assert data["total_sequences"] == 64
        finally:
            main_module.model_manager = original_manager
            main_module.gpu_distributor = original_distributor

    @pytest.mark.asyncio
    async def test_batch_predict_exceeds_max(self, mock_torch_no_cuda):
        """Test batch prediction fails when exceeding 64 sequences."""
        from app import main as main_module
        from app.main import app

        manager = MagicMock()
        manager.is_loaded = True

        distributor = MagicMock()
        distributor.gpu_count = 0

        original_manager = main_module.model_manager
        original_distributor = main_module.gpu_distributor
        main_module.model_manager = manager
        main_module.gpu_distributor = distributor

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post(
                    "/predict/batch", json={"sequences": ["MKTVRQ"] * 65}
                )

            # Should fail at validation level (max 64)
            assert response.status_code == 422
        finally:
            main_module.model_manager = original_manager
            main_module.gpu_distributor = original_distributor

    @pytest.mark.asyncio
    async def test_batch_predict_empty_batch(self, mock_torch_no_cuda):
        """Test batch prediction fails with empty batch."""
        from app import main as main_module
        from app.main import app

        manager = MagicMock()
        manager.is_loaded = True

        original_manager = main_module.model_manager
        main_module.model_manager = manager

        try:
            transport = ASGITransport(app=app)
            async with AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                response = await client.post("/predict/batch", json={"sequences": []})

            assert response.status_code == 422
        finally:
            main_module.model_manager = original_manager


class TestRootEndpoint:
    """Test suite for / endpoint."""

    def test_root_returns_info(self):
        """Test root endpoint returns service info."""
        with (
            patch("torch.cuda.is_available", return_value=False),
            patch("torch.cuda.device_count", return_value=0),
        ):

            from app.main import app

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/")

            assert response.status_code == 200
            data = response.json()
            assert "service" in data
            assert "version" in data
            assert data["docs"] == "/docs"
