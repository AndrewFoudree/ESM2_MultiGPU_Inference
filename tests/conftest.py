"""
Pytest configuration and shared fixtures for ESM-2 Multi-GPU tests.

Provides mocked GPU operations for testing without actual hardware.
"""

import asyncio
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ============================================================================
# GPU Mocking Utilities
# ============================================================================


class MockTensor:
    """Mock PyTorch tensor for testing."""

    def __init__(self, data: List[float] = None, shape: tuple = None):
        self._data = data or [0.1] * 1280  # Default ESM-2 embedding dim
        self._shape = shape or (len(self._data),)

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        import numpy as np

        return np.array(self._data).reshape(self._shape)

    def tolist(self):
        return self._data

    def mean(self, dim=None):
        import numpy as np

        arr = np.array(self._data)
        if dim is not None:
            return MockTensor(arr.mean(axis=dim).tolist())
        return MockTensor([float(arr.mean())])

    def sum(self):
        return MockTensor([sum(self._data)])

    def item(self):
        return self._data[0] if len(self._data) == 1 else sum(self._data)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            # Handle multi-dimensional indexing
            return MockTensor(self._data[:10])
        return MockTensor(
            [self._data[key]] if isinstance(key, int) else self._data[key]
        )

    @property
    def shape(self):
        return self._shape


class MockModelOutput:
    """Mock transformer model output."""

    def __init__(self, batch_size: int = 1, seq_len: int = 28, hidden_dim: int = 1280):
        self.last_hidden_state = MockTensor(
            [0.1] * (batch_size * seq_len * hidden_dim),
            shape=(batch_size, seq_len, hidden_dim),
        )
        # Mock attention: (batch, heads, seq_len, seq_len)
        self.attentions = [
            MockTensor(
                [0.1] * (batch_size * 20 * seq_len * seq_len),
                shape=(batch_size, 20, seq_len, seq_len),
            )
        ]


class MockTokenizerOutput:
    """Mock tokenizer output."""

    def __init__(self, batch_size: int = 1, seq_len: int = 28):
        self.input_ids = MockTensor(
            [1] * (batch_size * seq_len), shape=(batch_size, seq_len)
        )
        self.attention_mask = MockTensor(
            [1] * (batch_size * seq_len), shape=(batch_size, seq_len)
        )

    def __getitem__(self, key):
        if key == "input_ids":
            return self.input_ids
        elif key == "attention_mask":
            return self.attention_mask
        raise KeyError(key)

    def items(self):
        return [("input_ids", self.input_ids), ("attention_mask", self.attention_mask)]

    def keys(self):
        return ["input_ids", "attention_mask"]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_torch_cuda():
    """Mock PyTorch CUDA functions for GPU testing."""
    with (
        patch("torch.cuda.is_available") as mock_available,
        patch("torch.cuda.device_count") as mock_count,
        patch("torch.cuda.get_device_name") as mock_name,
        patch("torch.cuda.get_device_properties") as mock_props,
        patch("torch.cuda.empty_cache") as mock_cache,
    ):

        mock_available.return_value = True
        mock_count.return_value = 4  # Simulate 4 GPUs
        mock_name.return_value = "NVIDIA A10G"

        # Mock device properties
        mock_prop = MagicMock()
        mock_prop.total_memory = 24 * 1e9  # 24GB
        mock_props.return_value = mock_prop

        yield {
            "available": mock_available,
            "count": mock_count,
            "name": mock_name,
            "properties": mock_props,
            "empty_cache": mock_cache,
        }


@pytest.fixture
def mock_torch_cuda_8gpu():
    """Mock PyTorch CUDA functions for 8-GPU testing."""
    with (
        patch("torch.cuda.is_available") as mock_available,
        patch("torch.cuda.device_count") as mock_count,
        patch("torch.cuda.get_device_name") as mock_name,
        patch("torch.cuda.get_device_properties") as mock_props,
        patch("torch.cuda.empty_cache") as mock_cache,
    ):

        mock_available.return_value = True
        mock_count.return_value = 8  # Simulate 8 GPUs
        mock_name.return_value = "NVIDIA A100"

        mock_prop = MagicMock()
        mock_prop.total_memory = 80 * 1e9  # 80GB
        mock_props.return_value = mock_prop

        yield {
            "available": mock_available,
            "count": mock_count,
            "name": mock_name,
            "properties": mock_props,
            "empty_cache": mock_cache,
        }


@pytest.fixture
def mock_torch_no_cuda():
    """Mock PyTorch for CPU-only testing."""
    with (
        patch("torch.cuda.is_available") as mock_available,
        patch("torch.cuda.device_count") as mock_count,
    ):

        mock_available.return_value = False
        mock_count.return_value = 0

        yield {"available": mock_available, "count": mock_count}


@pytest.fixture
def mock_model():
    """Create a mock ESM-2 model."""
    model = MagicMock()
    model.eval = MagicMock(return_value=model)
    model.to = MagicMock(return_value=model)

    def mock_forward(*args, **kwargs):
        batch_size = 1
        if "input_ids" in kwargs:
            batch_size = (
                kwargs["input_ids"].shape[0]
                if hasattr(kwargs["input_ids"], "shape")
                else 1
            )
        return MockModelOutput(batch_size=batch_size)

    model.__call__ = mock_forward
    model.return_value = MockModelOutput()

    return model


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer."""
    tokenizer = MagicMock()

    def mock_tokenize(sequences, *args, **kwargs):
        if isinstance(sequences, str):
            return MockTokenizerOutput(batch_size=1, seq_len=len(sequences))
        return MockTokenizerOutput(batch_size=len(sequences), seq_len=30)

    tokenizer.__call__ = mock_tokenize
    tokenizer.return_value = MockTokenizerOutput()

    return tokenizer


@pytest.fixture
def mock_transformers(mock_model, mock_tokenizer):
    """Mock HuggingFace transformers loading."""
    with (
        patch("transformers.AutoModel.from_pretrained") as mock_auto_model,
        patch("transformers.AutoTokenizer.from_pretrained") as mock_auto_tokenizer,
    ):

        mock_auto_model.return_value = mock_model
        mock_auto_tokenizer.return_value = mock_tokenizer

        yield {"model": mock_auto_model, "tokenizer": mock_auto_tokenizer}


@pytest.fixture
def sample_sequences() -> List[str]:
    """Sample protein sequences for testing."""
    return [
        "MKTVRQERLKSIVRILERSKEPVSGAQL",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQ",
        "MSGSHHHHHHSSGLVPRGSH",
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLL",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERM",
        "GLSDGEWQQVLNVWGKVEADIPGHGQEVLIRLFK",
        "MHSSIVLATVLFVAIASASKTRELCMKSLEHAKVG",
        "DYKDDDDKGSENLYFQSMGSHHHHHHSSGLVPRGS",
    ]


@pytest.fixture
def sample_batch_64() -> List[str]:
    """64 sample sequences for max batch testing."""
    base_sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQL",
        "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQ",
        "MSGSHHHHHHSSGLVPRGSH",
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLL",
    ]
    # Repeat to get 64 sequences
    return (base_sequences * 16)[:64]


@pytest.fixture
def test_client():
    """Create FastAPI test client with mocked dependencies."""
    # Import here to avoid circular imports
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.device_count", return_value=0),
    ):

        from app.main import app

        # We need to skip the lifespan for basic route testing
        # For full integration tests, use async test client
        client = TestClient(app, raise_server_exceptions=False)
        yield client


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# ============================================================================
# Test Utilities
# ============================================================================


def assert_valid_embedding(embedding: List[float], expected_dim: int = 1280):
    """Assert that an embedding is valid."""
    assert embedding is not None
    assert isinstance(embedding, list)
    assert len(embedding) == expected_dim
    assert all(isinstance(x, (int, float)) for x in embedding)


def assert_valid_health_response(response: Dict[str, Any]):
    """Assert health response is valid."""
    assert "status" in response
    assert "gpu_count" in response
    assert "gpu_ids" in response
    assert "model_loaded" in response
    assert "model_name" in response
    assert "message" in response
