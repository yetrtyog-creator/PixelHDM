"""
Shared pytest fixtures and utilities for PixelHDM-RPEA-DinoV3 tests.

This module provides:
- Device fixtures (CPU/CUDA)
- Config fixtures (minimal, small, default)
- Tensor fixtures (images, text embeddings, timesteps)
- Mock model fixtures
- Utility functions (tensor validation, shape checking, gradient flow)

Author: PixelHDM-RPEA-DinoV3 Project
Date: 2025-12-30
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pytest
import torch
import torch.nn as nn

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


# ============================================================================
# Custom Markers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


# ============================================================================
# Device Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return available device (CUDA if available, else CPU). Session-scoped for efficiency."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def cpu_device() -> torch.device:
    """Return CPU device. Session-scoped for efficiency."""
    return torch.device("cpu")


# Skip marker for CUDA-only tests
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available"
)


# ============================================================================
# Config Fixtures (session-scoped for efficiency)
# ============================================================================

@pytest.fixture(scope="session")
def minimal_config():
    """
    Minimal configuration for fast unit tests.

    Disables REPA and FreqLoss, uses smallest dimensions.
    Session-scoped to avoid recreating on every test.
    """
    from src.config import PixelHDMConfig
    return PixelHDMConfig.for_testing()


@pytest.fixture(scope="session")
def small_config():
    """
    Small configuration for integration tests.

    Uses reduced dimensions but keeps all features enabled.
    Session-scoped to avoid recreating on every test.
    """
    from src.config import PixelHDMConfig
    return PixelHDMConfig.small()


@pytest.fixture(scope="session")
def default_config():
    """
    Default configuration (full model).

    Used for testing real model behavior.
    Session-scoped to avoid recreating on every test.
    """
    from src.config import PixelHDMConfig
    return PixelHDMConfig.default()


@pytest.fixture(scope="session")
def training_config():
    """Training configuration fixture (session-scoped)."""
    from src.config import TrainingConfig
    return TrainingConfig()


@pytest.fixture(scope="session")
def data_config():
    """Data configuration fixture (session-scoped)."""
    from src.config import DataConfig
    return DataConfig()


@pytest.fixture(scope="session")
def full_config():
    """Complete configuration (model + training + data). Session-scoped."""
    from src.config import Config
    return Config.default()


@pytest.fixture(scope="session")
def test_full_config():
    """Complete configuration for testing (session-scoped)."""
    from src.config import Config
    return Config.for_testing()


# ============================================================================
# Tensor Dimension Fixtures (session-scoped since they're constants)
# ============================================================================

@pytest.fixture(scope="session")
def batch_size() -> int:
    """Default batch size for tests."""
    return 2


@pytest.fixture(scope="session")
def hidden_dim() -> int:
    """Hidden dimension for fast tests (reduced from 1024)."""
    return 256


@pytest.fixture(scope="session")
def text_len() -> int:
    """Default text sequence length."""
    return 32


@pytest.fixture(scope="session")
def image_size() -> int:
    """Default image size."""
    return 256


@pytest.fixture(scope="session")
def patch_size() -> int:
    """Default patch size (matches DINOv3)."""
    return 16


@pytest.fixture(scope="session")
def num_patches(image_size: int, patch_size: int) -> int:
    """Number of patches for given image size."""
    return (image_size // patch_size) ** 2


# ============================================================================
# Tensor Fixtures
# ============================================================================

@pytest.fixture
def dummy_images(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate random 256x256 images."""
    return torch.randn(batch_size, 3, 256, 256, device=device)


@pytest.fixture
def dummy_images_512(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate random 512x512 images."""
    return torch.randn(batch_size, 3, 512, 512, device=device)


@pytest.fixture
def dummy_images_64(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate random 64x64 images (for fast tests)."""
    return torch.randn(batch_size, 3, 64, 64, device=device)


@pytest.fixture
def dummy_images_rect(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate random rectangular 512x256 images."""
    return torch.randn(batch_size, 3, 256, 512, device=device)


@pytest.fixture
def dummy_text_embeddings(
    batch_size: int,
    hidden_dim: int,
    text_len: int,
    device: torch.device
) -> torch.Tensor:
    """Generate random text embeddings."""
    return torch.randn(batch_size, text_len, hidden_dim, device=device)


@pytest.fixture
def dummy_timesteps(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate random timesteps in [0, 1]."""
    return torch.rand(batch_size, device=device)


@pytest.fixture
def dummy_timesteps_uniform(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate uniformly spaced timesteps."""
    return torch.linspace(0.1, 0.9, batch_size, device=device)


@pytest.fixture
def dummy_noise(batch_size: int, device: torch.device) -> torch.Tensor:
    """Generate random noise for 256x256 images."""
    return torch.randn(batch_size, 3, 256, 256, device=device)


@pytest.fixture
def dummy_latents(
    batch_size: int,
    hidden_dim: int,
    num_patches: int,
    device: torch.device
) -> torch.Tensor:
    """Generate random latent features."""
    return torch.randn(batch_size, num_patches, hidden_dim, device=device)


@pytest.fixture
def dummy_pixel_features(
    batch_size: int,
    num_patches: int,
    device: torch.device
) -> torch.Tensor:
    """Generate random pixel features (p^2 * D_pix = 256 * 16 = 4096)."""
    # patch_size=16, pixel_dim=16 -> 16*16*16 = 4096
    pixel_feature_dim = 16 * 16 * 16  # p^2 * D_pix
    return torch.randn(batch_size, num_patches, pixel_feature_dim, device=device)


# ============================================================================
# Mock Model Fixtures
# ============================================================================

class DummyModel(nn.Module):
    """
    Minimal dummy model for testing samplers and losses.

    Accepts:
        x_t: (B, C, H, W) noisy image
        t: (B,) timesteps
        text_emb: (B, L, D) text embeddings (optional)

    Returns:
        Prediction of same shape as x_t
    """

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv2d(3, 3, kernel_size=1, bias=True)
        # Initialize to identity-like behavior
        nn.init.eye_(self.conv.weight[:, :, 0, 0])
        nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass returning prediction of same shape."""
        return self.conv(x_t)


class DummyModelWithFeatures(nn.Module):
    """
    Dummy model that also returns intermediate features (for REPA testing).
    """

    def __init__(self, hidden_dim: int = 256, num_patches: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches
        self.conv = nn.Conv2d(3, 3, kernel_size=1)
        self.feature_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        text_emb: Optional[torch.Tensor] = None,
        return_features: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Forward pass with optional feature return."""
        output = self.conv(x_t)

        if return_features:
            B = x_t.shape[0]
            # Generate dummy features
            features = torch.randn(B, self.num_patches, self.hidden_dim, device=x_t.device)
            return output, features

        return output


class DummyTextEncoder(nn.Module):
    """
    Minimal text encoder mock for testing.

    Returns random but consistent embeddings based on input hash.
    """

    def __init__(self, hidden_size: int = 256, max_length: int = 32):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embed = nn.Embedding(32000, hidden_size)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Encode input tokens."""
        embeddings = self.embed(input_ids)
        return {
            "last_hidden_state": embeddings,
            "pooler_output": embeddings.mean(dim=1),
        }


@pytest.fixture(scope="module")
def dummy_model(hidden_dim: int) -> DummyModel:
    """Create a dummy model for testing. Module-scoped since stateless."""
    model = DummyModel(hidden_dim)
    model.eval()
    return model


@pytest.fixture(scope="module")
def dummy_model_with_features(hidden_dim: int, num_patches: int) -> DummyModelWithFeatures:
    """Create a dummy model with feature extraction. Module-scoped since stateless."""
    model = DummyModelWithFeatures(hidden_dim, num_patches)
    model.eval()
    return model


@pytest.fixture(scope="module")
def dummy_text_encoder(hidden_dim: int) -> DummyTextEncoder:
    """Create a dummy text encoder. Module-scoped since stateless."""
    model = DummyTextEncoder(hidden_size=hidden_dim)
    model.eval()
    return model


# ============================================================================
# Temporary File Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_json_file(temp_dir: Path):
    """Create a temporary JSON file path."""
    return temp_dir / "test_config.json"


@pytest.fixture
def temp_checkpoint_dir(temp_dir: Path):
    """Create a temporary directory for checkpoints."""
    checkpoint_dir = temp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


# ============================================================================
# Utility Functions
# ============================================================================

def assert_tensor_valid(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Check that a tensor contains no NaN or Inf values.

    Args:
        tensor: Tensor to validate
        name: Name for error messages

    Raises:
        AssertionError: If tensor contains NaN or Inf
    """
    assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
    assert not torch.isinf(tensor).any(), f"{name} contains Inf values"


def assert_shapes_equal(
    tensor: torch.Tensor,
    expected: Tuple[int, ...],
    name: str = "tensor"
) -> None:
    """
    Check that tensor shape matches expected.

    Args:
        tensor: Tensor to check
        expected: Expected shape tuple
        name: Name for error messages

    Raises:
        AssertionError: If shapes don't match
    """
    assert tensor.shape == expected, \
        f"{name} shape {tensor.shape} != expected {expected}"


def assert_shapes_match(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    name1: str = "tensor1",
    name2: str = "tensor2"
) -> None:
    """
    Check that two tensors have matching shapes.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        name1: Name for first tensor in error messages
        name2: Name for second tensor in error messages

    Raises:
        AssertionError: If shapes don't match
    """
    assert tensor1.shape == tensor2.shape, \
        f"{name1} shape {tensor1.shape} != {name2} shape {tensor2.shape}"


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count parameters in a model.

    Args:
        model: PyTorch model
        trainable_only: If True, only count trainable parameters

    Returns:
        Number of parameters
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def assert_gradient_flow(model: nn.Module, loss: torch.Tensor) -> None:
    """
    Verify that gradients flow correctly through the model.

    Args:
        model: Model to check gradients for
        loss: Loss tensor to backpropagate

    Raises:
        AssertionError: If any trainable parameter has no gradient
    """
    # Clear existing gradients
    model.zero_grad()

    # Backpropagate
    loss.backward()

    # Check all trainable parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter '{name}' has no gradient"
            assert not torch.isnan(param.grad).any(), f"Parameter '{name}' gradient contains NaN"
            assert not torch.isinf(param.grad).any(), f"Parameter '{name}' gradient contains Inf"


def assert_loss_decreases(
    losses: list,
    name: str = "loss",
    min_decrease: float = 0.0
) -> None:
    """
    Check that loss values generally decrease (allows some fluctuation).

    Args:
        losses: List of loss values
        name: Name for error messages
        min_decrease: Minimum expected total decrease

    Raises:
        AssertionError: If losses don't decrease sufficiently
    """
    assert len(losses) >= 2, f"Need at least 2 {name} values"
    total_decrease = losses[0] - losses[-1]
    assert total_decrease >= min_decrease, \
        f"{name} did not decrease: first={losses[0]:.6f}, last={losses[-1]:.6f}, " \
        f"decrease={total_decrease:.6f} < min_decrease={min_decrease}"


def create_random_image_batch(
    batch_size: int,
    height: int,
    width: int,
    channels: int = 3,
    device: torch.device = torch.device("cpu"),
    normalized: bool = True,
) -> torch.Tensor:
    """
    Create a batch of random images.

    Args:
        batch_size: Number of images
        height: Image height
        width: Image width
        channels: Number of channels (default 3)
        device: Target device
        normalized: If True, values in [-1, 1]; else [0, 1]

    Returns:
        Tensor of shape (B, C, H, W)
    """
    images = torch.rand(batch_size, channels, height, width, device=device)
    if normalized:
        images = images * 2 - 1  # [0, 1] -> [-1, 1]
    return images


def create_deterministic_generator(seed: int = 42) -> torch.Generator:
    """
    Create a deterministic random generator for reproducible tests.

    Args:
        seed: Random seed

    Returns:
        PyTorch Generator
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


# ============================================================================
# Comparison Utilities
# ============================================================================

def tensors_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """
    Check if two tensors are element-wise close.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if tensors are close
    """
    return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)


def assert_tensors_close(
    tensor1: torch.Tensor,
    tensor2: torch.Tensor,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    name: str = "tensors",
) -> None:
    """
    Assert that two tensors are element-wise close.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for error messages

    Raises:
        AssertionError: If tensors are not close
    """
    if not torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
        diff = (tensor1 - tensor2).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        raise AssertionError(
            f"{name} are not close: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
            f"rtol={rtol}, atol={atol}"
        )


# ============================================================================
# Memory Utilities
# ============================================================================

def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage statistics.

    Returns:
        Dictionary with memory stats in MB
    """
    stats = {"cpu_mb": 0.0}

    if torch.cuda.is_available():
        stats["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        stats["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
        stats["cuda_max_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024 / 1024

    return stats


def reset_cuda_memory_stats() -> None:
    """Reset CUDA memory statistics."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


# ============================================================================
# Timing Utilities
# ============================================================================

class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = "operation"):
        self.name = name
        self.start_time = 0.0
        self.elapsed = 0.0

    def __enter__(self):
        import time
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        import time
        self.elapsed = time.perf_counter() - self.start_time


# ============================================================================
# Fixtures for Other Test Modules
# ============================================================================

@pytest.fixture
def seed() -> int:
    """Default random seed for reproducibility."""
    return 42


@pytest.fixture
def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
