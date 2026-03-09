"""Tests for pipeline.denoising module."""

import numpy as np
import pytest
import torch

from pipeline.denoising import DnCNN, load_denoising_model


def _make_noisy(h: int = 32, w: int = 32, channels: int = 3, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    frame = rng.integers(50, 200, (h, w, channels) if channels > 1 else (h, w),
                         dtype=np.uint8)
    return frame


CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# DnCNN model architecture
# ---------------------------------------------------------------------------

class TestDnCNNArchitecture:
    def test_forward_shape_unchanged(self):
        model = DnCNN(channels=1)
        x = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_forward_rgb(self):
        model = DnCNN(channels=3)
        x = torch.rand(2, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_output_clamped_to_unit_range(self):
        model = DnCNN(channels=1)
        x = torch.rand(1, 1, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_custom_depth_and_features(self):
        model = DnCNN(channels=1, num_features=32, depth=10)
        x = torch.rand(1, 1, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# DnCNN.denoise_frame
# ---------------------------------------------------------------------------

class TestDenoiseFrame:
    def test_output_shape_and_dtype_rgb(self):
        model = DnCNN(channels=3)
        frame = _make_noisy(channels=3)
        result = model.denoise_frame(frame, device=CPU)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_output_shape_and_dtype_grayscale(self):
        model = DnCNN(channels=1)
        frame = _make_noisy(channels=1)
        result = model.denoise_frame(frame, device=CPU)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_output_values_in_range(self):
        model = DnCNN(channels=3)
        frame = _make_noisy(channels=3)
        result = model.denoise_frame(frame, device=CPU)
        assert result.min() >= 0
        assert result.max() <= 255


# ---------------------------------------------------------------------------
# load_denoising_model
# ---------------------------------------------------------------------------

class TestLoadDenoisingModel:
    def test_returns_dncnn_instance(self):
        model = load_denoising_model(device=CPU)
        assert isinstance(model, DnCNN)

    def test_model_in_eval_mode(self):
        model = load_denoising_model(device=CPU)
        assert not model.training

    def test_custom_params(self):
        model = load_denoising_model(channels=3, num_features=32, depth=10, device=CPU)
        x = torch.rand(1, 3, 16, 16)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_invalid_checkpoint_raises(self):
        with pytest.raises((FileNotFoundError, RuntimeError, Exception)):
            load_denoising_model(checkpoint_path="/nonexistent/path.pth", device=CPU)
