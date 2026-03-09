"""Tests for pipeline.metrics module."""

import numpy as np
import pytest

from pipeline.metrics import compute_psnr, compute_ssim, evaluate_quality


def _make_frame(h: int = 64, w: int = 64, channels: int = 3, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, channels) if channels > 1 else (h, w),
                        dtype=np.uint8)


# ---------------------------------------------------------------------------
# compute_psnr
# ---------------------------------------------------------------------------

class TestComputePSNR:
    def test_identical_images_returns_inf(self):
        frame = _make_frame()
        result = compute_psnr(frame, frame)
        assert np.isinf(result) or result > 100

    def test_different_images_returns_finite_value(self):
        ref = _make_frame(seed=1)
        enh = _make_frame(seed=2)
        result = compute_psnr(ref, enh)
        assert np.isfinite(result)
        assert result > 0

    def test_higher_noise_lower_psnr(self):
        """Noisier reconstruction should yield lower PSNR."""
        rng = np.random.default_rng(42)
        ref = _make_frame(seed=10)
        low_noise = np.clip(ref.astype(np.int32) + rng.integers(-5, 5, ref.shape), 0, 255).astype(np.uint8)
        high_noise = np.clip(ref.astype(np.int32) + rng.integers(-30, 30, ref.shape), 0, 255).astype(np.uint8)
        assert compute_psnr(ref, low_noise) > compute_psnr(ref, high_noise)

    def test_returns_float(self):
        frame = _make_frame()
        assert isinstance(compute_psnr(frame, frame), float)

    def test_shape_mismatch_raises(self):
        a = _make_frame(h=32, w=32)
        b = _make_frame(h=64, w=64)
        with pytest.raises(ValueError, match="shape"):
            compute_psnr(a, b)

    def test_grayscale_images(self):
        ref = _make_frame(channels=1)
        enh = _make_frame(channels=1, seed=5)
        result = compute_psnr(ref, enh)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# compute_ssim
# ---------------------------------------------------------------------------

class TestComputeSSIM:
    def test_identical_images_return_one(self):
        frame = _make_frame()
        result = compute_ssim(frame, frame)
        assert abs(result - 1.0) < 1e-6

    def test_different_images_less_than_one(self):
        ref = _make_frame(seed=1)
        enh = _make_frame(seed=2)
        result = compute_ssim(ref, enh)
        assert result < 1.0

    def test_range_is_valid(self):
        ref = _make_frame(seed=3)
        enh = _make_frame(seed=4)
        result = compute_ssim(ref, enh)
        assert -1.0 <= result <= 1.0

    def test_higher_noise_lower_ssim(self):
        rng = np.random.default_rng(99)
        ref = _make_frame(seed=7)
        low_noise = np.clip(ref.astype(np.int32) + rng.integers(-5, 5, ref.shape), 0, 255).astype(np.uint8)
        high_noise = np.clip(ref.astype(np.int32) + rng.integers(-40, 40, ref.shape), 0, 255).astype(np.uint8)
        assert compute_ssim(ref, low_noise) > compute_ssim(ref, high_noise)

    def test_returns_float(self):
        frame = _make_frame()
        assert isinstance(compute_ssim(frame, frame), float)

    def test_shape_mismatch_raises(self):
        a = _make_frame(h=32, w=32)
        b = _make_frame(h=64, w=64)
        with pytest.raises(ValueError, match="shape"):
            compute_ssim(a, b)


# ---------------------------------------------------------------------------
# evaluate_quality
# ---------------------------------------------------------------------------

class TestEvaluateQuality:
    def test_returns_dict_with_psnr_and_ssim(self):
        frame = _make_frame()
        result = evaluate_quality(frame, frame)
        assert "psnr" in result
        assert "ssim" in result

    def test_identical_frames_optimal_metrics(self):
        frame = _make_frame()
        result = evaluate_quality(frame, frame)
        assert result["ssim"] > 0.99
        assert result["psnr"] > 100 or np.isinf(result["psnr"])

    def test_values_consistent_with_individual_functions(self):
        ref = _make_frame(seed=8)
        enh = _make_frame(seed=9)
        combined = evaluate_quality(ref, enh)
        assert abs(combined["psnr"] - compute_psnr(ref, enh)) < 1e-10
        assert abs(combined["ssim"] - compute_ssim(ref, enh)) < 1e-10
