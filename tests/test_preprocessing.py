"""Tests for pipeline.preprocessing module."""

import numpy as np
import pytest

from pipeline.preprocessing import FramePreprocessor


def _make_frame(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (h, w, 3), dtype=np.uint8)


def _make_dark_frame(h: int = 64, w: int = 64, seed: int = 0) -> np.ndarray:
    """Return a low-light (dark) frame with pixel values mostly below 60."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 60, (h, w, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# FramePreprocessor.preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_shape_preserved(self):
        frame = _make_frame()
        result = FramePreprocessor().preprocess(frame)
        assert result.shape == frame.shape

    def test_output_dtype_uint8(self):
        frame = _make_frame()
        result = FramePreprocessor().preprocess(frame)
        assert result.dtype == np.uint8

    def test_dark_frame_brightness_increases(self):
        """Preprocessed low-light frame should be brighter on average."""
        dark = _make_dark_frame()
        result = FramePreprocessor().preprocess(dark)
        assert result.mean() > dark.mean(), (
            "CLAHE should increase mean brightness of a dark frame"
        )

    def test_returns_copy_not_in_place(self):
        frame = _make_frame()
        original = frame.copy()
        FramePreprocessor().preprocess(frame)
        np.testing.assert_array_equal(frame, original)

    def test_different_clip_limits(self):
        frame = _make_dark_frame()
        result_low = FramePreprocessor(clip_limit=1.0).preprocess(frame)
        result_high = FramePreprocessor(clip_limit=4.0).preprocess(frame)
        # higher clip limit → stronger contrast enhancement → higher mean
        assert result_high.mean() >= result_low.mean()


# ---------------------------------------------------------------------------
# FramePreprocessor.normalize_contrast
# ---------------------------------------------------------------------------

class TestNormalizeContrast:
    def test_output_shape_and_dtype(self):
        frame = _make_frame()
        result = FramePreprocessor().normalize_contrast(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_dark_frame_enhanced(self):
        dark = _make_dark_frame()
        result = FramePreprocessor().normalize_contrast(dark)
        assert result.mean() > dark.mean()


# ---------------------------------------------------------------------------
# FramePreprocessor.reduce_noise
# ---------------------------------------------------------------------------

class TestReduceNoise:
    def test_output_shape_and_dtype(self):
        frame = _make_frame()
        result = FramePreprocessor().reduce_noise(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_wrong_type_raises(self):
        fp = FramePreprocessor()
        with pytest.raises(ValueError, match="numpy.ndarray"):
            fp.preprocess([[1, 2, 3]])  # type: ignore[arg-type]

    def test_wrong_channels_raises(self):
        fp = FramePreprocessor()
        gray = np.zeros((64, 64), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            fp.preprocess(gray)

    def test_wrong_dtype_raises(self):
        fp = FramePreprocessor()
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        with pytest.raises(ValueError, match="dtype"):
            fp.preprocess(frame)
