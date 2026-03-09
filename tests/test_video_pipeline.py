"""Tests for pipeline.video_pipeline module."""

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch

from pipeline.denoising import DnCNN
from pipeline.preprocessing import FramePreprocessor
from pipeline.video_pipeline import VideoPipeline


CPU = torch.device("cpu")


def _write_test_video(path: str, num_frames: int = 5, h: int = 64, w: int = 64) -> None:
    """Write a small synthetic video to *path* for testing."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, 10.0, (w, h))
    rng = np.random.default_rng(42)
    for _ in range(num_frames):
        frame = rng.integers(0, 60, (h, w, 3), dtype=np.uint8)  # dark frames
        writer.write(frame)
    writer.release()


# ---------------------------------------------------------------------------
# VideoPipeline.process_frame
# ---------------------------------------------------------------------------

class TestProcessFrame:
    def test_output_shape_and_dtype(self):
        pipeline = VideoPipeline()
        frame = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_with_cnn_model(self):
        model = DnCNN(channels=3)
        pipeline = VideoPipeline(denoising_model=model, device=CPU)
        frame = np.random.default_rng(0).integers(10, 50, (32, 32, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8


# ---------------------------------------------------------------------------
# VideoPipeline.process_video
# ---------------------------------------------------------------------------

class TestProcessVideo:
    def test_process_video_returns_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            _write_test_video(video_path, num_frames=3)

            pipeline = VideoPipeline()
            summary = pipeline.process_video(input_path=video_path)

        assert "total_frames" in summary
        assert "avg_psnr" in summary
        assert "avg_ssim" in summary
        assert "frame_metrics" in summary
        assert summary["total_frames"] == 3

    def test_process_video_writes_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            output_path = os.path.join(tmpdir, "output.mp4")
            _write_test_video(video_path, num_frames=3)

            pipeline = VideoPipeline()
            pipeline.process_video(input_path=video_path, output_path=output_path)

            assert os.path.isfile(output_path)
            assert os.path.getsize(output_path) > 0

    def test_max_frames_limits_processing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            _write_test_video(video_path, num_frames=10)

            pipeline = VideoPipeline()
            summary = pipeline.process_video(input_path=video_path, max_frames=4)

        assert summary["total_frames"] == 4

    def test_nonexistent_input_raises(self):
        pipeline = VideoPipeline()
        with pytest.raises(FileNotFoundError):
            pipeline.process_video(input_path="/nonexistent/video.mp4")

    def test_metrics_with_reference(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "input.mp4")
            ref_path = os.path.join(tmpdir, "ref.mp4")
            _write_test_video(video_path, num_frames=3)
            _write_test_video(ref_path, num_frames=3)

            pipeline = VideoPipeline(compute_metrics=True)
            summary = pipeline.process_video(
                input_path=video_path, reference_path=ref_path
            )

        assert summary["avg_psnr"] is not None
        assert summary["avg_ssim"] is not None
        assert len(summary["frame_metrics"]) == 3

    def test_on_frame_callback(self):
        collected: list = []

        def cb(idx: int, frame: np.ndarray, metrics: dict) -> None:
            collected.append(idx)

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            _write_test_video(video_path, num_frames=4)

            pipeline = VideoPipeline()
            pipeline.process_video(input_path=video_path, on_frame=cb)

        assert collected == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# VideoPipeline.frame_generator
# ---------------------------------------------------------------------------

class TestFrameGenerator:
    def test_yields_correct_number_of_frames(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            _write_test_video(video_path, num_frames=5)

            pipeline = VideoPipeline()
            frames = list(pipeline.frame_generator(video_path))

        assert len(frames) == 5
        for idx, (i, frame) in enumerate(frames):
            assert i == idx
            assert frame.dtype == np.uint8

    def test_max_frames_generator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, "test.mp4")
            _write_test_video(video_path, num_frames=10)

            pipeline = VideoPipeline()
            frames = list(pipeline.frame_generator(video_path, max_frames=3))

        assert len(frames) == 3

    def test_generator_nonexistent_raises(self):
        pipeline = VideoPipeline()
        with pytest.raises(FileNotFoundError):
            list(pipeline.frame_generator("/no/such/file.mp4"))
