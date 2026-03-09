"""End-to-end video frame enhancement pipeline."""

from __future__ import annotations

import os
from typing import Callable, Generator, Iterator

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .denoising import DnCNN
from .metrics import evaluate_quality
from .preprocessing import FramePreprocessor


class VideoPipeline:
    """Frame-level enhancement pipeline for continuous video streams.

    The pipeline applies the following steps to every frame extracted
    from a video file:

    1. **Preprocessing** – CLAHE contrast normalisation via
       :class:`~pipeline.preprocessing.FramePreprocessor`.
    2. **CNN denoising** – optional deep CNN denoising via
       :class:`~pipeline.denoising.DnCNN`.
    3. **Metrics evaluation** – optional per-frame PSNR / SSIM
       computation when a reference video is available.

    Args:
        preprocessor: :class:`FramePreprocessor` instance.  A default
            instance is created when *None*.
        denoising_model: Optional :class:`DnCNN` model.  When provided
            each preprocessed frame is further denoised by the CNN.
        device: Torch device used for CNN inference.  Auto-selected when
            *None*.
        compute_metrics: When ``True`` the pipeline computes per-frame
            PSNR and SSIM and includes them in the output.
    """

    def __init__(
        self,
        preprocessor: FramePreprocessor | None = None,
        denoising_model: DnCNN | None = None,
        device: torch.device | None = None,
        compute_metrics: bool = False,
    ) -> None:
        self.preprocessor = preprocessor or FramePreprocessor()
        self.denoising_model = denoising_model
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.compute_metrics = compute_metrics

        if self.denoising_model is not None:
            self.denoising_model.to(self.device)
            self.denoising_model.eval()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_video(
        self,
        input_path: str,
        output_path: str | None = None,
        reference_path: str | None = None,
        max_frames: int | None = None,
        on_frame: Callable[[int, np.ndarray, dict], None] | None = None,
    ) -> dict:
        """Process all frames of a video file and write results.

        Args:
            input_path: Path to the input video file.
            output_path: Optional path to write the enhanced video.
                When *None* frames are processed but not written to
                disk.
            reference_path: Optional path to a reference (clean) video
                used to compute PSNR / SSIM.  Only used when
                ``compute_metrics`` is ``True``.
            max_frames: Maximum number of frames to process.  Useful for
                quick evaluation on large datasets.
            on_frame: Optional callback invoked after each frame with
                signature ``(frame_index, enhanced_frame, frame_metrics)``.

        Returns:
            Summary dictionary with keys:

            - ``"total_frames"`` – number of frames processed.
            - ``"avg_psnr"`` – average PSNR over all frames
              (``None`` if metrics disabled or no reference).
            - ``"avg_ssim"`` – average SSIM (same caveat).
            - ``"frame_metrics"`` – list of per-frame metric dicts
              (empty when metrics disabled).

        Raises:
            FileNotFoundError: If *input_path* does not exist.
            RuntimeError: If OpenCV cannot open the input video.
        """
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")

        cap_ref: cv2.VideoCapture | None = None
        if reference_path is not None and self.compute_metrics:
            if not os.path.isfile(reference_path):
                raise FileNotFoundError(f"Reference video not found: {reference_path}")
            cap_ref = cv2.VideoCapture(reference_path)
            if not cap_ref.isOpened():
                raise RuntimeError(f"Cannot open reference video: {reference_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer: cv2.VideoWriter | None = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_metrics: list[dict] = []
        frame_idx = 0

        try:
            progress = tqdm(total=total if max_frames is None else max_frames,
                            desc="Processing frames", unit="frame")
            while True:
                if max_frames is not None and frame_idx >= max_frames:
                    break

                ok, frame = cap.read()
                if not ok:
                    break

                enhanced = self._enhance_frame(frame)

                metrics: dict = {}
                if self.compute_metrics and cap_ref is not None:
                    ok_ref, ref_frame = cap_ref.read()
                    if ok_ref:
                        metrics = evaluate_quality(ref_frame, enhanced)
                    frame_metrics.append(metrics)

                if writer is not None:
                    writer.write(enhanced)

                if on_frame is not None:
                    on_frame(frame_idx, enhanced, metrics)

                frame_idx += 1
                progress.update(1)

            progress.close()
        finally:
            cap.release()
            if cap_ref is not None:
                cap_ref.release()
            if writer is not None:
                writer.release()

        avg_psnr: float | None = None
        avg_ssim: float | None = None
        if frame_metrics:
            psnr_vals = [m["psnr"] for m in frame_metrics if "psnr" in m]
            ssim_vals = [m["ssim"] for m in frame_metrics if "ssim" in m]
            if psnr_vals:
                avg_psnr = float(np.mean(psnr_vals))
            if ssim_vals:
                avg_ssim = float(np.mean(ssim_vals))

        return {
            "total_frames": frame_idx,
            "avg_psnr": avg_psnr,
            "avg_ssim": avg_ssim,
            "frame_metrics": frame_metrics,
        }

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Enhance a single BGR frame.

        A convenience wrapper around :meth:`_enhance_frame` for use
        outside the video I/O loop (e.g. in unit tests or notebooks).

        Args:
            frame: Input BGR ``uint8`` frame (H, W, 3).

        Returns:
            Enhanced BGR ``uint8`` frame.
        """
        return self._enhance_frame(frame)

    def frame_generator(
        self, video_path: str, max_frames: int | None = None
    ) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield ``(index, enhanced_frame)`` tuples for streaming use.

        Designed for large video datasets where loading all frames into
        memory at once is impractical.

        Args:
            video_path: Path to the input video file.
            max_frames: Stop after this many frames when set.

        Yields:
            Tuple of ``(frame_index, enhanced_bgr_frame)``.

        Raises:
            FileNotFoundError: If *video_path* does not exist.
            RuntimeError: If OpenCV cannot open the video.
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        idx = 0
        try:
            while True:
                if max_frames is not None and idx >= max_frames:
                    break
                ok, frame = cap.read()
                if not ok:
                    break
                yield idx, self._enhance_frame(frame)
                idx += 1
        finally:
            cap.release()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run preprocessing and optional CNN denoising on one frame."""
        enhanced = self.preprocessor.preprocess(frame)
        if self.denoising_model is not None:
            enhanced = self.denoising_model.denoise_frame(enhanced, device=self.device)
        return enhanced
