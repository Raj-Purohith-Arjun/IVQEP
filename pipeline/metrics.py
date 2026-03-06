"""Image quality evaluation metrics: PSNR and SSIM."""

from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(
    reference: np.ndarray,
    enhanced: np.ndarray,
    data_range: float = 255.0,
) -> float:
    """Compute Peak Signal-to-Noise Ratio (PSNR) between two images.

    Higher values indicate better reconstruction quality.  A PSNR above
    30 dB is generally considered acceptable; above 40 dB is excellent.

    Args:
        reference: Ground-truth image as a ``uint8`` or ``float32``
            NumPy array.  Can be grayscale ``(H, W)`` or colour
            ``(H, W, C)``.
        enhanced: Reconstructed/enhanced image with the same shape and
            dtype as *reference*.
        data_range: The data range of the images (difference between
            maximum and minimum values).  Defaults to ``255.0`` for
            ``uint8`` images.

    Returns:
        PSNR value in decibels (dB) as a Python ``float``.

    Raises:
        ValueError: If *reference* and *enhanced* have different shapes.
    """
    _check_shapes(reference, enhanced)
    ref = reference.astype(np.float64)
    enh = enhanced.astype(np.float64)
    return float(peak_signal_noise_ratio(ref, enh, data_range=data_range))


def compute_ssim(
    reference: np.ndarray,
    enhanced: np.ndarray,
    data_range: float = 255.0,
) -> float:
    """Compute Structural Similarity Index (SSIM) between two images.

    SSIM ranges from ``-1`` to ``1``, where ``1`` indicates identical
    images.  Values above ``0.9`` are typically considered high quality.

    Args:
        reference: Ground-truth image as a ``uint8`` or ``float32``
            NumPy array.  Grayscale ``(H, W)`` or colour ``(H, W, C)``.
        enhanced: Reconstructed/enhanced image with the same shape.
        data_range: Data range of the input images.  Defaults to
            ``255.0`` for ``uint8`` images.

    Returns:
        SSIM score as a Python ``float`` in ``[-1, 1]``.

    Raises:
        ValueError: If *reference* and *enhanced* have different shapes.
    """
    _check_shapes(reference, enhanced)
    ref = reference.astype(np.float64)
    enh = enhanced.astype(np.float64)
    multichannel = ref.ndim == 3
    kwargs: dict = {"data_range": data_range}
    if multichannel:
        kwargs["channel_axis"] = -1
    return float(structural_similarity(ref, enh, **kwargs))


def evaluate_quality(
    reference: np.ndarray,
    enhanced: np.ndarray,
    data_range: float = 255.0,
) -> dict[str, float]:
    """Compute both PSNR and SSIM in a single call.

    Args:
        reference: Ground-truth image.
        enhanced: Reconstructed/enhanced image.
        data_range: Data range of images.  Defaults to ``255.0``.

    Returns:
        Dictionary with keys ``"psnr"`` and ``"ssim"`` and their
        respective metric values.

    Example::

        metrics = evaluate_quality(original_frame, denoised_frame)
        print(f"PSNR: {metrics['psnr']:.2f} dB, SSIM: {metrics['ssim']:.4f}")
    """
    return {
        "psnr": compute_psnr(reference, enhanced, data_range=data_range),
        "ssim": compute_ssim(reference, enhanced, data_range=data_range),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_shapes(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape != b.shape:
        raise ValueError(
            f"reference and enhanced must have the same shape; "
            f"got {a.shape} vs {b.shape}"
        )
