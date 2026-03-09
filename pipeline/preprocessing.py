"""OpenCV-based frame preprocessing for low-light video enhancement."""

import cv2
import numpy as np


class FramePreprocessor:
    """Preprocesses individual video frames for quality enhancement.

    Applies contrast normalization (CLAHE) and noise reduction using
    OpenCV to prepare low-light frames for downstream CNN denoising.

    Args:
        clip_limit: CLAHE contrast clipping limit. Higher values give
            stronger contrast enhancement. Default is 2.0.
        tile_grid_size: Grid size used for CLAHE adaptive histogram
            equalization. Default is (8, 8).
        denoise_h: Filter strength for luminance in ``cv2.fastNlMeansDenoisingColored``.
            Higher values remove more noise but may blur edges. Default is 3.
        denoise_h_color: Filter strength for color components. Default is 3.
        denoise_template_window: Template patch size for denoising. Default is 7.
        denoise_search_window: Search window size for denoising. Default is 21.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        denoise_h: int = 3,
        denoise_h_color: int = 3,
        denoise_template_window: int = 7,
        denoise_search_window: int = 21,
    ) -> None:
        self._clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )
        self._denoise_h = denoise_h
        self._denoise_h_color = denoise_h_color
        self._denoise_template_window = denoise_template_window
        self._denoise_search_window = denoise_search_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Apply the full preprocessing pipeline to a single BGR frame.

        Steps:
        1. Convert to LAB colour space.
        2. Apply CLAHE contrast normalisation to the L channel.
        3. Convert back to BGR.
        4. Apply light OpenCV non-local-means denoising.

        Args:
            frame: Input BGR image as a ``numpy.ndarray`` with dtype
                ``uint8`` and shape ``(H, W, 3)``.

        Returns:
            Preprocessed BGR image with the same shape and dtype as
            *frame*.

        Raises:
            ValueError: If *frame* is not a 3-channel uint8 BGR image.
        """
        self._validate_frame(frame)
        enhanced = self._apply_clahe(frame)
        enhanced = self._apply_denoise(enhanced)
        return enhanced

    def normalize_contrast(self, frame: np.ndarray) -> np.ndarray:
        """Apply only CLAHE contrast normalisation (no denoising).

        Useful when denoising is handled by the CNN model downstream.

        Args:
            frame: Input BGR image (H, W, 3) uint8.

        Returns:
            Contrast-normalised BGR image.

        Raises:
            ValueError: If *frame* is not a 3-channel uint8 BGR image.
        """
        self._validate_frame(frame)
        return self._apply_clahe(frame)

    def reduce_noise(self, frame: np.ndarray) -> np.ndarray:
        """Apply only non-local-means noise reduction (no CLAHE).

        Args:
            frame: Input BGR image (H, W, 3) uint8.

        Returns:
            Noise-reduced BGR image.

        Raises:
            ValueError: If *frame* is not a 3-channel uint8 BGR image.
        """
        self._validate_frame(frame)
        return self._apply_denoise(frame)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray):
            raise ValueError(
                f"frame must be a numpy.ndarray, got {type(frame).__name__}"
            )
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must have shape (H, W, 3), got {frame.shape}"
            )
        if frame.dtype != np.uint8:
            raise ValueError(
                f"frame must have dtype uint8, got {frame.dtype}"
            )

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Convert to LAB, equalise the L channel, convert back to BGR."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self._clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _apply_denoise(self, frame: np.ndarray) -> np.ndarray:
        """Lightweight non-local-means denoising (OpenCV)."""
        return cv2.fastNlMeansDenoisingColored(
            frame,
            None,
            self._denoise_h,
            self._denoise_h_color,
            self._denoise_template_window,
            self._denoise_search_window,
        )
