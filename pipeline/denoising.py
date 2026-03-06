"""CNN-based image denoising model (DnCNN architecture) using PyTorch."""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class DnCNN(nn.Module):
    """Deep CNN denoiser inspired by DnCNN (Zhang et al., 2017).

    The network learns a residual mapping: given a noisy image *y*, the
    model predicts the noise component *v* so that the denoised output
    is ``y - v``.  This residual learning strategy speeds up training
    and improves quality over direct-mapping approaches.

    Architecture:
        - **Input layer**: Conv2d → ReLU (no batch norm).
        - **Hidden layers**: Conv2d → BatchNorm2d → ReLU (×``depth``-2).
        - **Output layer**: Conv2d (no BN / activation), outputs
          predicted noise.

    Args:
        channels: Number of image channels (1 for grayscale, 3 for RGB).
            Default is 1.
        num_features: Number of feature maps in each convolutional layer.
            Default is 64.
        depth: Total number of convolutional layers. Default is 17.
        kernel_size: Convolution kernel size. Default is 3.
    """

    def __init__(
        self,
        channels: int = 1,
        num_features: int = 64,
        depth: int = 17,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        layers: list[nn.Module] = []

        # --- Input layer (no BN) ---
        layers += [
            nn.Conv2d(channels, num_features, kernel_size, padding=padding, bias=False),
            nn.ReLU(inplace=True),
        ]

        # --- Hidden layers (with BN) ---
        for _ in range(depth - 2):
            layers += [
                nn.Conv2d(
                    num_features,
                    num_features,
                    kernel_size,
                    padding=padding,
                    bias=False,
                ),
                nn.BatchNorm2d(num_features),
                nn.ReLU(inplace=True),
            ]

        # --- Output layer (no BN / activation) ---
        layers.append(
            nn.Conv2d(num_features, channels, kernel_size, padding=padding, bias=False)
        )

        self.net = nn.Sequential(*layers)
        self._init_weights()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return denoised image tensor.

        Args:
            x: Noisy input tensor of shape ``(N, C, H, W)`` with values
               in the range ``[0, 1]``.

        Returns:
            Denoised tensor of the same shape and range.
        """
        noise = self.net(x)
        return torch.clamp(x - noise, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def denoise_frame(
        self,
        frame: np.ndarray,
        device: torch.device | None = None,
    ) -> np.ndarray:
        """Denoise a single BGR or grayscale ``uint8`` frame (NumPy array).

        Converts the frame to a float32 tensor in ``[0, 1]``, runs the
        forward pass, then converts back to ``uint8``.

        Args:
            frame: Input BGR (H, W, 3) or grayscale (H, W) ``uint8``
                array.
            device: Torch device to run inference on. Defaults to
                ``cuda`` if available, otherwise ``cpu``.

        Returns:
            Denoised ``uint8`` NumPy array with the same shape as
            *frame*.
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        grayscale = frame.ndim == 2
        if grayscale:
            inp = frame.astype(np.float32) / 255.0
            tensor = torch.from_numpy(inp).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        else:
            inp = frame.astype(np.float32) / 255.0
            # BGR → channel-first → (1, C, H, W)
            tensor = torch.from_numpy(inp.transpose(2, 0, 1)).unsqueeze(0)

        tensor = tensor.to(device)
        with torch.no_grad():
            out = self.forward(tensor)

        out_np = out.squeeze(0).cpu().numpy()
        if grayscale:
            out_np = out_np.squeeze(0)  # (H, W)
        else:
            out_np = out_np.transpose(1, 2, 0)  # (H, W, C)

        return (np.clip(out_np, 0.0, 1.0) * 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


def load_denoising_model(
    checkpoint_path: str | None = None,
    channels: int = 1,
    num_features: int = 64,
    depth: int = 17,
    device: torch.device | None = None,
) -> DnCNN:
    """Create and optionally load a pretrained :class:`DnCNN` model.

    Args:
        checkpoint_path: Path to a PyTorch checkpoint (``state_dict``)
            saved with ``torch.save``.  When *None* the model is
            returned with freshly initialised weights.
        channels: Number of image channels passed to :class:`DnCNN`.
        num_features: Feature map count passed to :class:`DnCNN`.
        depth: Layer depth passed to :class:`DnCNN`.
        device: Target device.  Defaults to ``cuda`` if available.

    Returns:
        :class:`DnCNN` instance in evaluation mode.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DnCNN(channels=channels, num_features=num_features, depth=depth)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model
