"""Image quality metrics: PSNR, SSIM, LPIPS.

All functions expect BGR tensors in [-1, 1] with shape (B, 3, H, W).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from perceptual import bgr_to_rgb, bgr_to_rgb01


# ---------------------------------------------------------------------------
# PSNR  (Peak Signal-to-Noise Ratio)
# ---------------------------------------------------------------------------
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    """Per-image PSNR in dB.  data_range=2.0 for [-1,1] tensors.

    Returns: (B,) tensor of PSNR values.
    """
    mse = (pred - target).pow(2).mean(dim=(1, 2, 3))
    return 10.0 * torch.log10(data_range ** 2 / mse.clamp(min=1e-10))


def psnr_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
    return psnr(pred, target).mean().item()


# ---------------------------------------------------------------------------
# SSIM  (Structural Similarity Index)
# ---------------------------------------------------------------------------
def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2.0
    g = torch.exp(-coords.pow(2) / (2 * sigma ** 2))
    return g / g.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device: torch.device) -> torch.Tensor:
    k1d = _gaussian_kernel_1d(size, sigma, device)
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    kernel = k2d.expand(channels, 1, size, size).contiguous()
    return kernel


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    data_range: float = 2.0,
) -> torch.Tensor:
    """Per-image SSIM.  Returns: (B,) tensor."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    B, C, H, W = pred.shape

    kernel = _gaussian_kernel_2d(window_size, 1.5, C, pred.device)
    pad = window_size // 2

    mu1 = F.conv2d(pred, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(target, kernel, padding=pad, groups=C)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(target * target, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(pred * target, kernel, padding=pad, groups=C) - mu12

    num = (2 * mu12 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = num / den
    return ssim_map.mean(dim=(1, 2, 3))  # (B,)


def ssim_mean(pred: torch.Tensor, target: torch.Tensor) -> float:
    return ssim(pred, target).mean().item()


# ---------------------------------------------------------------------------
# LPIPS  (Learned Perceptual Image Patch Similarity)
# ---------------------------------------------------------------------------
class LPIPSMetric:
    """Lazy-loaded LPIPS metric (lower is better)."""

    def __init__(self, device: Optional[torch.device] = None):
        self._lpips = None
        self._device = device

    def _ensure(self, device: torch.device):
        if self._lpips is None:
            try:
                import lpips  # type: ignore

                self._lpips = lpips.LPIPS(net="alex").eval().to(device)
                for p in self._lpips.parameters():
                    p.requires_grad = False
            except Exception:
                # Offline/frozen environments may not have lpips installed.
                # Keep train/val running; report LPIPS as NaN.
                self._lpips = False
        return self._lpips

    @torch.no_grad()
    def __call__(self, pred_bgr: torch.Tensor, target_bgr: torch.Tensor) -> torch.Tensor:
        """Returns (B,) tensor of LPIPS distances."""
        fn = self._ensure(pred_bgr.device)
        if fn is False:
            return torch.full((pred_bgr.shape[0],), float("nan"), device=pred_bgr.device)
        # LPIPS expects RGB in [-1, 1]
        pred_rgb = bgr_to_rgb(pred_bgr)
        target_rgb = bgr_to_rgb(target_bgr)
        return fn(pred_rgb, target_rgb).squeeze()

    def mean(self, pred_bgr: torch.Tensor, target_bgr: torch.Tensor) -> float:
        return self(pred_bgr, target_bgr).mean().item()
