"""SwinFace Swin-T inference for face recognition (backbone-only, 512-D)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from . import net

DEFAULT_CKPT = Path(__file__).resolve().parent / "pretrained" / "checkpoint_step_79999_gpu_0.pt"


def pil_to_rgb_tensor(pil_img: Image.Image, size: Tuple[int, int] = (112, 112)) -> torch.Tensor:
    """Convert a PIL RGB image to an RGB tensor in [-1, 1], shape [3, H, W]."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != size:
        pil_img = pil_img.resize(size)
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    rgb = (np_img - 0.5) / 0.5
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()


def load_model(
    ckpt_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    ckpt = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not ckpt.is_file():
        raise FileNotFoundError(f"SwinFace checkpoint not found: {ckpt}")

    model = net.SwinTransformer(
        img_size=112, patch_size=2, in_chans=3, num_classes=512,
        embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
        window_size=7, drop_path_rate=0.1,
    )

    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["state_dict_backbone"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model.to(device)
    return model


@torch.no_grad()
def encode_batch(model: torch.nn.Module, rgb_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch, return (l2-normalized embeddings, pre-norm norms)."""
    raw = model(rgb_batch)
    norms = torch.norm(raw, 2, 1, keepdim=True)
    embeddings = raw / (norms + 1e-10)
    return embeddings.float(), norms.squeeze(1).float()
