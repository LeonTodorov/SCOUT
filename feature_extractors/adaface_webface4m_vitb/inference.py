from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from . import net

DEFAULT_CKPT = Path(__file__).resolve().parent / "pretrained" / "adaface_webface4m_vitb.pt"
LEGACY_CKPT = Path(__file__).resolve().parent / "pretrained" / "pretrained_model" / "model.pt"


def pil_to_rgb_tensor(pil_img: Image.Image, size: Tuple[int, int] = (112, 112)) -> torch.Tensor:
    """Convert a PIL image to RGB tensor in [-1, 1], shape [3, H, W]."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != size:
        pil_img = pil_img.resize(size, Image.BILINEAR)
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    np_img = (np_img - 0.5) / 0.5
    return torch.from_numpy(np_img).permute(2, 0, 1).contiguous()


def load_model(
    ckpt_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    if ckpt_path is None:
        if DEFAULT_CKPT.is_file():
            ckpt = DEFAULT_CKPT
        elif LEGACY_CKPT.is_file():
            ckpt = LEGACY_CKPT
        else:
            raise FileNotFoundError(
                "AdaFace-ViT checkpoint not found. Expected one of:\n"
                f"  {DEFAULT_CKPT}\n"
                f"  {LEGACY_CKPT}"
            )
    else:
        ckpt = Path(ckpt_path).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"AdaFace-ViT checkpoint not found: {ckpt}")

    model = net.build_model(variant="base", output_dim=512, color_space="RGB")
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise RuntimeError(f"Expected checkpoint state_dict dict in {ckpt}, got {type(state)!r}")
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model = model.to(device)
    return model


@torch.no_grad()
def encode_batch(model: torch.nn.Module, rgb_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch and return (raw_embeddings, l2_norms)."""
    emb = model(rgb_batch).float()
    if emb.ndim != 2:
        raise RuntimeError(f"AdaFace-ViT embeddings must be 2D, got shape {tuple(emb.shape)}")
    norms = torch.linalg.norm(emb, ord=2, dim=1)
    return emb, norms.float()


@torch.no_grad()
def encode_pil(
    model: torch.nn.Module,
    pil_img: Image.Image,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = pil_to_rgb_tensor(pil_img).unsqueeze(0)
    if device is not None:
        x = x.to(device)
    emb, norms = encode_batch(model, x)
    return emb[0], norms[0]
