from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

from . import net

DEFAULT_CKPT = Path(__file__).resolve().parent / "pretrained" / "adaface_ir50_ms1mv2.ckpt"


def pil_to_bgr_tensor(pil_img: Image.Image, size: Tuple[int, int] = (112, 112)) -> torch.Tensor:
    """Convert a PIL RGB image to a BGR tensor in [-1, 1], shape [3, H, W]."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != size:
        # Match the historical extraction path where PIL resize used default
        # interpolation (BICUBIC for RGB).
        pil_img = pil_img.resize(size)

    np_img = np.array(pil_img)  # RGB uint8
    bgr = np_img[:, :, ::-1].astype(np.float32) / 255.0
    bgr = (bgr - 0.5) / 0.5
    return torch.from_numpy(bgr).permute(2, 0, 1).contiguous()


def load_model(
    ckpt_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    ckpt = Path(ckpt_path) if ckpt_path else DEFAULT_CKPT
    if not ckpt.is_file():
        raise FileNotFoundError(f"AdaFace checkpoint not found: {ckpt}")

    model = net.build_model("ir_50")
    state = torch.load(ckpt, map_location=device, weights_only=False)["state_dict"]
    model_state = {k[6:]: v for k, v in state.items() if k.startswith("model.")}
    model.load_state_dict(model_state)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model.to(device)
    return model


@torch.no_grad()
def encode_batch(model: torch.nn.Module, bgr_batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch and return (normalized_embeddings, pre-normalization_norms)."""
    embeddings, norms = model(bgr_batch)
    if embeddings.ndim != 2:
        raise RuntimeError(f"AdaFace embeddings must be 2D, got shape {tuple(embeddings.shape)}")
    norms = norms.reshape(-1)
    if norms.shape[0] != embeddings.shape[0]:
        raise RuntimeError(
            f"AdaFace norms length {norms.shape[0]} does not match batch size {embeddings.shape[0]}"
        )
    return embeddings.float(), norms.float()


@torch.no_grad()
def encode_pil(
    model: torch.nn.Module,
    pil_img: Image.Image,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = pil_to_bgr_tensor(pil_img).unsqueeze(0)
    if device is not None:
        x = x.to(device)
    emb, norms = encode_batch(model, x)
    return emb[0], norms[0]
