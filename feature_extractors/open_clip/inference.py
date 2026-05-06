"""CLIP ViT-L/14@336 recognizer via open_clip — same interface as adaface/inference.py.

Produces L2-normalized embeddings (768-dim) from CLIP's vision encoder
using the open_clip library (https://github.com/mlfoundations/open_clip).

Usage via extract_templates.py:
  python feature_extractors/extract_templates.py --model open_clip_vitl_p14_336 ...
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image

CLIP_MODEL_NAME = "ViT-L-14-336-quickgelu"
CLIP_PRETRAINED = "openai"
EMBED_DIM = 768
IMAGE_SIZE = 336

# CLIP normalization constants (from OpenAI CLIP)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def pil_to_clip_tensor(
    pil_img: Image.Image, size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
) -> torch.Tensor:
    """Convert a PIL RGB image to a CLIP-normalized tensor [3, H, W] float32."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != size:
        pil_img = pil_img.resize(size, Image.BICUBIC)
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    for c in range(3):
        np_img[:, :, c] = (np_img[:, :, c] - CLIP_MEAN[c]) / CLIP_STD[c]
    return torch.from_numpy(np_img).permute(2, 0, 1).contiguous()


def load_model(
    model_name: Optional[str] = None,
    pretrained: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Load CLIP model via open_clip.

    Uses ``open_clip.create_model`` which returns the full CLIP model.
    ``encode_image()`` produces 768-dim projected embeddings in the
    joint vision-language space.
    """
    import open_clip

    name = model_name or CLIP_MODEL_NAME
    ptag = pretrained or CLIP_PRETRAINED
    cache_dir = os.environ.get("OPENCLIP_CACHE_DIR")
    model = open_clip.create_model(name, pretrained=ptag, cache_dir=cache_dir)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model = model.to(device)
    return model


@torch.no_grad()
def encode_batch(
    model: torch.nn.Module, pixel_values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch → (L2-normalized embeddings [B, 768], pre-norm norms [B]).

    Uses ``model.encode_image()`` which applies the visual projection,
    producing 768-dim vectors in the joint CLIP embedding space.
    """
    emb = model.encode_image(pixel_values).float()
    norms = torch.linalg.norm(emb, ord=2, dim=1)
    emb = emb / (norms.unsqueeze(1) + 1e-8)
    return emb, norms


@torch.no_grad()
def encode_pil(
    model: torch.nn.Module,
    pil_img: Image.Image,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single PIL image → (embedding [768], norm scalar)."""
    x = pil_to_clip_tensor(pil_img).unsqueeze(0)
    if device is not None:
        x = x.to(device)
    emb, norms = encode_batch(model, x)
    return emb[0], norms[0]
