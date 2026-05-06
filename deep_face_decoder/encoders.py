"""Frozen face-recognition encoders, named after the paper convention.

One Encoder class dispatched by a name in ENCODER_NAMES:
  - adaface_ms1mv2_ires50
  - adaface_webface4m_vitb
  - arcface_webface42m_res100
  - swinface_ms1mv2_swint

Each encoder accepts BGR tensors in [-1, 1] and returns L2-normalized embeddings.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.path_setup import ensure_recognizers_on_path

ensure_recognizers_on_path()


REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTORS_ROOT = REPO_ROOT / "feature_extractors"


@dataclass(frozen=True)
class _EncoderSpec:
    ckpt_path: Path
    expects_rgb: bool       # convert BGR -> RGB before forward
    l2_normalize: bool      # normalize after encoding
    load_kwarg: str = "ckpt_path"  # name of the ckpt arg accepted by load_model


_SPECS = {
    "adaface_ms1mv2_ires50": _EncoderSpec(
        ckpt_path=EXTRACTORS_ROOT / "adaface_ms1mv2_ires50" / "pretrained" / "adaface_ir50_ms1mv2.ckpt",
        expects_rgb=False,
        l2_normalize=False,
    ),
    "adaface_webface4m_vitb": _EncoderSpec(
        ckpt_path=EXTRACTORS_ROOT / "adaface_webface4m_vitb" / "pretrained" / "adaface_webface4m_vitb.pt",
        expects_rgb=True,
        l2_normalize=True,
    ),
    "arcface_webface42m_res100": _EncoderSpec(
        ckpt_path=EXTRACTORS_ROOT / "arcface_webface42m_res100" / "pretrained" / "arc2face_arcface_r100_model.pth",
        expects_rgb=False,
        l2_normalize=False,
        load_kwarg="model_path",
    ),
    "swinface_ms1mv2_swint": _EncoderSpec(
        ckpt_path=EXTRACTORS_ROOT / "swinface_ms1mv2_swint" / "pretrained" / "checkpoint_step_79999_gpu_0.pt",
        expects_rgb=True,
        l2_normalize=True,
    ),
}

ENCODER_NAMES: Tuple[str, ...] = tuple(_SPECS.keys())


def _load_inference(name: str):
    """Lazily import the inference module for the chosen encoder."""
    if name == "adaface_ms1mv2_ires50":
        from feature_extractors.adaface_ms1mv2_ires50 import inference
    elif name == "adaface_webface4m_vitb":
        from feature_extractors.adaface_webface4m_vitb import inference
    elif name == "arcface_webface42m_res100":
        from feature_extractors.arcface_webface42m_res100 import inference
    elif name == "swinface_ms1mv2_swint":
        from feature_extractors.swinface_ms1mv2_swint import inference
    else:
        raise ValueError(f"Unknown encoder: {name}. Choose from {ENCODER_NAMES}")
    return inference


def pil_to_bgr_tensor(pil_img: Image.Image, size: Tuple[int, int] = (112, 112)) -> torch.Tensor:
    """Convert a PIL RGB image to BGR tensor in [-1, 1], shape [3, H, W]."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != size:
        pil_img = pil_img.resize(size, Image.BILINEAR)
    np_img = np.array(pil_img)[:, :, ::-1].astype(np.float32) / 255.0
    bgr = (np_img - 0.5) / 0.5
    return torch.from_numpy(bgr).permute(2, 0, 1).contiguous()


class Encoder(torch.nn.Module):
    """Frozen face-recognition encoder. Expects BGR tensors in [-1, 1]."""

    def __init__(self, name: str, ckpt_path: Optional[Path] = None, device: Optional[torch.device] = None):
        super().__init__()
        if name not in _SPECS:
            raise ValueError(f"Unknown encoder: {name}. Choose from {ENCODER_NAMES}")
        self.name = name
        self.spec = _SPECS[name]
        self.ckpt_path = Path(ckpt_path) if ckpt_path else self.spec.ckpt_path
        if not self.ckpt_path.exists():
            raise FileNotFoundError(f"{name} checkpoint not found: {self.ckpt_path}")
        self._inference = _load_inference(name)
        self.model = self._inference.load_model(
            **{self.spec.load_kwarg: self.ckpt_path}, device=device
        )

    @torch.no_grad()
    def forward(self, bgr_tensor: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings of shape [B, 512]."""
        x = bgr_tensor[:, [2, 1, 0], :, :] if self.spec.expects_rgb else bgr_tensor
        emb, _ = self._inference.encode_batch(self.model, x)
        if self.spec.l2_normalize:
            emb = F.normalize(emb.float(), p=2, dim=1)
        return emb

    @torch.no_grad()
    def encode_pil(self, pil_img: Image.Image, device: Optional[torch.device] = None) -> torch.Tensor:
        tensor = pil_to_bgr_tensor(pil_img).unsqueeze(0)
        if device is not None:
            tensor = tensor.to(device)
        return self.forward(tensor)


def build_encoder(name: str, device: Optional[torch.device] = None) -> Encoder:
    enc = Encoder(name, device=device)
    return enc.to(device) if device is not None else enc
