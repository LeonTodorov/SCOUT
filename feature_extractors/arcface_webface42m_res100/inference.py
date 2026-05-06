"""Arc2Face ArcFace iResNet-100 (WebFace42M) inference.

Supports both ONNX Runtime and onnx2torch-converted PyTorch models.
The PyTorch path is preferred for DFD training (GPU support, forward hooks
for perceptual loss). The ONNX path is kept for backward compatibility.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

DEFAULT_ONNX_MODEL = Path(__file__).resolve().parent / "pretrained" / "arcface.onnx"
DEFAULT_TORCH_MODEL = Path(__file__).resolve().parent / "pretrained" / "arc2face_arcface_r100_model.pth"


def pil_to_bgr_tensor(pil_img: Image.Image, size: Tuple[int, int] = (112, 112)) -> torch.Tensor:
    """Convert a PIL RGB image to a BGR tensor in [-1, 1], shape [3, H, W]."""
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != size:
        pil_img = pil_img.resize(size)
    np_img = np.array(pil_img)[:, :, ::-1].astype(np.float32) / 255.0
    bgr = (np_img - 0.5) / 0.5
    return torch.from_numpy(bgr).permute(2, 0, 1).contiguous()


# ---------------------------------------------------------------------------
# ONNX Runtime path (kept for backward compatibility)
# ---------------------------------------------------------------------------

class OnnxArcFaceModel:
    """Thin wrapper around an ONNX Runtime session that behaves like a callable."""

    def __init__(self, session):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        np_input = x.cpu().numpy()
        out = self.session.run([self.output_name], {self.input_name: np_input})[0]
        return torch.from_numpy(out)

    def eval(self):
        return self

    def to(self, device):
        return self

    def parameters(self):
        return iter([])


def load_onnx_model(
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> OnnxArcFaceModel:
    """Load the ONNX model via ONNX Runtime."""
    import onnxruntime as ort

    path = Path(model_path) if model_path else DEFAULT_ONNX_MODEL
    if not path.is_file():
        raise FileNotFoundError(f"Arc2Face ArcFace ONNX model not found: {path}")
    providers = ["CPUExecutionProvider"]
    if device is not None and device.type == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(str(path), providers=providers)
    return OnnxArcFaceModel(sess)


# ---------------------------------------------------------------------------
# PyTorch path (preferred for DFD training)
# ---------------------------------------------------------------------------

def load_model(
    model_path: Optional[Path] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    """Load the onnx2torch-converted PyTorch model."""
    path = Path(model_path) if model_path else DEFAULT_TORCH_MODEL
    if not path.is_file():
        raise FileNotFoundError(f"Arc2Face ArcFace PyTorch model not found: {path}")
    model = torch.load(path, map_location=device, weights_only=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if device is not None:
        model.to(device)
    return model


@torch.no_grad()
def encode_batch(
    model: Union[torch.nn.Module, OnnxArcFaceModel],
    bgr_batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode a batch, return (l2-normalized embeddings, pre-norm norms)."""
    raw = model(bgr_batch)
    norms = torch.norm(raw, 2, 1, keepdim=True)
    embeddings = raw / (norms + 1e-10)
    return embeddings.float(), norms.squeeze(1).float()
