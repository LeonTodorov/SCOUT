"""DFD-backed manipulation helpers: encoder/decoder/SAE loaders + neuron-direction edits.

Wires SCOUT's paper-named encoders ([deep_face_decoder/encoders.py]) and the
trained DFD decoder ([deep_face_decoder/decoder.py]) into the analysis script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from PIL import Image

from _paths import setup_paths

setup_paths()

from encoders import ENCODER_NAMES, build_encoder, pil_to_bgr_tensor  # noqa: E402
from decoder import Decoder  # noqa: E402

from batchtopk.inference import load_model as load_sae_model


def load_sae(weights_path: Path, device: torch.device):
    return load_sae_model(weights_path, device)


def load_encoder(encoder_name: str, device: torch.device) -> tuple[torch.nn.Module, Callable]:
    """Build the named encoder and return (encoder, pil_to_bgr_tensor)."""
    if encoder_name not in ENCODER_NAMES:
        raise ValueError(
            f"Unknown encoder: {encoder_name}. Choose from {ENCODER_NAMES}."
        )
    encoder = build_encoder(encoder_name, device)
    return encoder, pil_to_bgr_tensor


def load_decoder(decoder_ckpt: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    """Load a DFD decoder checkpoint, return (decoder, output_size)."""
    ckpt = torch.load(decoder_ckpt, map_location=device, weights_only=False)
    output_size = 112
    if isinstance(ckpt, dict) and "config" in ckpt:
        output_size = int(ckpt["config"].get("output_size", 112))
    decoder = Decoder(output_size=output_size).to(device)
    state = ckpt["decoder"] if isinstance(ckpt, dict) and "decoder" in ckpt else ckpt
    decoder.load_state_dict(state)
    decoder.eval()
    return decoder, output_size


def direction_from_neuron(sae, neuron_idx: int) -> torch.Tensor:
    """Unit-length decoder direction for a single SAE neuron."""
    if neuron_idx < 0 or neuron_idx >= sae.d_sae:
        raise ValueError(f"Neuron {neuron_idx} out of range [0, {sae.d_sae}).")
    direction = sae.W_dec[neuron_idx, :]
    return direction / (direction.norm() + 1e-8)


def manipulate_linear(template: torch.Tensor, direction: torch.Tensor, alpha: float) -> torch.Tensor:
    """Add alpha*direction to a face template and re-normalize."""
    modified = template + alpha * direction.unsqueeze(0)
    return modified / (modified.norm(dim=1, keepdim=True) + 1e-8)


def list_images_under(root: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        p.resolve() for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )


def decoded_tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a single BGR [-1, 1] decoder output (3,H,W) to RGB PIL."""
    rgb = tensor[[2, 1, 0], :, :]
    rgb = ((rgb + 1.0) / 2.0).clamp(0, 1)
    return Image.fromarray((rgb.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
