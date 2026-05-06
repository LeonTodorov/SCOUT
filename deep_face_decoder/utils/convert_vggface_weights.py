#!/usr/bin/env python3
"""Convert VGGFace (Keras) VGG16 weights to PyTorch format.

Produces a state_dict for torchvision.models.vgg16().features (conv layers only).
Requires: pip install keras keras-vggface tensorflow

Usage:
    python utils/convert_vggface_weights.py [--output pretrained/vggface_vgg16_features.pth]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def convert(output_path: str = "pretrained/vggface_vgg16_features.pth"):
    try:
        from keras_vggface.vggface import VGGFace
    except ImportError:
        raise SystemExit(
            "keras_vggface not installed. Run:\n"
            "  pip install keras-vggface tensorflow\n"
            "Then re-run this script."
        )

    print("Loading Keras VGGFace VGG16 model...")
    keras_model = VGGFace(model="vgg16", weights="vggface", include_top=False)

    from torchvision.models import vgg16
    torch_model = vgg16(weights=None)

    # Map Keras conv layer names to torchvision feature indices
    # VGG16 has 13 conv layers; torchvision indices for Conv2d layers:
    torch_conv_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]

    keras_conv_layers = [l for l in keras_model.layers if "conv" in l.name.lower()]
    assert len(keras_conv_layers) == 13, f"Expected 13 conv layers, got {len(keras_conv_layers)}"

    print(f"Converting {len(keras_conv_layers)} conv layers...")
    for k_layer, t_idx in zip(keras_conv_layers, torch_conv_indices):
        weights = k_layer.get_weights()
        w = weights[0]  # (H, W, C_in, C_out) in Keras
        b = weights[1]  # (C_out,)
        # Keras: (H, W, C_in, C_out) -> PyTorch: (C_out, C_in, H, W)
        w_pt = torch.from_numpy(np.transpose(w, (3, 2, 0, 1)).copy())
        b_pt = torch.from_numpy(b.copy())
        torch_model.features[t_idx].weight.data = w_pt
        torch_model.features[t_idx].bias.data = b_pt
        print(f"  {k_layer.name} -> features[{t_idx}]: {list(w_pt.shape)}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_model.features.state_dict(), out)
    print(f"Saved VGGFace VGG16 features state_dict to {out}")
    print(f"File size: {out.stat().st_size / 1e6:.1f} MB")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="pretrained/vggface_vgg16_features.pth")
    args = p.parse_args()
    convert(args.output)


if __name__ == "__main__":
    main()
