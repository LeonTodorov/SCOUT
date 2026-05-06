"""JumpReLU SAE inference. Loads checkpoints saved by batchtopk training.

At training time the SAE selects active features via BatchTopK; at inference
time we use the learned global threshold to gate ReLU activations
(JumpReLU style), which is the standard substitution.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from batchtopk.batchtopk import DTYPE_MAP


class JumpReLUInferenceSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.dtype = dtype
        self.device = device

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae, dtype=dtype, device=device), requires_grad=False)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in, dtype=dtype, device=device), requires_grad=False)
        self.b_enc = nn.Parameter(torch.zeros(d_sae, dtype=dtype, device=device), requires_grad=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device), requires_grad=False)
        self.register_buffer("threshold", torch.zeros(d_sae, dtype=dtype, device=device))
        self.global_threshold: float = 0.0

    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_pre = x @ self.W_enc + self.b_enc
        relu_acts = torch.relu(hidden_pre)
        mask = (hidden_pre > self.threshold).to(relu_acts.dtype)
        return relu_acts * mask, hidden_pre

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_acts, hidden_pre = self.encode_with_hidden_pre(x)
        recon = self.decode(feature_acts)
        return recon, feature_acts, hidden_pre


def _required(state_dict: dict, key: str) -> torch.Tensor:
    value = state_dict.get(key)
    if value is None:
        raise ValueError(f"State dict is missing '{key}'.")
    return value


def load_model(weights_path: Path, device: torch.device) -> JumpReLUInferenceSAE:
    payload = torch.load(weights_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Expected model payload to be a dict with 'state_dict' and 'config'.")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Model payload is missing a valid 'state_dict'.")

    cfg = dict(payload.get("config", {}))
    if not cfg:
        raise ValueError("Model payload is missing the config needed to rebuild the SAE.")

    d_in = int(cfg["d_in"])
    d_sae = int(cfg["d_sae"])
    dtype_name = cfg.get("dtype", "float32")
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype in config: {dtype_name}")
    dtype = DTYPE_MAP[dtype_name]

    topk_threshold = state_dict.get("topk_threshold")
    if topk_threshold is None or topk_threshold.numel() != 1:
        raise ValueError("Expected scalar 'topk_threshold' buffer in state dict.")
    threshold_value = float(topk_threshold.item())

    converted = {
        "W_enc": _required(state_dict, "W_enc").to(dtype=dtype).detach().clone(),
        "W_dec": _required(state_dict, "W_dec").to(dtype=dtype).detach().clone(),
        "b_enc": _required(state_dict, "b_enc").to(dtype=dtype).detach().clone(),
        "b_dec": _required(state_dict, "b_dec").to(dtype=dtype).detach().clone(),
        "threshold": torch.full((d_sae,), threshold_value, dtype=dtype),
    }

    model = JumpReLUInferenceSAE(d_in=d_in, d_sae=d_sae, dtype=dtype, device=torch.device("cpu"))
    model.load_state_dict(converted)
    model.global_threshold = threshold_value
    model.to(device)
    model.device = device
    return model
