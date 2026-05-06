from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn


DTypeName = Literal["float32", "float64", "bfloat16", "float16"]

DTYPE_MAP: dict[DTypeName, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass
class BatchTopKConfig:
    d_in: int
    d_sae: int
    device: str = "cpu"
    dtype: DTypeName = "float32"
    # Number of active features per sample on average (k * batch_size kept globally)
    k: float = 32.0
    # EMA learning rate for the global positive min threshold
    topk_threshold_lr: float = 0.01
    # Anthropic-style decoder init norm; None disables
    decoder_init_norm: float | None = 0.1


class BatchTopKTrainingSAE(nn.Module):
    def __init__(self, cfg: BatchTopKConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.dtype not in DTYPE_MAP:
            raise ValueError(f"Unsupported dtype: {cfg.dtype}")
        if cfg.k < 0:
            raise ValueError(f"k must be non-negative, got {cfg.k}.")
        self.dtype = DTYPE_MAP[cfg.dtype]
        self.device = torch.device(cfg.device)

        # Parameters
        self.W_enc = nn.Parameter(
            torch.empty(cfg.d_in, cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.W_dec = nn.Parameter(
            torch.empty(cfg.d_sae, cfg.d_in, dtype=self.dtype, device=self.device)
        )
        self.b_enc = nn.Parameter(
            torch.zeros(cfg.d_sae, dtype=self.dtype, device=self.device)
        )
        self.b_dec = nn.Parameter(
            torch.zeros(cfg.d_in, dtype=self.dtype, device=self.device)
        )

        # Global top-k positive min threshold (tracked for potential inference use)
        self.register_buffer(
            "topk_threshold",
            torch.tensor(0.0, dtype=torch.double, device=self.device),
        )

        self.reset_parameters(cfg.decoder_init_norm)

    def reset_parameters(self, decoder_init_norm: float | None) -> None:
        with torch.no_grad():
            nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
            if decoder_init_norm is not None:
                norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-6)
                self.W_dec.copy_(self.W_dec / norms * decoder_init_norm)
            # Tied init.
            self.W_enc.copy_(self.W_dec.T)
            nn.init.zeros_(self.b_enc)
            nn.init.zeros_(self.b_dec)

    def _batch_topk(self, hidden_pre: torch.Tensor) -> torch.Tensor:
        # ReLU then select the top int(k * batch) across the flattened batch
        acts = hidden_pre.relu()
        if acts.ndim != 2:
            raise ValueError("BatchTopKTrainingSAE expects 2D input: [batch, d_in]")
        k_total = int(self.cfg.k * acts.shape[0])
        if k_total <= 0:
            return torch.zeros_like(acts)
        flat = acts.flatten()
        if k_total >= flat.numel():
            return acts
        topk = torch.topk(flat, k=k_total, dim=-1)
        out_flat = torch.zeros_like(flat)
        out_flat.scatter_(dim=-1, index=topk.indices, src=topk.values)
        return out_flat.view_as(acts)

    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_pre = x @ self.W_enc + self.b_enc
        feature_acts = self._batch_topk(hidden_pre)
        return feature_acts, hidden_pre

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_acts, hidden_pre = self.encode_with_hidden_pre(x)
        recon = self.decode(feature_acts)
        return recon, feature_acts, hidden_pre

    @torch.no_grad()
    def update_topk_threshold(self, feature_acts: torch.Tensor) -> None:
        # EMA update toward the minimum positive activation in the batch
        pos_vals = feature_acts[feature_acts > 0]
        if pos_vals.numel() == 0:
            return
        min_pos = pos_vals.min().to(dtype=self.topk_threshold.dtype)
        lr = float(self.cfg.topk_threshold_lr)
        # Disable autocast to avoid precision issues
        with torch.autocast(self.topk_threshold.device.type, enabled=False):
            updated = (1 - lr) * self.topk_threshold + lr * min_pos
            self.topk_threshold.copy_(updated)


class JumpReLUInferenceSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.dtype = dtype
        self.device = device

        self.W_enc = nn.Parameter(
            torch.empty(d_in, d_sae, dtype=dtype, device=device), requires_grad=False
        )
        self.W_dec = nn.Parameter(
            torch.empty(d_sae, d_in, dtype=dtype, device=device), requires_grad=False
        )
        self.b_enc = nn.Parameter(
            torch.zeros(d_sae, dtype=dtype, device=device), requires_grad=False
        )
        self.b_dec = nn.Parameter(
            torch.zeros(d_in, dtype=dtype, device=device), requires_grad=False
        )
        self.register_buffer("threshold", torch.zeros(d_sae, dtype=dtype, device=device))
        self.global_threshold: float = 0.0

    def encode_with_hidden_pre(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_pre = x @ self.W_enc + self.b_enc
        relu_acts = torch.relu(hidden_pre)
        mask = (hidden_pre > self.threshold).to(relu_acts.dtype)
        feature_acts = relu_acts * mask
        return feature_acts, hidden_pre

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature_acts, hidden_pre = self.encode_with_hidden_pre(x)
        recon = self.decode(feature_acts)
        return recon, feature_acts, hidden_pre


def build_batch_topk_training_sae(
    d_in: int,
    d_sae: int,
    *,
    device: str = "cpu",
    dtype: DTypeName = "float32",
    k: float = 100.0,
    topk_threshold_lr: float = 0.01,
    decoder_init_norm: float | None = 0.1,
) -> BatchTopKTrainingSAE:
    cfg = BatchTopKConfig(
        d_in=d_in,
        d_sae=d_sae,
        device=device,
        dtype=dtype,
        k=k,
        topk_threshold_lr=topk_threshold_lr,
        decoder_init_norm=decoder_init_norm,
    )
    return BatchTopKTrainingSAE(cfg)


@torch.no_grad()
def estimate_decoder_prescale(
    model: BatchTopKTrainingSAE,
    *,
    batch_size: int = 4096,
) -> float:
    """Estimate the decoder prescale for the initialized SAE.

    The estimate matches the old compute_prescale.py workflow, but uses the
    actual model instance that will be trained.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    x = torch.randn(
        batch_size,
        model.cfg.d_in,
        dtype=model.dtype,
        device=model.device,
    )
    x = x / x.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    y, _, _ = model(x)

    input_std = x.std()
    if input_std <= 0:
        raise RuntimeError("Cannot estimate decoder prescale from zero-variance input.")
    return float((y.std() / input_std).item())
