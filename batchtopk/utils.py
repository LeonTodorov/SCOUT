from __future__ import annotations

from os import PathLike
from pathlib import Path

import torch

from batchtopk import DTYPE_MAP, JumpReLUInferenceSAE


def _torch_load_cpu(path: str | PathLike[str]) -> object:
    # Keep compatibility with older torch versions while silencing the
    # FutureWarning on newer versions where `weights_only` default will change.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _get_required_state_tensor(
    state_dict: dict[str, torch.Tensor],
    key: str,
) -> torch.Tensor:
    value = state_dict.get(key)
    if value is None:
        raise ValueError(f"State dict is missing '{key}'.")
    return value


def load_model(weights_path: Path, device: torch.device) -> JumpReLUInferenceSAE:
    payload = _torch_load_cpu(weights_path)
    if not isinstance(payload, dict):
        raise ValueError("Expected model payload to be a dict with 'state_dict' and 'config'.")

    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("Model payload is missing a valid 'state_dict'.")

    cfg_dict = dict(payload.get("config", {}))
    if not cfg_dict:
        raise ValueError("Model payload is missing the config needed to rebuild the SAE.")

    d_in = int(cfg_dict["d_in"])
    d_sae = int(cfg_dict["d_sae"])
    dtype_name = cfg_dict.get("dtype", "float32")
    if dtype_name not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype in config: {dtype_name}")
    dtype = DTYPE_MAP[dtype_name]

    topk_threshold_tensor = state_dict.get("topk_threshold")
    if topk_threshold_tensor is None:
        raise ValueError("State dict is missing 'topk_threshold' buffer.")
    if topk_threshold_tensor.numel() != 1:
        raise ValueError("Expected scalar 'topk_threshold' buffer.")
    threshold_value = float(topk_threshold_tensor.item())

    converted_state = {
        "W_enc": _get_required_state_tensor(state_dict, "W_enc").to(dtype=dtype).detach().clone(),
        "W_dec": _get_required_state_tensor(state_dict, "W_dec").to(dtype=dtype).detach().clone(),
        "b_enc": _get_required_state_tensor(state_dict, "b_enc").to(dtype=dtype).detach().clone(),
        "b_dec": _get_required_state_tensor(state_dict, "b_dec").to(dtype=dtype).detach().clone(),
        "threshold": torch.full((d_sae,), threshold_value, dtype=dtype),
    }

    model = JumpReLUInferenceSAE(d_in=d_in, d_sae=d_sae, dtype=dtype, device=torch.device("cpu"))
    model.load_state_dict(converted_state)
    model.global_threshold = threshold_value
    model.to(device)
    model.device = device
    return model
