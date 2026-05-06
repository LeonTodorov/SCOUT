from __future__ import annotations

from os import PathLike

import torch
from torch.utils.data import Dataset

from utils import _torch_load_cpu


class VectorDataset(Dataset[torch.Tensor]):
    """Dataset backed by a 2D tensor stored on disk via `torch.save`."""

    def __init__(self, src_file: str | PathLike[str], N: int | None = None):
        super().__init__()
        data = _torch_load_cpu(src_file)
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                f"Expected tensor payload in {src_file}, got {type(data).__name__}."
            )
        if data.ndim != 2:
            raise ValueError(
                f"Expected a 2D tensor in {src_file}, got shape {tuple(data.shape)}."
            )
        if N is not None:
            if N < 0:
                raise ValueError(f"N must be non-negative, got {N}.")
            data = data[:N]
        self.all_data = data

    def __len__(self) -> int:
        return self.all_data.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.all_data[idx]
