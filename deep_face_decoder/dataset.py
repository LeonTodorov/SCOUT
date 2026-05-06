from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_log = logging.getLogger(__name__)


def build_line_index(list_path: Path, index_path: Optional[Path] = None) -> Path:
    """Build a byte-offset index file for fast random access."""
    list_path = Path(list_path)
    if index_path is None:
        index_path = list_path.with_suffix(list_path.suffix + ".idx.npy")
    offsets: List[int] = []
    offset = 0
    with list_path.open("rb") as f:
        for line in f:
            offsets.append(offset)
            offset += len(line)
    arr = np.array(offsets, dtype=np.int64)
    np.save(index_path, arr)
    return index_path


def load_line_index(list_path: Path) -> np.ndarray:
    list_path = Path(list_path)
    index_path = list_path.with_suffix(list_path.suffix + ".idx.npy")
    if not index_path.exists():
        build_line_index(list_path, index_path)
    return np.load(index_path)


def pil_to_bgr_tensor(pil_img: Image.Image, size: Tuple[int, int] = (112, 112)) -> torch.Tensor:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    if pil_img.size != size:
        pil_img = pil_img.resize(size, Image.BILINEAR)
    np_img = np.array(pil_img)  # RGB
    bgr = np_img[:, :, ::-1].astype(np.float32) / 255.0
    bgr = (bgr - 0.5) / 0.5
    tensor = torch.from_numpy(bgr).permute(2, 0, 1).contiguous()
    return tensor


class IndexedLineDataset(Dataset):

    _MISSING_LOG: str | None = None        # set via class attr or per-instance
    _missing_logged: Set[str] = set()       # shared across workers (per process)

    def __init__(
        self,
        list_path: str | Path,
        root: str | Path,
        image_size: int = 112,
        return_path: bool = False,
        missing_log: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.list_path = Path(list_path)
        self.root = Path(root)
        self.image_size = image_size
        self.return_path = return_path
        self.index = load_line_index(self.list_path)
        self._file_size = self.list_path.stat().st_size
        self._fd = os.open(str(self.list_path), os.O_RDONLY)
        if missing_log is not None:
            self._missing_log = str(missing_log)

    def __len__(self) -> int:
        return int(len(self.index))

    def _read_line(self, idx: int) -> str:
        start = int(self.index[idx])
        if idx + 1 < len(self.index):
            nbytes = int(self.index[idx + 1]) - start
        else:
            nbytes = self._file_size - start
        data = os.pread(self._fd, nbytes, start)
        return data.decode("utf-8").strip()

    def __getitem__(self, idx: int):
        rel = self._read_line(idx)
        img_path = self.root / rel if not os.path.isabs(rel) else Path(rel)
        try:
            with Image.open(img_path) as img:
                tensor = pil_to_bgr_tensor(img, size=(self.image_size, self.image_size))
        except FileNotFoundError:
            self._record_missing(str(img_path))
            # Return a random other sample instead
            alt_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(alt_idx)
        if self.return_path:
            return tensor, str(img_path)
        return tensor

    # ---- missing-file bookkeeping ----

    def _record_missing(self, path: str) -> None:
        """Log and append missing path to a file (once per path per process)."""
        if path in self._missing_logged:
            return
        self._missing_logged.add(path)
        _log.warning("Skipping missing file: %s", path)
        log_path = self._missing_log
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(path + "\n")
            except OSError:
                pass

    def __del__(self):
        try:
            os.close(self._fd)
        except Exception:
            pass
