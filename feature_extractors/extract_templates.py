#!/usr/bin/env python3
"""Unified Glint360K template extraction for SCOUT feature extractors."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore", message="Can't initialize NVML")
warnings.filterwarnings(
    "ignore",
    message=("CUDA initialization: Unexpected error from cudaGetDeviceCount.*"),
)
warnings.filterwarnings(
    "ignore",
    message="Importing from timm.models.layers is deprecated, please import via timm.layers",
    category=FutureWarning,
)

from feature_extractors.adaface_ms1mv2_ires50 import inference as adaface_inference
from feature_extractors.adaface_webface4m_vitb import inference as adaface_vit_inference
from feature_extractors.arcface_webface42m_res100 import inference as arcface_res100_inference
from feature_extractors.open_clip import inference as open_clip_inference
from feature_extractors.swinface_ms1mv2_swint import inference as swinface_inference
from paths import load_paths


@dataclass(frozen=True)
class ModelAdapter:
    load_model: Callable[[torch.device], torch.nn.Module]
    preprocess_pil: Callable[[Image.Image], torch.Tensor]
    encode_batch: Callable[[torch.nn.Module, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]
    embed_dim: int


def get_model_adapter(model_name: str) -> ModelAdapter:
    if model_name == "adaface_ms1mv2_ires50":
        return ModelAdapter(
            load_model=lambda device: adaface_inference.load_model(device=device),
            preprocess_pil=adaface_inference.pil_to_bgr_tensor,
            encode_batch=adaface_inference.encode_batch,
            embed_dim=512,
        )
    if model_name == "adaface_webface4m_vitb":
        return ModelAdapter(
            load_model=lambda device: adaface_vit_inference.load_model(device=device),
            preprocess_pil=adaface_vit_inference.pil_to_rgb_tensor,
            encode_batch=adaface_vit_inference.encode_batch,
            embed_dim=512,
        )
    if model_name == "open_clip_vitl_p14_336":
        return ModelAdapter(
            load_model=lambda device: open_clip_inference.load_model(device=device),
            preprocess_pil=open_clip_inference.pil_to_clip_tensor,
            encode_batch=open_clip_inference.encode_batch,
            embed_dim=768,
        )
    if model_name == "arcface_webface42m_res100":
        return ModelAdapter(
            load_model=lambda device: arcface_res100_inference.load_model(device=device),
            preprocess_pil=arcface_res100_inference.pil_to_bgr_tensor,
            encode_batch=arcface_res100_inference.encode_batch,
            embed_dim=512,
        )
    if model_name == "swinface_ms1mv2_swint":
        return ModelAdapter(
            load_model=lambda device: swinface_inference.load_model(device=device),
            preprocess_pil=swinface_inference.pil_to_rgb_tensor,
            encode_batch=swinface_inference.encode_batch,
            embed_dim=512,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def parse_args() -> argparse.Namespace:
    P = load_paths()
    parser = argparse.ArgumentParser(
        description="Extract Glint360K templates/norms using a unified feature extractor interface.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=(
            "adaface_ms1mv2_ires50",
            "adaface_webface4m_vitb",
            "arcface_webface42m_res100",
            "swinface_ms1mv2_swint",
            "open_clip_vitl_p14_336",
        ),
    )
    parser.add_argument(
        "--ids-and-fns",
        type=Path,
        default=P["ids_and_fns"],
        help="Path to existing ids_and_fns.npy metadata (required order source).",
    )
    parser.add_argument(
        "--glint360k-root",
        type=Path,
        default=P["glint360k_root"],
        help="Root directory containing unpacked Glint360K images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to feature_extractors/templates/<model>.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device to use. 'auto' resolves to cuda:0 when available else cpu.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from memmaps/progress when available.",
    )
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument(
        "--skip-finalize",
        action="store_true",
        help="Only write memmaps/progress; do not write templates.pth/norms.pth.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate model + data mapping and exit without extraction.",
    )
    return parser.parse_args()


def to_abs(path: Path) -> Path:
    return path.expanduser().resolve()


def detect_default_device() -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        try:
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"


def atomic_json_dump(path: Path, obj: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(obj, indent=2))
    tmp_path.replace(path)


def glint_image_path(images_root: Path, identity: str, suffix: str) -> Path:
    return images_root / f"id_{identity}" / f"{identity}_{suffix}"


def load_ids_and_fns(
    ids_and_fns_path: Path,
    start_idx: int,
    end_idx: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    if not ids_and_fns_path.is_file():
        raise RuntimeError(f"ids_and_fns file not found: {ids_and_fns_path}")
    arr = np.load(ids_and_fns_path, allow_pickle=True)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise RuntimeError(f"Expected ids_and_fns shape [N,2], got {arr.shape}")

    n_total = arr.shape[0]
    if start_idx < 0 or start_idx >= n_total:
        raise RuntimeError(f"Invalid --start-index={start_idx} for total rows={n_total}")
    if end_idx is None:
        end_idx = n_total
    if end_idx <= start_idx or end_idx > n_total:
        raise RuntimeError(
            f"Invalid slice start={start_idx}, end={end_idx}, total rows={n_total}"
        )

    slice_arr = arr[start_idx:end_idx]
    ids = slice_arr[:, 0].astype(str, copy=False)
    fns = slice_arr[:, 1].astype(str, copy=False)
    return ids, fns


def compute_identity_blocks(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ids.ndim != 1:
        raise RuntimeError(f"Expected ids shape [N], got {ids.shape}")
    n_rows = ids.shape[0]
    if n_rows == 0:
        raise RuntimeError("No rows to process.")

    changes = np.where(ids[1:] != ids[:-1])[0] + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), changes.astype(np.int64)])
    ends = np.concatenate([changes.astype(np.int64), np.array([n_rows], dtype=np.int64)])

    block_ids = ids[starts]
    unique_blocks = np.unique(block_ids).shape[0]
    if unique_blocks != starts.shape[0]:
        raise RuntimeError(
            "ids are not contiguous by identity in selected slice. "
            "Expected one contiguous block per identity."
        )
    return starts, ends


class GlintSliceDataset(Dataset):
    def __init__(
        self,
        identities: Sequence[str],
        filenames: Sequence[str],
        images_root: Path,
        preprocess_pil: Callable[[Image.Image], torch.Tensor],
        start_index: int,
    ) -> None:
        if len(identities) != len(filenames):
            raise RuntimeError(
                f"Identity/filename length mismatch: {len(identities)} vs {len(filenames)}"
            )
        self.identities = identities
        self.filenames = filenames
        self.images_root = images_root
        self.preprocess_pil = preprocess_pil
        self.start_index = start_index

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        identity = self.identities[idx]
        suffix = self.filenames[idx]
        path = glint_image_path(self.images_root, identity, suffix)
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing image file at row={self.start_index + idx}: {path}"
            )
        with Image.open(path) as img:
            x = self.preprocess_pil(img.convert("RGB"))
        return x, idx


def multiprocessing_workers_available() -> bool:
    try:
        lock = mp.get_context("spawn").Lock()
        del lock
        return True
    except Exception:
        return False


def prepare_memmaps(
    output_dir: Path,
    n_rows: int,
    embed_dim: int,
    resume: bool,
) -> tuple[np.memmap, np.memmap, Path, Path, Path]:
    templates_mm_path = output_dir / "templates.f32.memmap"
    norms_mm_path = output_dir / "norms_flat.f32.memmap"
    progress_path = output_dir / "progress.json"

    expected_templates_bytes = n_rows * embed_dim * np.dtype(np.float32).itemsize
    expected_norms_bytes = n_rows * np.dtype(np.float32).itemsize

    if resume and templates_mm_path.exists() and norms_mm_path.exists():
        if templates_mm_path.stat().st_size != expected_templates_bytes:
            raise RuntimeError(
                f"templates memmap size mismatch at {templates_mm_path}. "
                "Delete output dir or run with --no-resume."
            )
        if norms_mm_path.stat().st_size != expected_norms_bytes:
            raise RuntimeError(
                f"norms memmap size mismatch at {norms_mm_path}. "
                "Delete output dir or run with --no-resume."
            )
        templates_mm = np.memmap(
            templates_mm_path, mode="r+", dtype=np.float32, shape=(n_rows, embed_dim)
        )
        norms_mm = np.memmap(norms_mm_path, mode="r+", dtype=np.float32, shape=(n_rows,))
    else:
        templates_mm = np.memmap(
            templates_mm_path, mode="w+", dtype=np.float32, shape=(n_rows, embed_dim)
        )
        norms_mm = np.memmap(norms_mm_path, mode="w+", dtype=np.float32, shape=(n_rows,))

    return templates_mm, norms_mm, templates_mm_path, norms_mm_path, progress_path


@torch.no_grad()
def run_extraction(
    model_name: str,
    adapter: ModelAdapter,
    model: torch.nn.Module,
    identities: Sequence[str],
    filenames: Sequence[str],
    images_root: Path,
    templates_mm: np.memmap,
    norms_mm: np.memmap,
    start_offset: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    save_every: int,
    progress_path: Path,
    progress_seed: dict,
) -> int:
    if start_offset < 0 or start_offset > len(filenames):
        raise RuntimeError(
            f"Invalid resume offset {start_offset} for total rows {len(filenames)}"
        )

    effective_workers = num_workers
    if effective_workers > 0 and not multiprocessing_workers_available():
        print(
            "Warning: DataLoader multiprocessing is unavailable in this environment. "
            "Falling back to --num-workers 0."
        )
        effective_workers = 0

    sub_ids = identities[start_offset:]
    sub_filenames = filenames[start_offset:]
    dataset = GlintSliceDataset(
        identities=sub_ids,
        filenames=sub_filenames,
        images_root=images_root,
        preprocess_pil=adapter.preprocess_pil,
        start_index=start_offset,
    )

    def save_progress(next_index: int) -> None:
        payload = dict(progress_seed)
        payload["next_index"] = int(next_index)
        payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        atomic_json_dump(progress_path, payload)

    def iterate(active_workers: int) -> int:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=active_workers,
            pin_memory=(device.type == "cuda"),
        )
        next_index = start_offset
        batch_idx = 0
        total_rows = len(filenames)
        with tqdm(
            total=total_rows,
            initial=start_offset,
            desc=f"Extracting {model_name}",
            unit="img",
            dynamic_ncols=True,
            smoothing=0.05,
        ) as pbar:
            for images, relative_idx in loader:
                images = images.to(device, non_blocking=True)
                emb, out_norms = adapter.encode_batch(model, images)
                if emb.ndim != 2 or emb.shape[1] != adapter.embed_dim:
                    raise RuntimeError(
                        f"Unexpected embedding shape {tuple(emb.shape)} for model {model_name}"
                    )

                emb_np = emb.detach().cpu().numpy().astype(np.float32, copy=False)
                norm_np = out_norms.detach().cpu().numpy().astype(np.float32, copy=False).reshape(-1)
                rel_np = relative_idx.detach().cpu().numpy().astype(np.int64)
                abs_np = rel_np + start_offset
                if norm_np.shape[0] != abs_np.shape[0]:
                    raise RuntimeError(
                        f"Norm length {norm_np.shape[0]} does not match batch size {abs_np.shape[0]}"
                    )

                templates_mm[abs_np, :] = emb_np
                norms_mm[abs_np] = norm_np

                batch_count = int(abs_np.shape[0])
                pbar.update(batch_count)
                next_index = int(abs_np.max()) + 1
                batch_idx += 1
                if batch_idx % save_every == 0:
                    templates_mm.flush()
                    norms_mm.flush()
                    save_progress(next_index)
                    pbar.set_postfix_str(f"saved@{next_index}", refresh=False)

        templates_mm.flush()
        norms_mm.flush()
        save_progress(next_index)
        return next_index

    try:
        return iterate(effective_workers)
    except PermissionError as exc:
        if effective_workers > 0:
            print(
                "Warning: DataLoader multiprocessing failed "
                f"({exc}). Falling back to --num-workers 0."
            )
            return iterate(0)
        raise


def finalize_outputs(
    output_dir: Path,
    templates_mm_path: Path,
    norms_mm_path: Path,
    n_rows: int,
    embed_dim: int,
    block_starts: np.ndarray,
    block_ends: np.ndarray,
    metadata: dict,
) -> None:
    templates_pth = output_dir / "templates.pth"
    norms_pth = output_dir / "norms.pth"

    templates_mm = np.memmap(
        templates_mm_path, mode="r", dtype=np.float32, shape=(n_rows, embed_dim)
    )
    # Copy memmap into a regular tensor in chunks to avoid SIGBUS on large files.
    templates_tensor = torch.empty(n_rows, embed_dim, dtype=torch.float32)
    chunk = 1_000_000
    for i in range(0, n_rows, chunk):
        end = min(i + chunk, n_rows)
        templates_tensor[i:end] = torch.from_numpy(np.array(templates_mm[i:end]))
    del templates_mm
    torch.save(templates_tensor, templates_pth)
    del templates_tensor

    norms_mm = np.memmap(norms_mm_path, mode="r", dtype=np.float32, shape=(n_rows,))
    norms_list: list[torch.Tensor] = []
    for s, e in zip(block_starts, block_ends):
        block = np.array(norms_mm[s:e], dtype=np.float32, copy=True).reshape(-1, 1)
        norms_list.append(torch.from_numpy(block))
    torch.save(norms_list, norms_pth)

    for path in (
        templates_mm_path,
        norms_mm_path,
        output_dir / "progress.json",
        output_dir / "metadata.json",
    ):
        if path.exists():
            path.unlink()


def main() -> int:
    args = parse_args()
    args.ids_and_fns = to_abs(args.ids_and_fns)
    args.glint360k_root = to_abs(args.glint360k_root)
    args.output_dir = to_abs(args.output_dir) if args.output_dir else None
    if args.device == "auto":
        args.device = detect_default_device()

    if args.batch_size <= 0:
        raise RuntimeError("--batch-size must be > 0")
    if args.num_workers < 0:
        raise RuntimeError("--num-workers must be >= 0")
    if args.save_every <= 0:
        raise RuntimeError("--save-every must be > 0")
    if not args.glint360k_root.is_dir():
        raise RuntimeError(f"images root not found: {args.glint360k_root}")

    adapter = get_model_adapter(args.model)
    ids, filenames = load_ids_and_fns(args.ids_and_fns, args.start_index, args.end_index)
    n_rows = len(filenames)
    block_starts, block_ends = compute_identity_blocks(ids)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "templates" / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_rows > 0:
        first_path = glint_image_path(args.glint360k_root, ids[0], filenames[0])
        if not first_path.is_file():
            raise RuntimeError(f"Image root/files mismatch: first row points to missing file {first_path}")

    device = torch.device(args.device)
    model = adapter.load_model(device)

    print(f"Model:               {args.model}")
    print(f"Images root:         {args.glint360k_root}")
    print(f"ids_and_fns:         {args.ids_and_fns}")
    print(f"Rows to process:     {n_rows}")
    print(f"Identities:          {block_starts.shape[0]}")
    print(f"Embedding dim:       {adapter.embed_dim}")
    print(f"Output dir:          {output_dir}")
    print(f"Device:              {args.device}")

    if args.validate_only:
        print("Validation passed (--validate-only).")
        return 0

    templates_mm, norms_mm, templates_mm_path, norms_mm_path, progress_path = prepare_memmaps(
        output_dir=output_dir,
        n_rows=n_rows,
        embed_dim=adapter.embed_dim,
        resume=args.resume,
    )

    progress_seed = {
        "version": 1,
        "model": args.model,
        "ids_and_fns": str(args.ids_and_fns),
        "glint360k_root": str(args.glint360k_root),
        "slice_start": int(args.start_index),
        "slice_end": int(args.start_index + n_rows),
        "n_rows": int(n_rows),
        "embed_dim": int(adapter.embed_dim),
        "device": args.device,
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    next_index = 0
    if args.resume and progress_path.exists():
        saved = json.loads(progress_path.read_text())
        if int(saved.get("n_rows", -1)) != n_rows or int(saved.get("embed_dim", -1)) != adapter.embed_dim:
            raise RuntimeError(
                f"Incompatible progress file at {progress_path}; "
                "n_rows/embed_dim mismatch. Remove output dir or run with --no-resume."
            )
        next_index = int(saved.get("next_index", 0))

    print(f"Resume index:        {next_index}")
    if next_index < n_rows:
        first_pending_path = glint_image_path(args.glint360k_root, ids[next_index], filenames[next_index])
        if not first_pending_path.is_file():
            raise RuntimeError(
                "Image root/files mismatch: first pending row points to missing file "
                f"{first_pending_path}"
            )
        next_index = run_extraction(
            model_name=args.model,
            adapter=adapter,
            model=model,
            identities=ids,
            filenames=filenames,
            images_root=args.glint360k_root,
            templates_mm=templates_mm,
            norms_mm=norms_mm,
            start_offset=next_index,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            save_every=args.save_every,
            progress_path=progress_path,
            progress_seed=progress_seed,
        )
    else:
        print("Extraction already complete according to progress file.")

    if next_index != n_rows:
        raise RuntimeError(f"Extraction incomplete: next_index={next_index}, expected={n_rows}")

    if args.skip_finalize:
        print("Skipping finalization (--skip-finalize).")
        return 0

    metadata = dict(progress_seed)
    metadata["next_index"] = int(next_index)
    finalize_outputs(
        output_dir=output_dir,
        templates_mm_path=templates_mm_path,
        norms_mm_path=norms_mm_path,
        n_rows=n_rows,
        embed_dim=adapter.embed_dim,
        block_starts=block_starts,
        block_ends=block_ends,
        metadata=metadata,
    )
    print(f"Wrote templates:     {output_dir / 'templates.pth'}")
    print(f"Wrote norms:         {output_dir / 'norms.pth'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
