#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Iterable, List


def parse_args():
    p = argparse.ArgumentParser(description="Create identity-split train/val lists.")
    p.add_argument("--root", required=True, type=str, help="Root directory with identity subfolders.")
    p.add_argument("--out", default="splits", type=str, help="Output directory for split files.")
    p.add_argument("--val-id-frac", default=0.01, type=float, help="Fraction of IDs to use for validation.")
    p.add_argument("--seed", default=42, type=int, help="Random seed.")
    p.add_argument("--exts", nargs="+", default=[".jpg", ".png"], help="File extensions to include.")
    p.add_argument("--max-images-per-id", type=int, default=None, help="Optional cap per identity.")
    return p.parse_args()


def list_ids(root: Path) -> List[Path]:
    return sorted([Path(d.path) for d in os.scandir(root) if d.is_dir()])


def iter_images(id_dir: Path, exts: List[str]) -> Iterable[Path]:
    exts = [e.lower() for e in exts]
    for path in id_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def main():
    args = parse_args()
    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ids = list_ids(root)
    if not ids:
        raise SystemExit(f"No identity folders found in {root}")

    rng = random.Random(args.seed)
    rng.shuffle(ids)

    val_count = max(1, int(len(ids) * args.val_id_frac))
    val_ids = sorted(ids[:val_count])
    train_ids = sorted(ids[val_count:])

    ids_train_path = out / "ids_train.txt"
    ids_val_path = out / "ids_val.txt"
    with ids_train_path.open("w") as f:
        for p in train_ids:
            f.write(p.name + "\n")
    with ids_val_path.open("w") as f:
        for p in val_ids:
            f.write(p.name + "\n")

    train_list_path = out / "train.txt"
    val_list_path = out / "val.txt"

    def write_list(id_list: List[Path], out_path: Path) -> int:
        count = 0
        with out_path.open("w") as f:
            for id_dir in id_list:
                img_paths = list(iter_images(id_dir, args.exts))
                if args.max_images_per_id:
                    rng.shuffle(img_paths)
                    img_paths = img_paths[: args.max_images_per_id]
                for img_path in img_paths:
                    rel = img_path.relative_to(root)
                    f.write(str(rel) + "\n")
                    count += 1
        return count

    train_count = write_list(train_ids, train_list_path)
    val_count_imgs = write_list(val_ids, val_list_path)

    print(f"IDs: train={len(train_ids)} val={len(val_ids)}")
    print(f"Images: train={train_count} val={val_count_imgs}")
    print(f"Wrote: {train_list_path} {val_list_path}")


if __name__ == "__main__":
    main()
