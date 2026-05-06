#!/usr/bin/env python3
"""Train a deep face decoder against one of the four paper-named encoders.

Encoder choices (paper naming):
  - adaface_ms1mv2_ires50
  - adaface_webface4m_vitb
  - arcface_webface42m_res100
  - swinface_ms1mv2_swint

Each encoder is paired with its native intermediate-feature perceptual loss
(LayerNorm-CHW normalized, eps=1e-6).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import sys
import traceback
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from dataset import IndexedLineDataset
from decoder import Decoder
from encoders import ENCODER_NAMES, build_encoder
from perceptual import bgr_to_rgb01


SAMPLE_GRID_ROWS = 8
SAMPLE_GRID_COLS = 8


def build_perceptual(name: str, args, device: torch.device) -> torch.nn.Module:
    """Build the encoder-native perceptual loss (LayerNorm-CHW, eps=1e-6)."""
    from perceptual import (
        AdaFaceFeatureExtractor,
        AdaFaceViTPerceptualLoss,
        Arc2FaceArcFacePerceptualLoss,
        SwinFacePerceptualLoss,
    )

    if name == "adaface_ms1mv2_ires50":
        return AdaFaceFeatureExtractor(device=device)
    if name == "adaface_webface4m_vitb":
        return AdaFaceViTPerceptualLoss(
            upsample=args.vit_upsample,
            upsample_size=args.output_size,
            device=device,
        )
    if name == "arcface_webface42m_res100":
        return Arc2FaceArcFacePerceptualLoss(device=device)
    if name == "swinface_ms1mv2_swint":
        return SwinFacePerceptualLoss(device=device)
    raise ValueError(f"Unknown encoder: {name}")


def _load_yaml_config(config_path: str) -> Dict:
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {path} must be a YAML mapping/object.")
    return cfg


def parse_args():
    p = argparse.ArgumentParser(description="Train a deep face decoder (PyTorch).")
    p.add_argument("--config", default=None, type=str,
                   help="Path to YAML config file. CLI args override YAML values.")
    p.add_argument("--glint360k-root", default=None, type=str,
                   help="Root directory of unpacked Glint360K images. "
                        "Defaults from paths.yaml:glint360k_root.")
    p.add_argument("--train-list", default="splits/train.txt", type=str)
    p.add_argument("--val-list", default="splits/val.txt", type=str)
    p.add_argument("--output-dir", default="runs", type=str)
    p.add_argument("--run-id", default="", type=str,
                   help="Descriptive run ID appended to timestamp.")
    p.add_argument("--encoder-id", default="adaface_ms1mv2_ires50",
                   choices=list(ENCODER_NAMES),
                   help="Face-recognition encoder identifier (paper naming).")
    p.add_argument("--batch-size", default=16, type=int)
    p.add_argument("--lr", default=1e-3, type=float)
    p.add_argument("--epochs", default=10, type=int)
    p.add_argument("--val-every", default=10000, type=int)
    p.add_argument("--val-batches", default=256, type=int)

    p.add_argument("--vit-upsample", default="bilinear",
                   choices=["none", "bilinear"],
                   help="Upsample ViT perceptual features (adaface_webface4m_vitb only).")

    p.add_argument("--loss-weights", nargs=3, type=float, default=[1.0, 1.0, 1.0],
                   metavar=("PIX", "GRAD", "PERC"),
                   help="Weights for [pixel, gradient, perceptual] losses.")
    p.add_argument("--early-stop-patience", default=10, type=int,
                   help="Stop after N validation rounds without improvement (0 disables).")

    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--num-workers", default=4, type=int)
    p.add_argument("--device", default=None, type=str)
    p.add_argument("--log-every", default=100, type=int)
    p.add_argument("--output-size", default=112, type=int)

    pre_args, _ = p.parse_known_args()
    if pre_args.config:
        cfg = _load_yaml_config(pre_args.config)
        valid_keys = {action.dest for action in p._actions if action.dest != "help"}
        unknown_keys = sorted(k for k in cfg.keys() if k not in valid_keys)
        if unknown_keys:
            p.error(f"Unknown key(s) in --config: {', '.join(unknown_keys)}")
        p.set_defaults(**cfg)

    return p.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def gradient_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.mse_loss(pred_dx, target_dx) + F.mse_loss(pred_dy, target_dy)


def compute_losses(
    recon: torch.Tensor,
    target: torch.Tensor,
    perceptual_fn,
    loss_weights: list,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pixel = F.mse_loss(recon, target)
    grad = gradient_loss(recon, target)
    perc = perceptual_fn(recon, target)
    total = (
        loss_weights[0] * pixel
        + loss_weights[1] * grad
        + loss_weights[2] * perc
    )
    return total, pixel, grad, perc


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


class _Tee:
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log = log_file

    def write(self, data):
        self._stream.write(data)
        self._log.write(data)
        self._log.flush()

    def flush(self):
        self._stream.flush()
        self._log.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


def _setup_logging(log_path: Path):
    log_file = open(log_path, "a")  # noqa: SIM115
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)


def create_run_dir(base_dir: str, run_id: str = "") -> Path:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{ts}_{run_id}" if run_id else ts
    run_dir = Path(base_dir) / name
    ensure_dir(run_dir)
    ensure_dir(run_dir / "checkpoints")
    ensure_dir(run_dir / "samples")
    ensure_dir(run_dir / "tb")
    return run_dir


def save_checkpoint(path: Path, decoder: Decoder, optimizer: torch.optim.Optimizer, step: int, best: float, config: Dict):
    torch.save({
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best": best,
        "config": config,
    }, path)


def build_fixed_sample_indices(dataset_len: int, n_samples: int, seed: int = 123) -> list[int]:
    rng = random.Random(seed)
    if dataset_len <= n_samples:
        return list(range(dataset_len))
    step = dataset_len / n_samples
    indices = [int(i * step) for i in range(n_samples)]
    indices = [min(max(0, idx + rng.randint(-5, 5)), dataset_len - 1) for idx in indices]
    return indices


def save_recon_grid(path: Path, target: torch.Tensor, recon: torch.Tensor, ncols: int = SAMPLE_GRID_COLS):
    from torchvision.utils import save_image

    n = target.shape[0]
    target_rgb = bgr_to_rgb01(target)
    recon_rgb = bgr_to_rgb01(recon)

    rows = []
    for start in range(0, n, ncols):
        end = min(start + ncols, n)
        rows.append(target_rgb[start:end])
        rows.append(recon_rgb[start:end])
    grid = torch.cat(rows, dim=0)
    save_image(grid, path, nrow=ncols)


def collect_fixed_samples(dataset: IndexedLineDataset, indices: list[int], device: torch.device) -> torch.Tensor:
    tensors = [dataset[idx] for idx in indices]
    return torch.stack(tensors).to(device)


def run_validation(
    encoder, decoder, perceptual, val_loader, device,
    max_batches: int, loss_weights,
    amp: bool = False, lpips_metric=None,
) -> Dict[str, float]:
    from metrics import psnr, ssim

    sums: Dict[str, float] = {}
    count = 0
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= max_batches:
                break
            batch = batch.to(device, non_blocking=True)
            emb = encoder(batch)
            with torch.amp.autocast("cuda", enabled=amp):
                recon = decoder(emb)
                total, pixel, grad, perc = compute_losses(recon, batch, perceptual, loss_weights)

            for key, val in [("total", total), ("pixel", pixel), ("grad", grad), ("perc", perc)]:
                sums[key] = sums.get(key, 0.0) + float(val.item())
            count += 1

            recon_f = recon.float()
            batch_f = batch.float()
            psnr_vals.append(psnr(recon_f, batch_f).mean().item())
            ssim_vals.append(ssim(recon_f, batch_f).mean().item())
            if lpips_metric is not None:
                lpips_vals.append(lpips_metric.mean(recon_f, batch_f))

    if count == 0:
        return {k: float("nan") for k in
                ["total", "pixel", "grad", "perc", "psnr", "ssim", "lpips"]}

    result = {k: sums[k] / count for k in sums}
    result["psnr"] = float(np.mean(psnr_vals))
    result["ssim"] = float(np.mean(ssim_vals))
    result["lpips"] = float(np.mean(lpips_vals)) if lpips_vals else float("nan")
    return result


def _log_csv(path: Path, row: list):
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(row)


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, indent=2, sort_keys=True)


def main():
    args = parse_args()
    if args.glint360k_root is None:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from paths import load_paths  # noqa: E402
        args.glint360k_root = str(load_paths()["glint360k_root"])
    set_seed(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dir = create_run_dir(args.output_dir, args.run_id)
    _setup_logging(run_dir / "train.log")

    config = vars(args)
    (run_dir / "config.json").write_text(json_dumps(config))

    missing_log = run_dir / "missing_files.log"
    train_ds = IndexedLineDataset(args.train_list, args.glint360k_root, image_size=args.output_size, missing_log=missing_log)
    val_ds = IndexedLineDataset(args.val_list, args.glint360k_root, image_size=args.output_size, missing_log=missing_log)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)

    encoder = build_encoder(args.encoder_id, device)
    decoder = Decoder(output_size=args.output_size).to(device)
    perceptual = build_perceptual(args.encoder_id, args, device)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")
    steps_per_epoch = len(train_loader)
    if steps_per_epoch <= 0:
        raise ValueError("Training loader has zero batches; increase data size or reduce batch size.")
    max_steps = args.epochs * steps_per_epoch

    n_grid = SAMPLE_GRID_ROWS * SAMPLE_GRID_COLS
    sample_indices = build_fixed_sample_indices(len(val_ds), n_grid)

    from metrics import LPIPSMetric
    lpips_metric = LPIPSMetric(device=device)

    print(f"Device: {device}")
    print(f"Encoder: {args.encoder_id} (perceptual: native intermediate features, LayerNorm-CHW)")
    print(f"Decoder params: {sum(p.numel() for p in decoder.parameters()):,}")
    print(f"Loss weights (pix/grad/perc): {args.loss_weights}")
    if args.early_stop_patience > 0:
        print(f"Early stopping: patience={args.early_stop_patience} val rounds")
    print(f"Sample grid: {SAMPLE_GRID_ROWS}x{SAMPLE_GRID_COLS} ({n_grid} identities)")
    print(f"Epochs: {args.epochs}, steps/epoch: {steps_per_epoch}, total steps: {max_steps}, batch size: {args.batch_size}")
    print(f"Run dir: {run_dir}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    except Exception:
        writer = None

    csv_header = ["step", "split", "total", "pixel", "gradient", "perceptual",
                  "psnr", "ssim", "lpips"]
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        csv.writer(f).writerow(csv_header)

    best_val = float("inf")
    patience_counter = 0
    train_iter = iter(train_loader)

    for step in range(1, max_steps + 1):
        decoder.train()
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            emb = encoder(batch)

        with torch.amp.autocast("cuda", enabled=args.amp):
            recon = decoder(emb)
            total, pixel, grad, perc = compute_losses(recon, batch, perceptual, args.loss_weights)

        if args.amp:
            scaler.scale(total).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total.backward()
            optimizer.step()

        if step % args.log_every == 0 or step == 1:
            row = [step, "train", total.item(), pixel.item(), grad.item(), perc.item(), "", "", ""]
            _log_csv(metrics_path, row)
            print(f"[step {step}] total={total.item():.4f}  pix={pixel.item():.4f}  grad={grad.item():.4f}  perc={perc.item():.4f}", flush=True)
            if writer:
                writer.add_scalar("train/total", total.item(), step)
                writer.add_scalar("train/pixel", pixel.item(), step)
                writer.add_scalar("train/gradient", grad.item(), step)
                writer.add_scalar("train/perceptual", perc.item(), step)

        if step % args.val_every == 0 or step == max_steps:
            save_checkpoint(run_dir / "checkpoints" / f"step_{step}.pt", decoder, optimizer, step, best_val, config)
            save_checkpoint(run_dir / "checkpoints" / "last.pt", decoder, optimizer, step, best_val, config)
            decoder.eval()
            val_result = run_validation(
                encoder, decoder, perceptual, val_loader, device,
                args.val_batches, args.loss_weights,
                amp=args.amp, lpips_metric=lpips_metric,
            )
            val_total = val_result["total"]

            row = [step, "val", val_result["total"], val_result["pixel"],
                   val_result["grad"], val_result["perc"],
                   val_result["psnr"], val_result["ssim"], val_result["lpips"]]
            _log_csv(metrics_path, row)

            if writer:
                for tb_name, key in [("total", "total"), ("pixel", "pixel"),
                                     ("gradient", "grad"), ("perceptual", "perc")]:
                    writer.add_scalar(f"val/{tb_name}", val_result[key], step)
                for k in ["psnr", "ssim", "lpips"]:
                    writer.add_scalar(f"quality/{k}", val_result[k], step)

            print(f"[step {step}] val: total={val_total:.4f}  "
                  f"PSNR={val_result['psnr']:.2f}dB  SSIM={val_result['ssim']:.4f}  "
                  f"LPIPS={val_result['lpips']:.4f}", flush=True)

            try:
                sample_batch = collect_fixed_samples(val_ds, sample_indices, device)
                with torch.no_grad():
                    sample_emb = encoder(sample_batch)
                    sample_recon = decoder(sample_emb)
                save_recon_grid(
                    run_dir / "samples" / f"step_{step}.png",
                    sample_batch, sample_recon, ncols=SAMPLE_GRID_COLS,
                )
            except Exception as e:
                print(f"  Warning: sample grid failed: {e}", flush=True)
                traceback.print_exc()

            if val_total < best_val:
                best_val = val_total
                patience_counter = 0
                save_checkpoint(run_dir / "checkpoints" / "best.pt", decoder, optimizer, step, best_val, config)
                print(f"[step {step}] New best val_total={val_total:.6f}")
            else:
                patience_counter += 1
                print(f"[step {step}] val_total={val_total:.6f} (no improve {patience_counter}/{args.early_stop_patience})")

            if args.early_stop_patience > 0 and patience_counter >= args.early_stop_patience:
                print(f"Early stopping at step {step} (no improvement for {patience_counter} val rounds)")
                save_checkpoint(run_dir / "checkpoints" / "last.pt", decoder, optimizer, step, best_val, config)
                break

    if writer:
        writer.close()


if __name__ == "__main__":
    main()
