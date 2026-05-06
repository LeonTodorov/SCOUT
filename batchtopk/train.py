from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TextIO

import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from batchtopk import (
    BatchTopKTrainingSAE,
    build_batch_topk_training_sae,
    estimate_decoder_prescale,
)
from dataloader import VectorDataset
from paths import load_paths

EARLY_STOP_MIN_EPOCHS = 10
EARLY_STOP_MSE_THRESHOLD = 0.01


@dataclass(slots=True)
class TrainStepOutput:
    reconstruction: torch.Tensor
    features: torch.Tensor
    hidden_pre: torch.Tensor
    loss: torch.Tensor
    losses: dict[str, torch.Tensor]


def _calculate_topk_aux_acts(
    k_aux: int,
    hidden_pre: torch.Tensor,
    dead_neuron_mask: torch.Tensor,
) -> torch.Tensor:
    # Mask out living latents; pick top-k among dead.
    auxk_latents = torch.where(dead_neuron_mask[None], hidden_pre, -torch.inf)
    auxk_topk = auxk_latents.topk(k_aux, sorted=False)
    auxk_acts = torch.zeros_like(hidden_pre)
    auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)
    return auxk_acts


def batchtopk_step(
    model: BatchTopKTrainingSAE,
    batch: torch.Tensor,
    *,
    aux_coefficient: float = 1.0,
) -> TrainStepOutput:
    reconstruction, features, hidden_pre = model(batch)
    mse_loss = (reconstruction - batch).pow(2).mean()

    # Update global top-k threshold (EMA of minimum positive activation).
    model.update_topk_threshold(features)

    with torch.no_grad():
        alive_latents = features.gt(0).any(dim=0)
        dead_neuron_mask = ~alive_latents
        num_dead = int(dead_neuron_mask.sum().item())

    if num_dead == 0:
        aux_loss = features.new_tensor(0.0)
    else:
        residual = (batch - reconstruction).detach()
        k_aux = batch.shape[-1] // 2
        scale = min(num_dead / max(k_aux, 1), 1.0) if k_aux > 0 else 1.0
        k_aux_eff = min(k_aux, num_dead) if k_aux > 0 else 0
        if k_aux_eff == 0:
            aux_loss = features.new_tensor(0.0)
        else:
            auxk_acts = _calculate_topk_aux_acts(
                k_aux=k_aux_eff,
                hidden_pre=hidden_pre,
                dead_neuron_mask=dead_neuron_mask,
            )
            recons = model.decode(auxk_acts)
            auxk_loss = (recons - residual).pow(2).mean()
            aux_loss = aux_coefficient * scale * auxk_loss

    total_loss = mse_loss + aux_loss
    return TrainStepOutput(
        reconstruction=reconstruction,
        features=features,
        hidden_pre=hidden_pre,
        loss=total_loss,
        losses={"mse_loss": mse_loss, "aux_loss": aux_loss},
    )


def parse_args() -> argparse.Namespace:
    P = load_paths()
    parser = argparse.ArgumentParser(
        description="Minimal BatchTopK SAE training loop using VectorDataset and top-k auxiliary loss."
    )
    parser.add_argument(
        "--fr-templates",
        type=str,
        default=None,
        help="Path to a face-recognition template tensor (.pt/.pth), shape (num_samples, d_in). "
             "Defaults to <templates_root>/<encoder-id>/templates.pth when --encoder-id is provided.",
    )
    parser.add_argument(
        "--encoder-id",
        type=str,
        default=None,
        help="Optional encoder identifier; resolves --fr-templates from paths.yaml when set.",
    )
    parser.add_argument(
        "--latent-size",
        type=int,
        default=8192,
        help="Number of features (d_sae).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Samples per training batch.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of passes over the dataset.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate for Adam.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Torch device identifier, e.g. cpu or cuda:0.",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=32,
        help="BatchTopK: keeps int(k * batch_size) positive activations globally per batch.",
    )
    parser.add_argument(
        "--W_dec_prescale",
        type=str,
        default="auto",
        help="Scalar to divide W_dec by, or 'auto' to estimate it at startup.",
    )
    parser.add_argument(
        "--W_prescale",
        type=float,
        default=1.0,
        help="Scalar to divide W_enc and W_dec by to account for model gain.",
    )
    parser.add_argument(
        "--aux-coefficient",
        type=float,
        default=1.0,
        help="Coefficient for the auxiliary reconstruction loss (dead-neuron revival).",
    )
    parser.add_argument(
        "--topk-threshold-lr",
        type=float,
        default=0.01,
        help="EMA learning rate for the global min-positive top-k threshold.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile with the inductor backend for the model.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the trained model state_dict (directory created if needed).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Optional path to a log file where epoch metrics will be written.",
    )
    parser.add_argument(
        "--save-every-epoch",
        action="store_true",
        help="Append to the log file and save model weights after each epoch.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    args = parser.parse_args()
    if args.fr_templates is None:
        if not args.encoder_id:
            parser.error("Either --fr-templates or --encoder-id is required.")
        args.fr_templates = str(P["templates_root"] / args.encoder_id / "templates.pth")
    return args


def maybe_open_log_file(path: str | None, append: bool) -> TextIO | None:
    if path is None:
        return None
    log_path = Path(path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    return log_path.open(mode)


def log_line(log_file: TextIO | None, message: str) -> None:
    if log_file is None:
        return
    log_file.write(message + "\n")
    log_file.flush()


def apply_weight_prescale(
    model: torch.nn.Module,
    w_dec_prescale: str,
    w_prescale: float,
    *,
    prescale_batch_size: int,
) -> float | None:
    w_dec_prescale = w_dec_prescale.strip().lower()

    if w_prescale != 1:
        if w_dec_prescale == "auto":
            raise ValueError(
                "--W_prescale cannot be combined with the default --W_dec_prescale auto. "
                "Set --W_dec_prescale 1.0 to use full-weight prescaling."
            )
        explicit_w_dec_prescale = float(w_dec_prescale)
        if explicit_w_dec_prescale != 1:
            raise ValueError("--W_prescale and non-1 --W_dec_prescale are ambiguous.")
        with torch.no_grad():
            model.W_enc /= w_prescale
            model.W_dec /= w_prescale
        return None

    if w_dec_prescale == "auto":
        resolved_prescale = estimate_decoder_prescale(
            model,
            batch_size=prescale_batch_size,
        )
    else:
        resolved_prescale = float(w_dec_prescale)

    if resolved_prescale <= 0:
        raise ValueError(f"--W_dec_prescale must be positive, got {resolved_prescale}.")

    if resolved_prescale != 1:
        with torch.no_grad():
            model.W_dec /= resolved_prescale
    return resolved_prescale


def build_lr_schedule(
    optimizer: torch.optim.Optimizer,
    total_training_steps: int,
) -> lr_scheduler.LambdaLR:
    decay_steps = max(total_training_steps // 5, 1)
    decay_start = max(total_training_steps - decay_steps, 0)

    def lr_lambda(step: int) -> float:
        if step < decay_start:
            return 1.0
        remaining = max(decay_steps - (step - decay_start), 0)
        return remaining / decay_steps

    return lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def build_epoch_output_path(base_path: Path, epoch_index: int) -> Path:
    if base_path.suffix:
        return base_path.with_name(f"{base_path.stem}_epoch{epoch_index:04d}{base_path.suffix}")
    return base_path.with_name(f"{base_path.name}_epoch{epoch_index:04d}")


def save_model_payload(
    model: torch.nn.Module,
    destination: Path,
    log_file: TextIO | None,
    label: str | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {"state_dict": model.state_dict()}
    if hasattr(model, "cfg"):
        payload["config"] = asdict(model.cfg)
    torch.save(payload, destination)
    if label:
        message = f"Saved {label} model to {destination.resolve()}"
    else:
        message = f"Saved model to {destination.resolve()}"
    print(message)
    log_line(log_file, message)


def model_parameters_changed(
    initial_state: dict[str, torch.Tensor],
    current_state: dict[str, torch.Tensor],
) -> bool:
    for name, current_param in current_state.items():
        init_param = initial_state.get(name)
        if init_param is None:
            return True
        if not torch.allclose(current_param.detach().cpu(), init_param, rtol=1e-6, atol=1e-6):
            return True
    return False


def main() -> None:
    args = parse_args()

    if args.save_every_epoch and not args.output:
        raise ValueError("--save-every-epoch requires --output to be specified.")

    dataset = VectorDataset(args.fr_templates)
    if len(dataset) == 0:
        raise ValueError("VectorDataset is empty; ensure the source file contains data.")

    output_base_path = Path(args.output) if args.output else None
    if output_base_path:
        output_base_path.parent.mkdir(parents=True, exist_ok=True)

    sample_dim = dataset[0].numel()
    device = torch.device(args.device)
    log_file = maybe_open_log_file(args.log_file, append=args.save_every_epoch)

    try:
        hyper_summary = (
            f"Training config: d_in={sample_dim}, d_latent={args.latent_size}, "
            f"k={args.k}, W_dec_prescale={args.W_dec_prescale}, lr={args.lr}, "
            f"batch_size={args.batch_size}, epochs={args.epochs}"
        )
        print(hyper_summary)
        log_line(log_file, hyper_summary)

        model = build_batch_topk_training_sae(
            d_in=sample_dim,
            d_sae=args.latent_size,
            device=args.device,
            k=args.k,
            topk_threshold_lr=args.topk_threshold_lr,
        ).to(device)
        resolved_w_dec_prescale = apply_weight_prescale(
            model,
            args.W_dec_prescale,
            args.W_prescale,
            prescale_batch_size=args.batch_size,
        )
        if resolved_w_dec_prescale is not None:
            prescale_msg = f"Resolved W_dec_prescale={resolved_w_dec_prescale:.6f}"
            print(prescale_msg)
            log_line(log_file, prescale_msg)
        elif args.W_prescale != 1:
            prescale_msg = f"Applied W_prescale={args.W_prescale:.6f} to W_enc and W_dec"
            print(prescale_msg)
            log_line(log_file, prescale_msg)

        uncompiled_model = model
        initial_state = {
            name: param.detach().cpu().clone()
            for name, param in uncompiled_model.state_dict().items()
        }

        if args.compile:
            if not hasattr(torch, "compile"):
                raise RuntimeError("torch.compile is unavailable in this PyTorch build.")
            model = torch.compile(model, backend="inductor")
            torch.set_float32_matmul_precision("high")

        target_model = getattr(uncompiled_model, "module", uncompiled_model)
        model_dtype = target_model.W_enc.dtype

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=device.type == "cuda",
            num_workers=args.num_workers,
        )
        total_training_steps = max(len(data_loader) * args.epochs, 1)
        lr_schedule = build_lr_schedule(optimizer, total_training_steps)

        last_epoch_metrics: dict[str, float] | None = None

        for epoch in range(args.epochs):
            epoch_start = time.perf_counter()
            epoch_loss = 0.0
            epoch_mse = 0.0
            epoch_aux = 0.0
            epoch_active_sum = 0.0
            total_samples = 0
            alive_latents_epoch = torch.zeros(
                target_model.W_enc.shape[1], dtype=torch.bool, device=device
            )

            progress = tqdm(
                data_loader,
                desc=f"Epoch {epoch + 1:3d}/{args.epochs:3d}",
                leave=False,
            )
            for batch in progress:
                batch = batch.to(device=device, dtype=model_dtype)
                optimizer.zero_grad(set_to_none=True)

                step_output = batchtopk_step(
                    model,
                    batch,
                    aux_coefficient=args.aux_coefficient,
                )
                step_output.loss.backward()
                optimizer.step()
                lr_schedule.step()

                batch_size = batch.size(0)
                total_samples += batch_size
                epoch_loss += step_output.loss.detach().item() * batch_size
                epoch_mse += step_output.losses["mse_loss"].detach().item() * batch_size
                epoch_aux += step_output.losses["aux_loss"].detach().item() * batch_size
                batch_active_counts = step_output.features.gt(0).sum(dim=1)
                epoch_active_sum += batch_active_counts.float().sum().item()
                alive_latents_epoch |= step_output.features.gt(0).any(dim=0)

                avg_loss = epoch_loss / total_samples
                avg_mse = epoch_mse / total_samples
                avg_aux = epoch_aux / total_samples
                avg_l0_mean = epoch_active_sum / total_samples
                progress.set_postfix(
                    loss=f"{avg_loss:.4f}",
                    mse=f"{avg_mse:.4f}",
                    aux=f"{avg_aux:.4f}",
                    mean_l0=f"{avg_l0_mean:.2f}",
                    thr=f"{float(target_model.topk_threshold):.4f}",
                )

            if total_samples == 0:
                raise RuntimeError("No samples were processed during training.")

            alive_count = int(alive_latents_epoch.sum().item())
            alive_fraction = alive_count / target_model.cfg.d_sae
            epoch_duration = time.perf_counter() - epoch_start
            epoch_summary = (
                f"Epoch {epoch + 1:3d}/{args.epochs:3d} "
                f"loss={epoch_loss / total_samples:.6f} "
                f"mse={epoch_mse / total_samples:.6f} "
                f"aux={epoch_aux / total_samples:.6f} "
                f"mean_l0={epoch_active_sum / total_samples:.2f} "
                f"thr={float(target_model.topk_threshold):.6f} "
                f"alive={alive_count}/{target_model.cfg.d_sae} ({alive_fraction:.3f}) "
                f"time={epoch_duration:.2f}s"
            )
            print(epoch_summary)
            log_line(log_file, epoch_summary)

            last_epoch_metrics = {
                "loss": epoch_loss / total_samples,
                "mse": epoch_mse / total_samples,
                "aux": epoch_aux / total_samples,
                "mean_l0": epoch_active_sum / total_samples,
                "alive_count": float(alive_count),
                "alive_fraction": alive_fraction,
            }

            if args.save_every_epoch and output_base_path:
                epoch_index = epoch + 1
                epoch_output_path = build_epoch_output_path(output_base_path, epoch_index)
                save_model_payload(
                    target_model,
                    epoch_output_path,
                    log_file=log_file,
                    label=f"epoch {epoch_index:04d}",
                )

            if (epoch + 1) >= EARLY_STOP_MIN_EPOCHS and (epoch_mse / total_samples) >= EARLY_STOP_MSE_THRESHOLD:
                early_stop_msg = (
                    f"Early stop at epoch {epoch + 1}: MSE {epoch_mse / total_samples:.6f} "
                    f">= threshold {EARLY_STOP_MSE_THRESHOLD:.6f}"
                )
                print(early_stop_msg)
                log_line(log_file, early_stop_msg)
                break

        if output_base_path:
            if not model_parameters_changed(initial_state, target_model.state_dict()):
                warning_msg = (
                    "Warning: model parameters appear unchanged from initialization; "
                    "did training run?"
                )
                print(warning_msg)
                log_line(log_file, warning_msg)
            save_model_payload(target_model, output_base_path, log_file=log_file)

        if last_epoch_metrics is not None:
            final_summary = (
                f"Final summary: loss={last_epoch_metrics['loss']:.6f}, "
                f"mse={last_epoch_metrics['mse']:.6f}, "
                f"aux={last_epoch_metrics['aux']:.6f}, "
                f"mean_l0={last_epoch_metrics['mean_l0']:.2f}, "
                f"alive={int(last_epoch_metrics['alive_count'])}/{target_model.cfg.d_sae} "
                f"({last_epoch_metrics['alive_fraction']:.3f})"
            )
            print(final_summary)
            log_line(log_file, final_summary)
    finally:
        if log_file:
            log_file.close()


if __name__ == "__main__":
    main()
