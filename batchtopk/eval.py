from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from batchtopk import JumpReLUInferenceSAE
from dataloader import VectorDataset
from utils import load_model
from paths import load_paths

DEFAULT_DECODER_COSINE_PAIRS = 1_000_000
DECODER_COSINE_CHUNK_SIZE = 16_384


def infer_label(
    label: str | None,
    *,
    output_dir: Path,
    data_path: Path,
    weights_path: Path,
) -> str:
    if label is not None and label.strip():
        return label.strip()
    if output_dir.name and output_dir.name != "eval_results":
        return output_dir.name
    if data_path.parent.name:
        return data_path.parent.name
    if weights_path.parent.name:
        return weights_path.parent.name
    return weights_path.stem


def compute_decoder_mean_abs_cosine(
    model: JumpReLUInferenceSAE,
    num_pairs: int = DEFAULT_DECODER_COSINE_PAIRS,
) -> float:
    num_latents = model.W_dec.shape[0]
    if num_latents < 2 or num_pairs <= 0:
        return float("nan")

    sample_target = min(num_pairs, num_latents * (num_latents - 1))
    if sample_target <= 0:
        return float("nan")

    with torch.no_grad():
        dictionary_vectors = F.normalize(model.W_dec.detach().to(dtype=torch.float32), dim=1, eps=1e-12)
        total_abs_cos = 0.0
        processed = 0
        while processed < sample_target:
            chunk = min(DECODER_COSINE_CHUNK_SIZE, sample_target - processed)
            idx_i = torch.randint(0, num_latents, (chunk,), device=dictionary_vectors.device)
            idx_j = torch.randint(0, num_latents - 1, (chunk,), device=dictionary_vectors.device)
            idx_j = idx_j + (idx_j >= idx_i).to(idx_j.dtype)
            chunk_cos = (dictionary_vectors[idx_i] * dictionary_vectors[idx_j]).sum(dim=1).abs()
            total_abs_cos += chunk_cos.sum().item()
            processed += chunk
    return total_abs_cos / sample_target


def save_activation_histogram(
    fire_counts: torch.Tensor,
    total_samples: int,
    output_path: Path,
    *,
    title: str | None = None,
) -> None:
    if total_samples <= 0:
        raise ValueError("total_samples must be positive.")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = fire_counts.detach().cpu().numpy().astype(np.float64)
    alive_counts = counts[counts > 0]
    dead_count = int((counts == 0).sum())
    min_log = np.log10(1.0 / total_samples)
    bins = np.linspace(min_log, 0.0, 50)

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    if len(alive_counts) > 0:
        ax.hist(
            np.log10(alive_counts / total_samples),
            bins=bins,
            color="#4878a8",
            edgecolor="none",
            alpha=0.8,
            label="Alive features",
        )
    if dead_count > 0:
        ax.bar(
            min_log - 0.4,
            dead_count,
            width=0.3,
            color="#c44e52",
            edgecolor="none",
            alpha=0.85,
            label=f"Dead ({dead_count})",
        )

    ax.set_xlabel(r"$\log_{10}$(firing frequency)")
    ax.set_ylabel("Number of features")
    ax.set_title(title or "SAE feature activation density")
    ax.set_xlim(min_log - 0.8, 0.15)
    ax.grid(True, alpha=0.15, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", framealpha=0.85, edgecolor="none")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_checkpoint(
    *,
    weights_path: Path,
    data_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    decoder_cosine_pairs: int,
    no_progress: bool,
    label: str | None = None,
) -> dict[str, Any]:
    if not weights_path.exists():
        raise FileNotFoundError(f"Could not find weights file: {weights_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset file: {data_path}")

    dataset = VectorDataset(str(data_path))
    if len(dataset) == 0:
        raise ValueError("Loaded dataset is empty; cannot run inference.")
    eval_label = infer_label(
        label,
        output_dir=output_dir,
        data_path=data_path,
        weights_path=weights_path,
    )

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=num_workers,
    )

    model = load_model(weights_path, device)
    model.eval()

    total_samples = 0
    mse_sum = 0.0
    reconstruction_cosine_sum = 0.0
    alive_tracker = torch.zeros(model.d_sae, dtype=torch.bool, device=device)
    activation_counts = torch.zeros(model.d_sae, dtype=torch.long, device=device)

    with torch.no_grad():
        progress = tqdm(data_loader, desc=eval_label) if not no_progress else data_loader
        for batch in progress:
            batch = batch.to(device=device, dtype=model.W_enc.dtype)
            recon, features, _ = model(batch)
            activated = features.gt(0)

            batch_size_actual = batch.size(0)
            total_samples += batch_size_actual
            mse_sum += torch.mean((batch - recon) ** 2).item() * batch_size_actual
            reconstruction_cosine_sum += F.cosine_similarity(
                recon.float(), batch.float(), dim=1
            ).sum().item()
            alive_tracker |= activated.any(dim=0)
            activation_counts += activated.to(torch.long).sum(dim=0)

            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    mse=f"{mse_sum / total_samples:.6f}",
                    alive=f"{int(alive_tracker.sum().item())}",
                )

    if total_samples == 0:
        raise RuntimeError("No samples were processed during inference.")

    alive_count = int(alive_tracker.sum().item())
    alive_pct = 100.0 * alive_count / model.d_sae
    metrics: dict[str, Any] = {
        "label": eval_label,
        "weights_path": str(weights_path),
        "data_path": str(data_path),
        "total_samples": total_samples,
        "d_in": model.d_in,
        "d_sae": model.d_sae,
        "mse": mse_sum / total_samples,
        "alive_pct": alive_pct,
        "reconstruction_cosine": reconstruction_cosine_sum / total_samples,
        "decoder_mean_abs_cosine": compute_decoder_mean_abs_cosine(
            model,
            num_pairs=decoder_cosine_pairs,
        ),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    histogram_path = output_dir / "activation_histogram.pdf"
    save_activation_histogram(
        activation_counts,
        total_samples,
        histogram_path,
        title=eval_label,
    )
    metrics["activation_histogram_path"] = str(histogram_path)

    with (output_dir / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Loaded weights: {weights_path}")
    print(f"Dataset: {data_path} ({total_samples} samples)")
    print(f"MSE: {metrics['mse']:.6f}")
    print(f"Alive features: {alive_count}/{model.d_sae} ({alive_pct:.2f}%)")
    print(f"Reconstruction cosine: {metrics['reconstruction_cosine']:.6f}")
    print(f"Decoder mean |cos|: {metrics['decoder_mean_abs_cosine']:.6f}")
    print(f"Saved metrics and activation histogram to {output_dir}")
    return metrics


def _resolve_config_path(config_path: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def run_config_evaluation(args: argparse.Namespace) -> None:
    config_path = Path(args.eval_config).expanduser().resolve()
    with config_path.open() as f:
        config = json.load(f)

    models = config.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("Evaluation config must contain a non-empty 'models' list.")

    output_root = Path(args.output_dir).expanduser().resolve()
    device = torch.device(args.device)
    summary: list[dict[str, Any]] = []

    for entry in models:
        if not isinstance(entry, dict):
            raise ValueError("Each config model entry must be an object.")
        name = str(entry.get("name") or "").strip()
        if not name:
            raise ValueError("Each config model entry must define a non-empty 'name'.")
        label = str(entry.get("label") or name)
        weights = entry.get("weights")
        data = entry.get("fr_templates")
        if not isinstance(weights, str) or not isinstance(data, str):
            raise ValueError(f"Config entry '{name}' must define string weights and fr_templates paths.")

        print(f"Evaluating {label}...", flush=True)
        metrics = evaluate_checkpoint(
            weights_path=_resolve_config_path(config_path, weights),
            data_path=_resolve_config_path(config_path, data),
            output_dir=output_root / name,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            decoder_cosine_pairs=args.decoder_cosine_pairs,
            no_progress=args.no_progress,
            label=label,
        )
        summary.append({"name": name, **metrics})
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved batch summary to {output_root / 'summary.json'}")


def run_single_evaluation(args: argparse.Namespace) -> None:
    if not args.weights or not args.fr_templates:
        raise ValueError("--weights and --fr-templates are required unless --eval-config is provided.")
    evaluate_checkpoint(
        weights_path=Path(args.weights).expanduser().resolve(),
        data_path=Path(args.fr_templates).expanduser().resolve(),
        output_dir=Path(args.output_dir).expanduser().resolve(),
        device=torch.device(args.device),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        decoder_cosine_pairs=args.decoder_cosine_pairs,
        no_progress=args.no_progress,
        label=args.label,
    )


def parse_args() -> argparse.Namespace:
    P = load_paths()
    parser = argparse.ArgumentParser(
        description="Evaluate a trained BatchTopK SAE with the paper metrics."
    )
    parser.add_argument("--weights", type=str, help="Path to the trained model weights.")
    parser.add_argument("--fr-templates", type=str, default=None,
                        help="Path to the face-recognition template dataset (.pt/.pth). "
                             "Defaults to <templates_root>/<encoder-id>/templates.pth when --encoder-id is provided.")
    parser.add_argument("--encoder-id", type=str, default=None,
                        help="Optional encoder identifier; resolves --fr-templates from paths.yaml when set.")
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="JSON config with a 'models' list for batch evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Directory for metrics.json and activation_histogram.pdf.",
    )
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--label", type=str, default=None, help="Optional label for single-checkpoint output.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--decoder-cosine-pairs",
        type=int,
        default=DEFAULT_DECODER_COSINE_PAIRS,
        help="Number of random decoder-vector pairs for mean absolute cosine.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.eval_config:
        run_config_evaluation(args)
    else:
        run_single_evaluation(args)


if __name__ == "__main__":
    main()
