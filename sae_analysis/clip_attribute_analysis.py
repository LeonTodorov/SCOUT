#!/usr/bin/env python3
"""SAE neuron-attribute analysis using CLIP zero-shot classes.

Discovers SAE neurons selective for CLIP-defined positive/negative groups, then
re-ranks them by "CLIP agreeance" (how often the top-activating images are
classified into the expected class). Optionally runs a manipulation sweep:
encode a face -> linearly perturb the template along a neuron direction ->
decode back to an image, and score each alpha with CLIP.

Two manipulation backends:

  --backend dfd       Decode with a Deep Face Decoder checkpoint trained for
                      the same FR encoder.

  --backend arc2face  Decode with Arc2Face (Stable Diffusion conditioned on the
                      ArcFace template). Uses local checkpoints under
                      arc2face/pretrained/.

Examples:

  python sae_analysis/clip_attribute_analysis.py --backend dfd \\
      --weights batchtopk/runs/adaface_ms1mv2_ires50_k32/weights.pt \\
      --fr-templates data/glint360k/adaface_ms1mv2_ires50/templates.pth \\
      --clip-templates data/glint360k/open_clip_vitl_p14_336/templates_f16.pth \\
      --queries sae_analysis/queries/bald.json \\
      --encoder-id adaface_ms1mv2_ires50 \\
      --decoder-chkpt deep_face_decoder/runs/<run>/checkpoints/best.pt

  python sae_analysis/clip_attribute_analysis.py --backend arc2face \\
      --weights batchtopk/runs/arcface_webface42m_res100_k32/weights.pt \\
      --fr-templates data/glint360k/arcface_webface42m_res100/templates.pth \\
      --clip-templates data/glint360k/open_clip_vitl_p14_336/templates_f16.pth \\
      --queries sae_analysis/queries/bald.json \\
      --encoder-id arcface_webface42m_res100 \\
      --run-manipulation-scoring
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from _paths import setup_paths

setup_paths()

from sae_analysis.attributes import (  # noqa: E402
    build_clip_attributes,
    build_clip_report,
    build_top_neurons_positive_only,
    compute_clip_agreeance,
    compute_identity_diversity,
    compute_selectivity,
    full_dataset_scan,
    run_sae_on_subset,
    save_attribute_grids_with_agreeance,
)
from sae_analysis.clip_zero_shot import (  # noqa: E402
    _torch_load_cpu,
    encode_text_features,
    load_openclip,
    openclip_score_batched,
    parse_queries,
)
from sae_analysis.manipulation import (  # noqa: E402
    decoded_tensor_to_pil,
    direction_from_neuron,
    list_images_under,
    load_decoder,
    load_encoder,
    load_sae,
    manipulate_linear,
)

DEFAULT_OUTPUT_NAME = "clip_attribute_analysis.pt"


# ---------------------------------------------------------------------------
# Manipulation scoring -- shared display-grid helpers
# ---------------------------------------------------------------------------

def _display_indices(alphas: list[float]) -> tuple[list[int], list[str], set[int]]:
    """Pick which alpha columns appear in the figure: alpha=0 first (if present),
    then evenly spaced across the rest. The first grid column is reserved for
    the original input image.
    """
    n_alphas = len(alphas)
    zero_idx = alphas.index(0.0) if 0.0 in alphas else None
    display_indices: list[int] = [zero_idx] if zero_idx is not None else []

    display_pool = [i for i in range(n_alphas) if i != zero_idx]
    max_extra_alphas = 8 if zero_idx is not None else 9
    n_display_alpha = min(max_extra_alphas, len(display_pool))
    if n_display_alpha > 1:
        display_positions = [
            round(i * (len(display_pool) - 1) / (n_display_alpha - 1))
            for i in range(n_display_alpha)
        ]
        display_indices.extend(display_pool[pos] for pos in display_positions)
    elif n_display_alpha == 1:
        display_indices.append(display_pool[0])

    display_titles = ["input"] + [f"a={alphas[i]:.2f}" for i in display_indices]
    return display_indices, display_titles, set(display_indices)


def _self_and_id_metrics(
    all_fr_embeddings: list[torch.Tensor],
    base_fr_embeddings: list[torch.Tensor],
    all_reencoded: list[torch.Tensor],
    base_reencoded: list[torch.Tensor],
    n_seeds: int,
    n_alphas: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute self-similarity and re-encoded identity-preservation matrices."""
    emb_matrix = torch.stack(all_fr_embeddings).reshape(n_seeds, n_alphas, -1)
    base_emb = torch.stack(base_fr_embeddings)
    base_emb_norm = base_emb / (base_emb.norm(dim=1, keepdim=True) + 1e-8)

    self_sims = np.zeros((n_seeds, n_alphas))
    for si in range(n_seeds):
        embs = emb_matrix[si]
        embs_norm = embs / (embs.norm(dim=1, keepdim=True) + 1e-8)
        self_sims[si] = (embs_norm @ base_emb_norm[si]).numpy()

    reenc_matrix = torch.stack(all_reencoded).reshape(n_seeds, n_alphas, -1)
    base_reenc = torch.stack(base_reencoded)
    base_reenc_norm = base_reenc / (base_reenc.norm(dim=1, keepdim=True) + 1e-8)

    id_preserve = np.zeros((n_seeds, n_alphas))
    for si in range(n_seeds):
        reenc = reenc_matrix[si]
        reenc_norm = reenc / (reenc.norm(dim=1, keepdim=True) + 1e-8)
        id_preserve[si] = (reenc_norm @ base_reenc_norm[si]).numpy()

    return self_sims, id_preserve


def _build_manipulation_figure(
    attr_name: str,
    concept: str,
    neuron_results: list[dict],
    neuron_seed_images: list[list[list[Image.Image]]],
    n_seeds: int,
    alphas: list[float],
    display_titles: list[str],
    attr_dir: Path,
    backend_label: str,
) -> Path:
    """Render the combined figure (image grid on top, CLIP probability plot below)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    n_neurons = len(neuron_results)
    n_grid_cols = len(display_titles)
    n_display_seeds = min(25, n_seeds)
    n_grid_rows = n_neurons * n_display_seeds

    fig = plt.figure(figsize=(max(n_grid_cols * 2.5, 10), n_grid_rows * 1.5 + 4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[max(n_grid_rows, 1), 2], hspace=0.35)
    gs_images = gridspec.GridSpecFromSubplotSpec(
        n_grid_rows, n_grid_cols, subplot_spec=gs[0], wspace=0.05, hspace=0.08,
    )

    grid_axes: dict[tuple[int, int], object] = {}
    for neuron_row in range(n_neurons):
        neuron_idx = neuron_results[neuron_row]["neuron"]
        for seed_row in range(n_display_seeds):
            grid_row = neuron_row * n_display_seeds + seed_row
            for col_idx in range(n_grid_cols):
                ax = fig.add_subplot(gs_images[grid_row, col_idx])
                ax.imshow(neuron_seed_images[neuron_row][seed_row][col_idx])
                ax.set_xticks([])
                ax.set_yticks([])
                ax.axis("off")
                if grid_row == 0:
                    ax.set_title(display_titles[col_idx], fontsize=7)
                if col_idx == 0 and seed_row == 0:
                    ax.set_ylabel(
                        f"n{neuron_idx}", fontsize=8, rotation=0, labelpad=30, va="center",
                    )
                    ax.axis("on")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for sp in ax.spines.values():
                        sp.set_visible(False)
                grid_axes[(grid_row, col_idx)] = ax

    for neuron_row in range(n_neurons - 1):
        last_seed_row = neuron_row * n_display_seeds + (n_display_seeds - 1)
        left_ax = grid_axes[(last_seed_row, 0)]
        right_ax = grid_axes[(last_seed_row, n_grid_cols - 1)]
        left_bbox = left_ax.get_position()
        right_bbox = right_ax.get_position()
        y = left_bbox.y0
        line = Line2D(
            [left_bbox.x0, right_bbox.x1], [y, y],
            transform=fig.transFigure, color="gray",
            linewidth=1.0, clip_on=False,
        )
        fig.add_artist(line)

    ax_clip = fig.add_subplot(gs[1])
    for res in neuron_results:
        nidx = res["neuron"]
        means = np.array(res["mean_clip_probs"])
        stds = np.array(res["std_clip_probs"])
        label = f"n{nidx} (Δ={res['delta']:+.2f})"
        ax_clip.plot(alphas, means, linewidth=1.5, label=label)
        ax_clip.fill_between(alphas, means - stds, means + stds, alpha=0.15)

    ax_clip.set_xlabel("Alpha")
    ax_clip.set_ylabel("P(positive)")
    ax_clip.set_title(f"CLIP Softmax Probability: {concept}")
    ax_clip.set_ylim(0, 1)
    ax_clip.set_xlim(alphas[0], alphas[-1])
    if n_neurons <= 10:
        ax_clip.legend(fontsize=6, loc="best")
    ax_clip.grid(True, alpha=0.3)

    fig.suptitle(f"{attr_name} -- {backend_label} Manipulation Sweep", fontsize=12, y=0.99)
    out_path = attr_dir / "manipulation_sweep.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _resolve_attribute_prompts(
    attr_name: str,
    query_meta: dict,
    queries: list[dict[str, list[str]]],
) -> tuple[str, list[str], list[str]]:
    """Return (concept, pos_prompts, neg_prompts) for an attribute."""
    meta = query_meta[attr_name]
    class_names = meta["class_names"]
    pos_cls_idx = meta["pos_class_idx"]
    neg_cls_idx = meta.get("neg_class_idx")
    query = queries[meta["query_idx"]]

    pos_prompts = query[class_names[pos_cls_idx]]
    if neg_cls_idx is not None:
        neg_prompts = query[class_names[neg_cls_idx]]
    else:
        neg_prompts = []
        for ci, cn in enumerate(class_names):
            if ci != pos_cls_idx:
                neg_prompts.extend(query[cn])
    return class_names[pos_cls_idx], pos_prompts, neg_prompts


# ---------------------------------------------------------------------------
# Manipulation scoring -- DFD backend
# ---------------------------------------------------------------------------

def run_manipulation_scoring_dfd(
    neurons_with_agreeance: dict[str, list[tuple[int, float, float, float]]],
    attribute_names: list[str],
    query_meta: dict[str, dict],
    queries: list[dict[str, list[str]]],
    args: argparse.Namespace,
    device: torch.device,
    grid_dir: Path,
) -> dict[str, dict[str, object]]:
    """DFD-decoder manipulation sweep: encode -> perturb -> decode -> CLIP-score."""
    import random

    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step * 0.5, args.alpha_step).tolist()
    alphas = [round(a, 6) for a in alphas]
    n_alphas = len(alphas)
    display_indices, display_titles, display_set = _display_indices(alphas)

    lfw_root = Path(args.lfw_root).expanduser().resolve()
    if not lfw_root.is_dir():
        print(f"WARNING: --lfw-root not found: {lfw_root}, skipping manipulation scoring.")
        return {}

    print("\n=== Manipulation scoring (DFD backend): loading models ===")
    encoder, preprocess_fn = load_encoder(args.encoder_id, device)
    decoder, output_size = load_decoder(Path(args.decoder_chkpt), device)
    sae = load_sae(Path(args.weights), device)

    candidates = list_images_under(lfw_root)
    if not candidates:
        print(f"WARNING: No images under {lfw_root}, skipping manipulation scoring.")
        return {}
    rng = random.Random(args.seed)
    n_seeds = min(args.random_count, len(candidates))
    image_paths = rng.sample(candidates, n_seeds)
    print(f"Selected {n_seeds} seed images from {lfw_root}")

    print("Loading OpenCLIP for manipulation scoring...")
    clip_model, clip_preprocess, clip_tokenizer = load_openclip(device)

    saved_results: dict[str, dict[str, object]] = {}

    for attr_name in attribute_names:
        entries = neurons_with_agreeance.get(attr_name, [])
        if not entries:
            continue

        concept, pos_prompts, neg_prompts = _resolve_attribute_prompts(attr_name, query_meta, queries)
        attr_dir = grid_dir / attr_name.lower().replace(" ", "_")
        attr_dir.mkdir(parents=True, exist_ok=True)

        pos_text_feats, neg_text_feats = encode_text_features(
            clip_model, clip_tokenizer, pos_prompts, neg_prompts, device,
        )

        neuron_results: list[dict] = []
        neuron_seed_images: list[list[list[Image.Image]]] = []

        print(f"\n  {attr_name}: scoring {len(entries)} neurons...")
        for neuron_idx, selectivity, top_agree, full_agree in tqdm(entries, desc=f"  {attr_name}"):
            direction = direction_from_neuron(sae, neuron_idx)

            all_decoded_pils: list[Image.Image] = []
            all_fr_embeddings: list[torch.Tensor] = []
            base_fr_embeddings: list[torch.Tensor] = []
            all_reencoded: list[torch.Tensor] = []
            base_reencoded: list[torch.Tensor] = []
            seed_display_images: list[list[Image.Image]] = []

            with torch.no_grad():
                for path in tqdm(image_paths, desc=f"    n{neuron_idx} decode", leave=False):
                    with Image.open(path) as img:
                        input_pil = img.convert("RGB").copy()
                        tensor = preprocess_fn(input_pil, size=(output_size, output_size))
                    template = encoder(tensor.unsqueeze(0).to(device))

                    seed_display: list[Image.Image] = [input_pil]
                    for ai, alpha in enumerate(alphas):
                        modified = template if alpha == 0.0 else manipulate_linear(template, direction, alpha)

                        recon = decoder(modified)
                        pil_img = decoded_tensor_to_pil(recon.squeeze(0).cpu())
                        all_decoded_pils.append(pil_img)
                        all_fr_embeddings.append(modified.squeeze(0).cpu())

                        recon_input = preprocess_fn(pil_img, size=(output_size, output_size))
                        reenc_templ = encoder(recon_input.unsqueeze(0).to(device))
                        all_reencoded.append(reenc_templ.squeeze(0).cpu())

                        if ai == 0:
                            base_fr_embeddings.append(modified.squeeze(0).cpu())
                            base_reencoded.append(reenc_templ.squeeze(0).cpu())
                        if ai in display_set:
                            seed_display.append(pil_img)

                    seed_display_images.append(seed_display)

            neuron_seed_images.append(seed_display_images)

            all_probs = openclip_score_batched(
                clip_model, clip_preprocess, all_decoded_pils,
                pos_text_feats, neg_text_feats, device, batch_size=64,
            )
            all_probs_2d = all_probs.reshape(n_seeds, n_alphas)

            self_sims, id_preserve = _self_and_id_metrics(
                all_fr_embeddings, base_fr_embeddings,
                all_reencoded, base_reencoded,
                n_seeds, n_alphas,
            )

            mean_probs = all_probs_2d.numpy().mean(axis=0)
            std_probs = all_probs_2d.numpy().std(axis=0)
            delta = float(mean_probs[-1] - mean_probs[0])

            neuron_results.append({
                "neuron": neuron_idx,
                "concept": concept,
                "alphas": alphas,
                "mean_clip_probs": mean_probs.tolist(),
                "std_clip_probs": std_probs.tolist(),
                "mean_self_sim": self_sims.mean(axis=0).tolist(),
                "std_self_sim": self_sims.std(axis=0).tolist(),
                "mean_id_preserve": id_preserve.mean(axis=0).tolist(),
                "std_id_preserve": id_preserve.std(axis=0).tolist(),
                "delta": delta,
            })
            tqdm.write(f"    n{neuron_idx}: delta={delta:+.3f}")

        out_path = _build_manipulation_figure(
            attr_name, concept, neuron_results, neuron_seed_images,
            n_seeds, alphas, display_titles, attr_dir, backend_label="DFD",
        )
        print(f"  Saved: {out_path}")
        saved_results[attr_name] = {
            "plot_path": str(out_path),
            "display_titles": display_titles,
            "neurons": neuron_results,
        }

    return saved_results


# ---------------------------------------------------------------------------
# Manipulation scoring -- Arc2Face backend
# ---------------------------------------------------------------------------

def run_manipulation_scoring_arc2face(
    neurons_with_agreeance: dict[str, list[tuple[int, float, float, float]]],
    attribute_names: list[str],
    query_meta: dict[str, dict],
    queries: list[dict[str, list[str]]],
    args: argparse.Namespace,
    device: torch.device,
    grid_dir: Path,
) -> dict[str, dict[str, object]]:
    """Arc2Face manipulation sweep: encode -> perturb -> diffuse -> CLIP-score."""
    import random

    from arc2face.loader import (
        encode_input_image,
        generate_from_embedding,
        load_arc2face_pipeline,
    )
    from feature_extractors.arcface_webface42m_res100 import inference as a2f_inference

    alphas = np.arange(args.alpha_min, args.alpha_max + args.alpha_step * 0.5, args.alpha_step).tolist()
    alphas = [round(a, 6) for a in alphas]
    n_alphas = len(alphas)
    display_indices, display_titles, display_set = _display_indices(alphas)

    lfw_root = Path(args.lfw_root).expanduser().resolve()
    if not lfw_root.is_dir():
        print(f"WARNING: --lfw-root not found: {lfw_root}, skipping manipulation scoring.")
        return {}

    recognizer_path = Path(args.recognizer).expanduser().resolve()
    if not recognizer_path.is_file():
        print(f"WARNING: recognizer checkpoint not found: {recognizer_path}, skipping manipulation scoring.")
        return {}

    print("\n=== Manipulation scoring (Arc2Face backend): loading models ===")
    recognizer = a2f_inference.load_model(recognizer_path, device=device)
    pipeline, dtype = load_arc2face_pipeline(
        device,
        arc2face_model=args.arc2face_model,
        base_model=args.base_model,
        cache_dir=args.cache_dir,
    )
    sae = load_sae(Path(args.weights), device)

    candidates = list_images_under(lfw_root)
    if not candidates:
        print(f"WARNING: No images under {lfw_root}, skipping manipulation scoring.")
        return {}
    rng = random.Random(args.seed)
    n_seeds = min(args.random_count, len(candidates))
    image_paths = rng.sample(candidates, n_seeds)
    print(f"Selected {n_seeds} seed images from {lfw_root}")

    print("Loading OpenCLIP for manipulation scoring...")
    clip_model, clip_preprocess, clip_tokenizer = load_openclip(device)

    saved_results: dict[str, dict[str, object]] = {}

    for attr_name in attribute_names:
        entries = neurons_with_agreeance.get(attr_name, [])
        if not entries:
            continue

        concept, pos_prompts, neg_prompts = _resolve_attribute_prompts(attr_name, query_meta, queries)
        attr_dir = grid_dir / attr_name.lower().replace(" ", "_")
        attr_dir.mkdir(parents=True, exist_ok=True)

        pos_text_feats, neg_text_feats = encode_text_features(
            clip_model, clip_tokenizer, pos_prompts, neg_prompts, device,
        )

        neuron_results: list[dict] = []
        neuron_seed_images: list[list[list[Image.Image]]] = []

        print(f"\n  {attr_name}: scoring {len(entries)} neurons...")
        for neuron_idx, selectivity, top_agree, full_agree in tqdm(entries, desc=f"  {attr_name}"):
            direction = direction_from_neuron(sae, neuron_idx)

            all_generated_pils: list[Image.Image] = []
            all_fr_embeddings: list[torch.Tensor] = []
            base_fr_embeddings: list[torch.Tensor] = []
            all_reencoded: list[torch.Tensor] = []
            base_reencoded: list[torch.Tensor] = []
            seed_display_images: list[list[Image.Image]] = []

            with torch.no_grad():
                for img_idx, path in enumerate(tqdm(
                    image_paths, desc=f"    n{neuron_idx} generate", leave=False,
                )):
                    with Image.open(path) as img:
                        input_pil = img.convert("RGB").copy()

                    template = encode_input_image(recognizer, input_pil, device)
                    base_seed = args.seed + img_idx
                    seed_display: list[Image.Image] = [input_pil]

                    for ai, alpha in enumerate(alphas):
                        modified = template if alpha == 0.0 else manipulate_linear(template, direction, alpha)

                        gen_img = generate_from_embedding(
                            pipeline, dtype, modified,
                            args.num_steps, args.guidance_scale, base_seed,
                        )
                        all_generated_pils.append(gen_img)
                        all_fr_embeddings.append(modified.squeeze(0).cpu())

                        reenc_templ = encode_input_image(recognizer, gen_img, device)
                        all_reencoded.append(reenc_templ.squeeze(0).cpu())

                        if ai == 0:
                            base_fr_embeddings.append(modified.squeeze(0).cpu())
                            base_reencoded.append(reenc_templ.squeeze(0).cpu())
                        if ai in display_set:
                            seed_display.append(gen_img)

                    seed_display_images.append(seed_display)

            neuron_seed_images.append(seed_display_images)

            all_probs = openclip_score_batched(
                clip_model, clip_preprocess, all_generated_pils,
                pos_text_feats, neg_text_feats, device, batch_size=64,
            )
            all_probs_2d = all_probs.reshape(n_seeds, n_alphas)

            self_sims, id_preserve = _self_and_id_metrics(
                all_fr_embeddings, base_fr_embeddings,
                all_reencoded, base_reencoded,
                n_seeds, n_alphas,
            )

            mean_probs = all_probs_2d.numpy().mean(axis=0)
            std_probs = all_probs_2d.numpy().std(axis=0)
            delta = float(mean_probs[-1] - mean_probs[0])

            neuron_results.append({
                "neuron": neuron_idx,
                "concept": concept,
                "alphas": alphas,
                "mean_clip_probs": mean_probs.tolist(),
                "std_clip_probs": std_probs.tolist(),
                "mean_self_sim": self_sims.mean(axis=0).tolist(),
                "std_self_sim": self_sims.std(axis=0).tolist(),
                "mean_id_preserve": id_preserve.mean(axis=0).tolist(),
                "std_id_preserve": id_preserve.std(axis=0).tolist(),
                "delta": delta,
            })
            tqdm.write(f"    n{neuron_idx}: delta={delta:+.3f}")

        out_path = _build_manipulation_figure(
            attr_name, concept, neuron_results, neuron_seed_images,
            n_seeds, alphas, display_titles, attr_dir, backend_label="Arc2Face",
        )
        print(f"  Saved: {out_path}")
        saved_results[attr_name] = {
            "plot_path": str(out_path),
            "display_titles": display_titles,
            "neurons": neuron_results,
        }

    return saved_results


def run_manipulation_scoring(
    backend: str,
    *args, **kwargs,
) -> dict[str, dict[str, object]]:
    """Backend dispatcher for manipulation scoring."""
    if backend == "dfd":
        return run_manipulation_scoring_dfd(*args, **kwargs)
    if backend == "arc2face":
        return run_manipulation_scoring_arc2face(*args, **kwargs)
    raise ValueError(f"Unknown backend: {backend}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Core
    p.add_argument("--backend", required=True, choices=("dfd", "arc2face"),
                   help="Inversion backend for manipulation scoring.")
    p.add_argument("--weights", type=str, required=True,
                   help="Path to SAE (Sparse Autoencoder) weights checkpoint (.pt).")
    p.add_argument("--fr-templates", type=str, default=None,
                   help="Face-recognition templates (.pth), shape [N, d_in]. "
                        "Defaults from paths.yaml + --encoder-id. "
                        "Required (or derivable) unless --manipulation-only is set.")
    p.add_argument("--clip-templates", type=str, default=None,
                   help="CLIP templates (.pth), shape [N, 768]. Aligned with --fr-templates by index. "
                        "Defaults from paths.yaml. Required unless --manipulation-only is set.")
    p.add_argument("--queries", nargs="+", required=True,
                   help="CLIP query definitions: path to a JSON file or inline JSON.")

    # Discovery
    p.add_argument("--samples-per-class", type=int, default=1024,
                   help="Top-scoring samples per class for positive/negative groups.")
    p.add_argument("--batch-size", type=int, default=4096,
                   help="Batch size for SAE inference passes.")
    p.add_argument("--device", type=str, default="cuda:0",
                   help="Torch device (cuda:0, cpu, ...).")
    p.add_argument("--output", type=str, default=None,
                   help=f"Output path for results (.pt). Defaults to <weights_dir>/{DEFAULT_OUTPUT_NAME}.")
    p.add_argument("--top-k", type=int, default=20,
                   help="Neurons per attribute kept after agreeance filtering.")
    p.add_argument("--discovery-k", type=int, default=None,
                   help="Candidate neurons per attribute retrieved by selectivity before agreeance filtering. "
                        "Defaults to --top-k.")
    p.add_argument("--min-agreeance", type=float, default=0.0,
                   help="Minimum top agreeance threshold before truncating to --top-k.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for image/identity sampling and Arc2Face generation.")

    # Data paths
    p.add_argument("--glint360k-root", type=str, default=None,
                   help="Root directory of the Glint360K image dataset. "
                        "Defaults from paths.yaml:glint360k_root.")
    p.add_argument("--ids-and-fns", type=str, default=None,
                   help="Path to ids_and_fns.npy mapping sample indices to (identity, filename). "
                        "Defaults from paths.yaml:ids_and_fns.")
    p.add_argument("--grid-images", type=int, default=512,
                   help="Top-activating unique-ID images to collect per neuron for the grid.")

    # Manipulation
    p.add_argument("--run-manipulation-scoring", action="store_true",
                   help="Run the manipulation sweep after selectivity/agreeance.")
    p.add_argument("--manipulation-only", action="store_true",
                   help="Skip discovery and only run manipulation scoring on a previously saved results file.")
    p.add_argument("--encoder-id", type=str, default=None,
                   help="Face-recognition encoder identifier (e.g. adaface_ms1mv2_ires50). "
                        "DFD: must match the decoder. Arc2Face: identifies the FR encoder used to score templates.")
    p.add_argument("--lfw-root", type=str, default=None,
                   help="Directory of seed face images for the manipulation sweep. "
                        "Defaults from paths.yaml:lfw_root.")
    p.add_argument("--random-count", type=int, default=10,
                   help="Number of seed images sampled from --lfw-root.")
    p.add_argument("--alpha-min", type=float, default=0.0,
                   help="Start of the alpha sweep (inclusive).")
    p.add_argument("--alpha-max", type=float, default=0.6,
                   help="End of the alpha sweep (inclusive).")
    p.add_argument("--alpha-step", type=float, default=0.01,
                   help="Step size between alpha values in the sweep.")

    # DFD-only
    p.add_argument("--decoder-chkpt", type=str, default=None,
                   help="(--backend dfd) Path to a Deep Face Decoder checkpoint (.pt).")

    # Arc2Face-only
    p.add_argument("--num-steps", type=int, default=25,
                   help="(--backend arc2face) Diffusion inference steps.")
    p.add_argument("--guidance-scale", type=float, default=3.0,
                   help="(--backend arc2face) Classifier-free guidance scale.")
    p.add_argument("--cache-dir", type=str, default=None,
                   help="(--backend arc2face) HuggingFace hub cache used to resolve model IDs. "
                        "Optional when --arc2face-model is a local path.")
    p.add_argument("--arc2face-model", type=str, default=None,
                   help="(--backend arc2face) Arc2Face model ID or local path. "
                        "Defaults to <repo>/arc2face/pretrained.")
    p.add_argument("--base-model", type=str, default=None,
                   help="(--backend arc2face) Base Stable Diffusion model ID or local path. "
                        "Defaults to <repo>/arc2face/pretrained/stable-diffusion-v1-5.")
    p.add_argument("--recognizer", type=str, default=None,
                   help="(--backend arc2face) ArcFace recognizer checkpoint. Defaults to "
                        "feature_extractors/arcface_webface42m_res100/pretrained/arc2face_arcface_r100_model.pth.")

    args = p.parse_args()

    # Backend-specific validation
    if args.backend == "dfd":
        if not args.encoder_id:
            p.error("--backend dfd requires --encoder-id.")
        if (args.run_manipulation_scoring or args.manipulation_only) and not args.decoder_chkpt:
            p.error("--backend dfd manipulation requires --decoder-chkpt.")
    elif args.backend == "arc2face":
        if args.arc2face_model is None:
            from arc2face.loader import DEFAULT_ARC2FACE_MODEL
            args.arc2face_model = DEFAULT_ARC2FACE_MODEL
        if args.base_model is None:
            from arc2face.loader import DEFAULT_BASE_MODEL
            args.base_model = DEFAULT_BASE_MODEL
        if args.recognizer is None:
            args.recognizer = str(
                Path(__file__).resolve().parents[1]
                / "feature_extractors" / "arcface_webface42m_res100"
                / "pretrained" / "arc2face_arcface_r100_model.pth"
            )

    return args


def _reconstruct_query_meta(
    queries: list[dict[str, list[str]]],
    neurons_with_agreeance: dict,
) -> dict[str, dict]:
    """Rebuild the minimal query_meta needed for manipulation-only mode from the queries JSON."""
    query_meta: dict[str, dict] = {}
    for qi, query in enumerate(queries):
        class_names = list(query.keys())
        if len(class_names) == 2:
            attr_name = f"clip_{class_names[0]}"
            if attr_name in neurons_with_agreeance:
                query_meta[attr_name] = {
                    "query_idx": qi, "class_names": class_names,
                    "pos_class_idx": 0, "neg_class_idx": 1,
                }
        else:
            for cls_idx, cls_name in enumerate(class_names):
                attr_name = f"clip_{cls_name}"
                if attr_name in neurons_with_agreeance:
                    query_meta[attr_name] = {
                        "query_idx": qi, "class_names": class_names,
                        "pos_class_idx": cls_idx, "neg_class_idx": None,
                    }
    return query_meta


def _fill_path_defaults_from_yaml(args: argparse.Namespace) -> None:
    """Fill any path args left as None from paths.yaml.

    Explicit ``--flags`` always win — this only populates omitted ones.
    """
    from paths import load_paths  # noqa: E402

    P = load_paths()
    if args.glint360k_root is None:
        args.glint360k_root = str(P["glint360k_root"])
    if args.ids_and_fns is None:
        args.ids_and_fns = str(P["ids_and_fns"])
    if args.lfw_root is None:
        args.lfw_root = str(P["lfw_root"])
    if args.clip_templates is None:
        args.clip_templates = str(P["templates_root"] / "open_clip_vitl_p14_336" / "templates_f16.pth")
    if args.fr_templates is None and args.encoder_id is not None:
        args.fr_templates = str(P["templates_root"] / args.encoder_id / "templates.pth")


def main() -> int:
    args = parse_args()
    _fill_path_defaults_from_yaml(args)
    device = torch.device(args.device)
    weights_path = Path(args.weights).expanduser().resolve()

    discovery_k = args.discovery_k if args.discovery_k is not None else args.top_k
    if discovery_k < args.top_k:
        print(f"WARNING: --discovery-k ({discovery_k}) < --top-k ({args.top_k}); using top-k.", file=sys.stderr)
        discovery_k = args.top_k

    if not weights_path.exists():
        print(f"ERROR: SAE weights not found: {weights_path}", file=sys.stderr)
        return 1

    try:
        queries = parse_queries(args.queries)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else weights_path.parent / DEFAULT_OUTPUT_NAME
    )

    # --- Manipulation-only shortcut ---
    if args.manipulation_only:
        if not output_path.exists():
            print(f"ERROR: Results file not found: {output_path}. Run full pipeline first.", file=sys.stderr)
            return 1

        print(f"Loading saved results from {output_path} ...")
        saved = torch.load(output_path, map_location="cpu", weights_only=False)
        attribute_names = saved["attribute_names"]
        neurons_with_agreeance = saved["neurons_with_agreeance"]
        query_meta = _reconstruct_query_meta(queries, neurons_with_agreeance)

        grid_dir = output_path.parent / "clip_attribute_grids"
        manipulation_results = run_manipulation_scoring(
            args.backend,
            neurons_with_agreeance, attribute_names, query_meta,
            queries, args, device, grid_dir,
        )
        saved["manipulation_backend"] = args.backend
        saved["manipulation_results"] = manipulation_results
        saved["manipulation_plot_paths"] = {
            attr_name: data["plot_path"] for attr_name, data in manipulation_results.items()
        }
        torch.save(saved, output_path)
        print(f"Updated results saved to {output_path}")
        return 0

    # --- Full pipeline ---
    if not args.fr_templates or not args.clip_templates:
        print("ERROR: --fr-templates and --clip-templates are required for the full pipeline.", file=sys.stderr)
        return 1

    fr_path = Path(args.fr_templates).expanduser().resolve()
    clip_path = Path(args.clip_templates).expanduser().resolve()
    for p_path, label in ((fr_path, "FR templates"), (clip_path, "CLIP templates")):
        if not p_path.exists():
            print(f"ERROR: {label} not found: {p_path}", file=sys.stderr)
            return 1

    print(f"Loading CLIP templates: {clip_path}")
    clip_templates = _torch_load_cpu(clip_path)
    if not isinstance(clip_templates, torch.Tensor):
        clip_templates = torch.as_tensor(clip_templates)
    print(f"  CLIP templates shape: {clip_templates.shape}")

    print("\n=== PHASE 1: Build CLIP attributes & compute selectivity ===")
    source, query_meta, pos_indices, neg_indices, unique_indices = build_clip_attributes(
        queries, clip_templates, device, args.samples_per_class,
    )

    from batchtopk.inference import load_model as load_sae_model

    print(f"\nLoading SAE from {weights_path} ...")
    model = load_sae_model(weights_path, device)
    model.eval()
    print(f"  d_in={model.d_in}, d_sae={model.d_sae}, threshold={model.global_threshold:.6f}")

    print(f"Loading FR templates from {fr_path} ...")
    embeddings = _torch_load_cpu(fr_path)
    if not isinstance(embeddings, torch.Tensor):
        raise ValueError(f"Expected tensor at {fr_path}, got {type(embeddings)}")
    print(f"  Shape: {embeddings.shape}")

    print("Running SAE inference on CLIP-selected samples ...")
    activations = run_sae_on_subset(model, embeddings, unique_indices, args.batch_size, device)

    print("Computing selectivity metrics ...")
    metrics = compute_selectivity(activations, pos_indices, neg_indices, unique_indices)
    del activations

    top_neurons = build_top_neurons_positive_only(
        metrics["selectivity"], source.attribute_names, discovery_k,
    )

    print("\n=== PHASE 2: Full-dataset scan & CLIP agreeance ===")
    ids_fns_path = Path(args.ids_and_fns).expanduser().resolve()
    image_root = Path(args.glint360k_root).expanduser().resolve()
    ids_and_fns = np.load(ids_fns_path, allow_pickle=True)
    identities = ids_and_fns[:, 0]

    all_neuron_idxs: set[int] = set()
    for attr_name in source.attribute_names:
        for neuron_idx, _ in top_neurons.get(attr_name, [])[:discovery_k]:
            all_neuron_idxs.add(neuron_idx)
    neuron_list = sorted(all_neuron_idxs)
    print(
        f"Scanning full dataset for top-{args.grid_images} unique-ID images across "
        f"{len(neuron_list)} neurons ({discovery_k} candidates/attr) ..."
    )

    global_top_bottom, full_agreeance_stats = full_dataset_scan(
        model, embeddings, neuron_list,
        top_k=args.grid_images,
        batch_size=args.batch_size,
        device=device,
        identities=identities,
        query_meta=query_meta,
        top_neurons_per_attr=top_neurons,
        neurons_per_attr=discovery_k,
    )
    del embeddings

    identity_diversity = compute_identity_diversity(global_top_bottom, identities)

    print("Computing CLIP agreeance ...")
    neurons_with_agreeance = compute_clip_agreeance(
        top_neurons, global_top_bottom, query_meta, discovery_k, full_agreeance_stats,
    )

    if discovery_k > args.top_k or args.min_agreeance > 0.0:
        n_before = sum(len(v) for v in neurons_with_agreeance.values())
        for attr_name in list(neurons_with_agreeance.keys()):
            entries = neurons_with_agreeance[attr_name]
            if args.min_agreeance > 0.0:
                entries = [(n, s, ta, fa) for n, s, ta, fa in entries if ta >= args.min_agreeance]
            neurons_with_agreeance[attr_name] = entries[:args.top_k]
        n_after = sum(len(v) for v in neurons_with_agreeance.values())
        print(
            f"Agreeance filter: {n_before} -> {n_after} neurons "
            f"(discovery_k={discovery_k}, top_k={args.top_k}, min_agreeance={args.min_agreeance})"
        )

    for attr_name in source.attribute_names:
        entries = neurons_with_agreeance.get(attr_name, [])
        if entries:
            top = entries[0]
            print(
                f"  {attr_name}: best neuron n{top[0]}"
                f" top={top[2]:.1%} full={top[3]:.1%} sel={top[1]:+.3f}"
            )

    model_info = {
        "weights_path": str(weights_path),
        "d_in": model.d_in,
        "d_sae": model.d_sae,
        "threshold": f"{model.global_threshold:.6f}",
    }
    report = build_clip_report(
        source.attribute_names, neurons_with_agreeance, metrics,
        model_info, args.samples_per_class,
        identity_diversity=identity_diversity,
    )
    print("\n" + report)

    grid_dir = output_path.parent / "clip_attribute_grids"
    print(f"\nGenerating image grids in {grid_dir} ...")
    save_attribute_grids_with_agreeance(
        global_top_bottom=global_top_bottom,
        neurons_with_agreeance=neurons_with_agreeance,
        attribute_names=source.attribute_names,
        ids_and_fns=ids_and_fns,
        image_root=image_root,
        output_dir=grid_dir,
        neurons_per_attr=args.top_k,
        identity_diversity=identity_diversity,
    )

    manipulation_results: dict[str, dict[str, object]] | None = None
    if args.run_manipulation_scoring:
        manipulation_results = run_manipulation_scoring(
            args.backend,
            neurons_with_agreeance, source.attribute_names, query_meta,
            queries, args, device, grid_dir,
        )

    report_path = output_path.with_suffix(".txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"Report saved to {report_path}")

    results: dict[str, object] = {
        "weights_path": str(weights_path),
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "top_k": args.top_k,
        "discovery_k": discovery_k,
        "min_agreeance": args.min_agreeance,
        "attribute_names": source.attribute_names,
        "selectivity_matrix": metrics["selectivity"],
        "fire_rate_pos": metrics["fire_rate_pos"],
        "fire_rate_neg": metrics["fire_rate_neg"],
        "neurons_with_agreeance": {
            name: [(n, s, ta, fa) for n, s, ta, fa in entries]
            for name, entries in neurons_with_agreeance.items()
        },
        "manipulation_backend": args.backend,
    }
    if identity_diversity:
        results["identity_diversity"] = identity_diversity
    if manipulation_results is not None:
        results["manipulation_results"] = manipulation_results
        results["manipulation_plot_paths"] = {
            attr_name: data["plot_path"] for attr_name, data in manipulation_results.items()
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)
    print(f"Results saved to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
