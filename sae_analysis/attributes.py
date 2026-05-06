"""SAE selectivity, identity-diversity, and image-grid helpers."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


@dataclass
class AttributeSource:
    """A named set of binary attributes sharing the same sample ordering."""

    name: str
    attributes: torch.Tensor  # [N, num_attrs] uint8 binary
    attribute_names: list[str]


def run_sae_on_subset(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run SAE inference on a subset of embeddings, return [len(indices), d_sae] CPU activations."""
    subset = embeddings[indices]
    num_samples = subset.shape[0]
    dtype = model.W_enc.dtype
    activations = torch.zeros(num_samples, model.d_sae)

    for start in tqdm(range(0, num_samples, batch_size), desc="SAE inference"):
        end = min(start + batch_size, num_samples)
        batch = subset[start:end].to(device=device, dtype=dtype)
        with torch.no_grad():
            _, features, _ = model(batch)
        activations[start:end] = features.cpu()

    return activations


def compute_selectivity(
    activations: torch.Tensor,
    pos_indices: list[list[int]],
    neg_indices: list[list[int]],
    unique_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Per-attribute, per-neuron selectivity = fire_rate_pos - fire_rate_neg.

    Returns a dict of [num_attrs, d_sae] tensors:
        selectivity, fire_rate_pos, fire_rate_neg, mean_act_pos, mean_act_neg.
    """
    idx_to_pos = {global_idx: pos for pos, global_idx in enumerate(unique_indices.tolist())}

    num_attrs = len(pos_indices)
    d_sae = activations.shape[1]

    fire_rate_pos = torch.zeros(num_attrs, d_sae)
    fire_rate_neg = torch.zeros(num_attrs, d_sae)
    mean_act_pos = torch.zeros(num_attrs, d_sae)
    mean_act_neg = torch.zeros(num_attrs, d_sae)

    for attr_idx in range(num_attrs):
        pos_positions = [idx_to_pos[i] for i in pos_indices[attr_idx]]
        neg_positions = [idx_to_pos[i] for i in neg_indices[attr_idx]]

        if pos_positions:
            pos_acts = activations[pos_positions]
            fire_rate_pos[attr_idx] = (pos_acts > 0).float().mean(dim=0)
            mean_act_pos[attr_idx] = pos_acts.mean(dim=0)

        if neg_positions:
            neg_acts = activations[neg_positions]
            fire_rate_neg[attr_idx] = (neg_acts > 0).float().mean(dim=0)
            mean_act_neg[attr_idx] = neg_acts.mean(dim=0)

    return {
        "selectivity": fire_rate_pos - fire_rate_neg,
        "fire_rate_pos": fire_rate_pos,
        "fire_rate_neg": fire_rate_neg,
        "mean_act_pos": mean_act_pos,
        "mean_act_neg": mean_act_neg,
    }


def compute_identity_diversity(
    global_top_bottom: dict[int, tuple[torch.Tensor, torch.Tensor]],
    identities: np.ndarray,
) -> dict[int, int]:
    """Count unique identities among top-activating samples for each neuron."""
    result: dict[int, int] = {}
    for neuron_idx, (top_idxs, _) in global_top_bottom.items():
        valid = top_idxs[top_idxs >= 0].tolist()
        result[neuron_idx] = len({identities[i] for i in valid})
    return result


def resolve_image_path_from_ids(idx: int, ids_and_fns: np.ndarray, image_root: Path) -> Path:
    """Resolve an image path: ids_and_fns[idx] = [identity, suffix] -> root/id_<id>/<id>_<suffix>."""
    iid, suffix = ids_and_fns[idx]
    return image_root / f"id_{iid}" / f"{iid}_{suffix}"


def make_image_grid(
    image_paths: list[Path],
    thumb_size: int = 112,
    title: str | None = None,
) -> Image.Image | None:
    """Build a roughly square thumbnail grid with an optional title bar."""
    valid_images: list[Image.Image] = []
    for p in image_paths:
        if p.exists():
            try:
                img = Image.open(p).convert("RGB").resize((thumb_size, thumb_size), Image.LANCZOS)
                valid_images.append(img)
            except Exception:
                continue
    if not valid_images:
        return None

    n = len(valid_images)
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)

    title_height = 0
    font = None
    if title:
        from PIL import ImageDraw, ImageFont
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except (OSError, IOError):
            font = ImageFont.load_default()
        title_height = 24

    grid = Image.new("RGB", (ncols * thumb_size, nrows * thumb_size + title_height), (40, 40, 40))
    if title:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(grid)
        draw.text((6, 4), title, fill=(255, 255, 255), font=font)

    for i, img in enumerate(valid_images):
        r, c = divmod(i, ncols)
        grid.paste(img, (c * thumb_size, r * thumb_size + title_height))
    return grid


def build_top_neurons_positive_only(
    selectivity: torch.Tensor,
    attribute_names: list[str],
    top_k: int,
) -> dict[str, list[tuple[int, float]]]:
    """For each attribute, return the top-k neurons with positive selectivity."""
    result: dict[str, list[tuple[int, float]]] = {}
    for attr_idx, name in enumerate(attribute_names):
        row = selectivity[attr_idx]
        top_vals, top_idxs = torch.topk(row, k=min(top_k, len(row)))
        entries = [
            (idx, val)
            for idx, val in zip(top_idxs.tolist(), top_vals.tolist())
            if val > 0
        ]
        result[name] = entries
    return result


def build_clip_attributes(
    queries: list[dict[str, list[str]]],
    clip_templates: torch.Tensor,
    device: torch.device,
    samples_per_class: int,
) -> tuple[AttributeSource, dict[str, dict], list[list[int]], list[list[int]], torch.Tensor]:
    """Score CLIP queries against clip_templates and produce binary pos/neg attributes.

    Returns (source, query_meta, pos_indices_all, neg_indices_all, unique_indices).
    """
    from sae_analysis.clip_zero_shot import encode_text_openclip, process_query

    all_prompts: list[str] = []
    for q in queries:
        for prompts in q.values():
            all_prompts.extend(prompts)
    unique_prompts = list(dict.fromkeys(all_prompts))

    print(f"Encoding {len(unique_prompts)} unique text prompts across {len(queries)} queries...")
    text_emb_cache: dict[str, torch.Tensor] = {}
    all_embs, logit_scale = encode_text_openclip(unique_prompts, device)
    for prompt, emb in zip(unique_prompts, all_embs):
        text_emb_cache[prompt] = emb
    print(f"  CLIP logit_scale = {logit_scale:.2f}")

    attribute_names: list[str] = []
    pos_indices_all: list[list[int]] = []
    neg_indices_all: list[list[int]] = []
    query_meta: dict[str, dict] = {}

    for qi, query in enumerate(queries):
        query_label = " vs ".join(query.keys())
        print(f"Processing query {qi + 1}/{len(queries)}: {query_label}")

        class_names, pooled_probs, winners = process_query(
            query, clip_templates, text_emb_cache, device, logit_scale,
        )
        n_classes = len(class_names)

        if n_classes == 2:
            attr_name = f"clip_{class_names[0]}"
            pos_order = np.argsort(pooled_probs[:, 0])[::-1]
            neg_order = np.argsort(pooled_probs[:, 1])[::-1]
            pos_sel = pos_order[:samples_per_class].tolist()
            neg_sel = neg_order[:samples_per_class].tolist()

            attribute_names.append(attr_name)
            pos_indices_all.append(pos_sel)
            neg_indices_all.append(neg_sel)
            query_meta[attr_name] = {
                "query_idx": qi,
                "class_names": class_names,
                "pos_class_idx": 0,
                "neg_class_idx": 1,
                "pooled_probs": pooled_probs,
                "winners": winners,
            }
        else:
            for cls_idx, cls_name in enumerate(class_names):
                attr_name = f"clip_{cls_name}"
                pos_order = np.argsort(pooled_probs[:, cls_idx])[::-1]
                pos_sel = pos_order[:samples_per_class].tolist()

                other_max = np.max(np.delete(pooled_probs, cls_idx, axis=1), axis=1)
                neg_order = np.argsort(other_max)[::-1]
                neg_sel = neg_order[:samples_per_class].tolist()

                attribute_names.append(attr_name)
                pos_indices_all.append(pos_sel)
                neg_indices_all.append(neg_sel)
                query_meta[attr_name] = {
                    "query_idx": qi,
                    "class_names": class_names,
                    "pos_class_idx": cls_idx,
                    "neg_class_idx": None,
                    "pooled_probs": pooled_probs,
                    "winners": winners,
                }

    n_total = clip_templates.shape[0]
    n_attrs = len(attribute_names)
    attrs_bin = torch.zeros(n_total, n_attrs, dtype=torch.uint8)
    for attr_idx in range(n_attrs):
        for i in pos_indices_all[attr_idx]:
            attrs_bin[i, attr_idx] = 1

    all_idx: set[int] = set()
    for lst in pos_indices_all:
        all_idx.update(lst)
    for lst in neg_indices_all:
        all_idx.update(lst)
    unique_indices = torch.tensor(sorted(all_idx), dtype=torch.long)

    print(f"  {n_attrs} attributes, {len(unique_indices)} unique samples")

    source = AttributeSource(
        name="clip_zero_shot",
        attributes=attrs_bin,
        attribute_names=attribute_names,
    )
    return source, query_meta, pos_indices_all, neg_indices_all, unique_indices


def full_dataset_scan(
    model: torch.nn.Module,
    embeddings: torch.Tensor,
    neuron_indices: list[int],
    top_k: int,
    batch_size: int,
    device: torch.device,
    identities: np.ndarray,
    query_meta: dict[str, dict],
    top_neurons_per_attr: dict[str, list[tuple[int, float]]],
    neurons_per_attr: int,
) -> tuple[
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, dict[int, tuple[int, int]]],
]:
    """Single full-dataset SAE pass producing per-neuron top-k unique-ID samples plus
    per-(attr, neuron) (total_fires, agree_fires) counts for CLIP agreeance.
    """
    oversample_factor = 8
    candidate_k = top_k * oversample_factor

    n_neurons = len(neuron_indices)
    neuron_idx_tensor = torch.tensor(neuron_indices, dtype=torch.long)
    neuron_to_pos = {n: i for i, n in enumerate(neuron_indices)}
    num_samples = embeddings.shape[0]
    dtype = model.W_enc.dtype

    top_vals = torch.full((n_neurons, candidate_k), -float("inf"))
    top_idxs = torch.full((n_neurons, candidate_k), -1, dtype=torch.long)

    attr_specs: list[tuple[str, int, np.ndarray, list[int]]] = []
    for attr_name, neurons in top_neurons_per_attr.items():
        meta = query_meta[attr_name]
        winners = meta["winners"]
        pos_cls_idx = meta["pos_class_idx"]
        neuron_positions = [
            neuron_to_pos[n_idx]
            for n_idx, _ in neurons[:neurons_per_attr]
            if n_idx in neuron_to_pos
        ]
        attr_specs.append((attr_name, pos_cls_idx, winners, neuron_positions))

    total_fires = torch.zeros(n_neurons, dtype=torch.long)
    attr_agree: dict[str, torch.Tensor] = {
        name: torch.zeros(n_neurons, dtype=torch.long) for name, _, _, _ in attr_specs
    }

    for start in tqdm(range(0, num_samples, batch_size), desc="Full-dataset scan"):
        end = min(start + batch_size, num_samples)
        batch = embeddings[start:end].to(device=device, dtype=dtype)
        with torch.no_grad():
            _, features, _ = model(batch)
        batch_acts = features[:, neuron_idx_tensor].cpu()

        batch_global_idxs = torch.arange(start, end, dtype=torch.long)
        batch_acts_t = batch_acts.t()
        batch_idxs_t = batch_global_idxs.unsqueeze(0).expand(n_neurons, -1)
        cat_vals = torch.cat([top_vals, batch_acts_t], dim=1)
        cat_idxs = torch.cat([top_idxs, batch_idxs_t], dim=1)
        top_sel = cat_vals.topk(candidate_k, dim=1)
        top_vals = top_sel.values
        top_idxs = cat_idxs.gather(1, top_sel.indices)

        fired = batch_acts > 0
        total_fires += fired.sum(dim=0).long()
        for attr_name, pos_cls_idx, winners, neuron_positions in attr_specs:
            batch_winners = winners[start:end]
            agrees = torch.from_numpy((batch_winners == pos_cls_idx).astype(np.uint8)).bool()
            for npos in neuron_positions:
                mask = fired[:, npos] & agrees
                attr_agree[attr_name][npos] += mask.sum().long()

    global_top_bottom: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for i, neuron_idx in enumerate(neuron_indices):
        cand_idxs = top_idxs[i]
        seen_ids: set[str] = set()
        selected: list[int] = []
        for sample_idx in cand_idxs.tolist():
            if sample_idx < 0:
                continue
            identity = identities[sample_idx]
            if identity in seen_ids:
                continue
            seen_ids.add(identity)
            selected.append(sample_idx)
            if len(selected) >= top_k:
                break
        while len(selected) < top_k:
            selected.append(-1)
        global_top_bottom[neuron_idx] = (
            torch.tensor(selected, dtype=torch.long),
            torch.full((top_k,), -1, dtype=torch.long),
        )

    full_agreeance: dict[str, dict[int, tuple[int, int]]] = {}
    for attr_name, _, _, neuron_positions in attr_specs:
        full_agreeance[attr_name] = {}
        for npos in neuron_positions:
            n_idx = neuron_indices[npos]
            full_agreeance[attr_name][n_idx] = (
                int(total_fires[npos].item()),
                int(attr_agree[attr_name][npos].item()),
            )

    return global_top_bottom, full_agreeance


def compute_clip_agreeance(
    top_neurons_per_attr: dict[str, list[tuple[int, float]]],
    global_top_bottom: dict[int, tuple[torch.Tensor, torch.Tensor]],
    query_meta: dict[str, dict],
    neurons_per_attr: int,
    full_agreeance_stats: dict[str, dict[int, tuple[int, int]]],
) -> dict[str, list[tuple[int, float, float, float]]]:
    """For each (attribute, neuron): selectivity, top-grid CLIP agreeance, full-dataset agreeance.

    Returns {attr_name: [(neuron_idx, selectivity, top_agreeance, full_agreeance), ...]}
    sorted by top_agreeance desc, then |selectivity| desc.
    """
    result: dict[str, list[tuple[int, float, float, float]]] = {}

    for attr_name, neurons in top_neurons_per_attr.items():
        meta = query_meta[attr_name]
        winners = meta["winners"]
        pos_cls_idx = meta["pos_class_idx"]
        full_stats = full_agreeance_stats.get(attr_name, {})

        entries: list[tuple[int, float, float, float]] = []
        for neuron_idx, selectivity in neurons[:neurons_per_attr]:
            top_agree = 0.0
            if neuron_idx in global_top_bottom:
                t_idxs, _ = global_top_bottom[neuron_idx]
                valid_idxs = [i for i in t_idxs.tolist() if i >= 0]
                if valid_idxs:
                    agree_count = sum(1 for i in valid_idxs if winners[i] == pos_cls_idx)
                    top_agree = agree_count / len(valid_idxs)

            full_agree = 0.0
            if neuron_idx in full_stats:
                total_fires, agree_fires = full_stats[neuron_idx]
                if total_fires > 0:
                    full_agree = agree_fires / total_fires

            entries.append((neuron_idx, selectivity, top_agree, full_agree))

        entries.sort(key=lambda x: (x[2], abs(x[1])), reverse=True)
        result[attr_name] = entries

    return result


def save_attribute_grids_with_agreeance(
    global_top_bottom: dict[int, tuple[torch.Tensor, torch.Tensor]],
    neurons_with_agreeance: dict[str, list[tuple[int, float, float, float]]],
    attribute_names: list[str],
    ids_and_fns: np.ndarray,
    image_root: Path,
    output_dir: Path,
    neurons_per_attr: int = 3,
    identity_diversity: dict[int, int] | None = None,
) -> None:
    """Save image grids with selectivity and agreeance scores in the title bar."""
    output_dir.mkdir(parents=True, exist_ok=True)
    id_div = identity_diversity or {}

    for attr_name in tqdm(attribute_names, desc="Generating grids"):
        neurons = neurons_with_agreeance.get(attr_name, [])[:neurons_per_attr]
        if not neurons:
            continue

        attr_dir = output_dir / attr_name.lower().replace(" ", "_")
        attr_dir.mkdir(parents=True, exist_ok=True)

        for rank, (neuron_idx, selectivity, top_agree, full_agree) in enumerate(neurons):
            if neuron_idx not in global_top_bottom:
                continue
            t_idxs, _ = global_top_bottom[neuron_idx]

            max_paths = [
                resolve_image_path_from_ids(i, ids_and_fns, image_root)
                for i in t_idxs.tolist()
                if i >= 0
            ]

            sel_str = f"{selectivity:+.3f}"
            top_str = f"{top_agree:.1%}"
            full_str = f"{full_agree:.1%}"
            prefix = f"rank{rank}_n{neuron_idx}_sel{sel_str}_top{top_str}_full{full_str}"

            ids_part = f"  |  {id_div[neuron_idx]} IDs" if neuron_idx in id_div else ""
            max_title = (
                f"{attr_name} | n{neuron_idx} sel={sel_str}"
                f" top={top_str} full={full_str}"
                f" | TOP activating{ids_part}"
            )

            max_grid = make_image_grid(max_paths, title=max_title)
            if max_grid is not None:
                max_grid.save(attr_dir / f"{prefix}_max.jpg", quality=90)


def build_clip_report(
    attribute_names: list[str],
    neurons_with_agreeance: dict[str, list[tuple[int, float, float, float]]],
    metrics: dict[str, torch.Tensor],
    model_info: dict[str, object],
    samples_per_class: int,
    identity_diversity: dict[int, int] | None = None,
    top_k_display: int = 5,
) -> str:
    """Build a plain-text report including top and full agreeance scores."""
    lines: list[str] = []
    id_div = identity_diversity or {}

    lines.append("=" * 100)
    lines.append("CLIP-BASED SAE ATTRIBUTE ANALYSIS REPORT")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"Weights:            {model_info.get('weights_path', 'N/A')}")
    lines.append(f"d_in:               {model_info.get('d_in', 'N/A')}")
    lines.append(f"d_sae:              {model_info.get('d_sae', 'N/A')}")
    lines.append(f"Threshold:          {model_info.get('threshold', 'N/A')}")
    lines.append(f"Samples/class:      {samples_per_class}")
    lines.append(f"Attributes:         {len(attribute_names)}")
    lines.append("")

    selectivity = metrics["selectivity"]
    max_abs_sel = selectivity.abs().max(dim=0).values
    d_sae = selectivity.shape[1]
    strong_01 = int((max_abs_sel > 0.1).sum().item())
    strong_02 = int((max_abs_sel > 0.2).sum().item())
    strong_03 = int((max_abs_sel > 0.3).sum().item())
    lines.append(f"Neurons with |selectivity| > 0.1: {strong_01}/{d_sae}")
    lines.append(f"Neurons with |selectivity| > 0.2: {strong_02}/{d_sae}")
    lines.append(f"Neurons with |selectivity| > 0.3: {strong_03}/{d_sae}")
    lines.append("")

    lines.append("=" * 100)
    lines.append("ATTRIBUTE -> TOP NEURONS (sorted by top agreeance)")
    lines.append("  top=agreeance on grid images, full=agreeance on all firing samples")
    lines.append("=" * 100)

    fire_pos = metrics["fire_rate_pos"]
    fire_neg = metrics["fire_rate_neg"]

    for attr_idx, name in enumerate(attribute_names):
        entries = neurons_with_agreeance.get(name, [])
        parts = []
        for neuron_idx, sel, top_ag, full_ag in entries[:top_k_display]:
            ids_str = f",{id_div[neuron_idx]}ids" if neuron_idx in id_div else ""
            parts.append(
                f"n{neuron_idx}(sel={sel:+.3f},top={top_ag:.0%},full={full_ag:.0%}{ids_str})"
            )
        neurons_str = ", ".join(parts)
        lines.append(f"  {name:<30s} {neurons_str}")

    lines.append("")
    lines.append("=" * 100)
    lines.append("DETAILED PER-ATTRIBUTE BREAKDOWN")
    lines.append("=" * 100)

    for attr_idx, name in enumerate(attribute_names):
        entries = neurons_with_agreeance.get(name, [])
        if not entries:
            continue
        lines.append("")
        lines.append(f"--- {name} ---")
        lines.append(
            f"  {'Neuron':<10s} {'Sel':>8s} {'TopAgr':>8s} {'FullAgr':>8s} {'FR+':>8s} {'FR-':>8s} {'IDs':>6s}"
        )
        for neuron_idx, sel, top_ag, full_ag in entries[:top_k_display]:
            fr_p = fire_pos[attr_idx, neuron_idx].item()
            fr_n = fire_neg[attr_idx, neuron_idx].item()
            ids_str = str(id_div[neuron_idx]) if neuron_idx in id_div else "-"
            lines.append(
                f"  n{neuron_idx:<8d} {sel:>+8.3f} {top_ag:>7.1%} {full_ag:>7.1%} {fr_p:>8.3f} {fr_n:>8.3f} {ids_str:>6s}"
            )

    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines)
