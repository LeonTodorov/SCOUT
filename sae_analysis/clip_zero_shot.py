"""CLIP zero-shot helpers: query parsing, text encoding, and per-query softmax pooling."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from _paths import setup_paths

setup_paths()


Query = dict[str, list[str]]


def _torch_load_cpu(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def encode_text_openclip(prompts: list[str], device: torch.device) -> tuple[torch.Tensor, float]:
    """Encode text prompts via open_clip -> ([P, D] L2-normalized, logit_scale)."""
    import os

    import open_clip

    from feature_extractors.open_clip.inference import CLIP_MODEL_NAME, CLIP_PRETRAINED

    cache_dir = os.environ.get("OPENCLIP_CACHE_DIR")
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, cache_dir=cache_dir,
    )
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    logit_scale = model.logit_scale.exp().item()

    tokens = tokenizer(prompts).to(device)
    with torch.no_grad():
        text_emb = model.encode_text(tokens).float()
        text_emb = text_emb / (text_emb.norm(dim=1, keepdim=True) + 1e-8)
    return text_emb.cpu(), logit_scale


def _validate_query(q: object, label: str) -> Query:
    if not isinstance(q, dict):
        raise ValueError(f"{label} must be a JSON object, got {type(q).__name__}")
    for cls_name, prompts in q.items():
        if not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise ValueError(f"{label}, class '{cls_name}': value must be a list of strings")
        if len(prompts) == 0:
            raise ValueError(f"{label}, class '{cls_name}': must have at least one prompt")
    if len(q) < 2:
        raise ValueError(f"{label}: must have at least 2 classes, got {len(q)}")
    return q


def parse_queries(raw: list[str]) -> list[Query]:
    """Parse queries from inline JSON strings or a single .json file path.

    A single .json-file argument is loaded (must contain a list of query objects);
    otherwise each argument is parsed as an inline JSON object.
    """
    if len(raw) == 1 and raw[0].endswith(".json") and Path(raw[0]).is_file():
        with open(raw[0]) as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"JSON file must contain a list of query objects, got {type(data).__name__}")
        return [_validate_query(q, f"Query {i}") for i, q in enumerate(data)]

    queries: list[Query] = []
    for i, s in enumerate(raw):
        try:
            q = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Query {i} is not valid JSON: {e}") from e
        queries.append(_validate_query(q, f"Query {i}"))
    return queries


def process_query(
    query: Query,
    templates: torch.Tensor,
    text_emb_cache: dict[str, torch.Tensor],
    device: torch.device,
    logit_scale: float = 100.0,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Score templates against query prompts and pool by class.

    Returns:
        class_names:  list of class names
        pooled_probs: [N, n_classes] pooled softmax probabilities
        winners:      [N] index of winning class per sample
    """
    class_names = list(query.keys())

    all_prompts: list[str] = []
    prompt_to_class: list[int] = []
    for cls_idx, cls_name in enumerate(class_names):
        for prompt in query[cls_name]:
            all_prompts.append(prompt)
            prompt_to_class.append(cls_idx)

    uncached = [p for p in all_prompts if p not in text_emb_cache]
    if uncached:
        new_embs, _ = encode_text_openclip(uncached, device)
        for prompt, emb in zip(uncached, new_embs):
            text_emb_cache[prompt] = emb

    text_emb = torch.stack([text_emb_cache[p] for p in all_prompts])
    sims = templates @ text_emb.to(templates.dtype).T
    softmax_probs = torch.softmax(sims * logit_scale, dim=1).numpy()

    n_classes = len(class_names)
    pooled = np.zeros((templates.shape[0], n_classes), dtype=np.float32)
    for p_idx, cls_idx in enumerate(prompt_to_class):
        pooled[:, cls_idx] += softmax_probs[:, p_idx]

    winners = np.argmax(pooled, axis=1)
    return class_names, pooled, winners


def load_openclip(device: torch.device):
    """Load OpenCLIP model, preprocess transform, and tokenizer (matches encode_text_openclip)."""
    import os

    import open_clip

    from feature_extractors.open_clip.inference import CLIP_MODEL_NAME, CLIP_PRETRAINED

    cache_dir = os.environ.get("OPENCLIP_CACHE_DIR")
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED, cache_dir=cache_dir,
    )
    model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    return model, preprocess, tokenizer


@torch.no_grad()
def encode_text_features(
    clip_model, clip_tokenizer, pos_prompts: list[str], neg_prompts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute L2-normalised text features for pos/neg prompt sets."""
    pos_tokens = clip_tokenizer(pos_prompts).to(device)
    neg_tokens = clip_tokenizer(neg_prompts).to(device)
    pos_feats = clip_model.encode_text(pos_tokens).float()
    neg_feats = clip_model.encode_text(neg_tokens).float()
    pos_feats = pos_feats / (pos_feats.norm(dim=1, keepdim=True) + 1e-8)
    neg_feats = neg_feats / (neg_feats.norm(dim=1, keepdim=True) + 1e-8)
    return pos_feats, neg_feats


@torch.no_grad()
def openclip_score_batched(
    clip_model, clip_preprocess,
    images: list[Image.Image],
    pos_feats: torch.Tensor,
    neg_feats: torch.Tensor,
    device: torch.device,
    batch_size: int = 64,
) -> torch.Tensor:
    """Batched CLIP scoring: returns softmax P(positive) for each image, shape [N], in [0, 1]."""
    logit_scale = clip_model.logit_scale.exp()
    all_probs: list[torch.Tensor] = []
    n_batches = (len(images) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(images), batch_size), total=n_batches,
                  desc="      CLIP score", leave=False):
        batch_imgs = images[i : i + batch_size]
        img_tensors = torch.stack([clip_preprocess(img) for img in batch_imgs]).to(device)
        img_feats = clip_model.encode_image(img_tensors).float()
        img_feats = img_feats / (img_feats.norm(dim=1, keepdim=True) + 1e-8)
        pos_sim = (img_feats @ pos_feats.T).mean(dim=1)
        neg_sim = (img_feats @ neg_feats.T).mean(dim=1)
        logits = torch.stack([pos_sim, neg_sim], dim=1) * logit_scale
        probs = torch.softmax(logits, dim=1)[:, 0]
        all_probs.append(probs.cpu())
    return torch.cat(all_probs)
