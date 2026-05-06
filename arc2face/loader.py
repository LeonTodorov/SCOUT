"""Arc2Face pipeline loader and template-conditioned generation helpers.

Loads CLIPTextModelWrapper + UNet from `arc2face/pretrained/{encoder,arc2face}/`
and composes them with a Stable Diffusion 1.5 base into a `StableDiffusionPipeline`.
Mirrors the loading pattern from facemint-dfd/manipulate_arc2face.py.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .models import CLIPTextModelWrapper
from .utils import project_face_embs

PRETRAINED = Path(__file__).resolve().parent / "pretrained"

DEFAULT_BASE_MODEL = str(PRETRAINED / "stable-diffusion-v1-5")
DEFAULT_ARC2FACE_MODEL = str(PRETRAINED)


def resolve_local_snapshot(model_id_or_path: str, cache_dir: Optional[str] = None) -> Path:
    """Resolve a model id or path to a local directory.

    If `model_id_or_path` is an existing local directory, return it. Otherwise look
    under HuggingFace's hub cache layout (`<cache>/models--<org>--<name>/snapshots/<hash>/`).
    """
    candidate = Path(model_id_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    if cache_dir is None:
        raise FileNotFoundError(
            f"{model_id_or_path!r} is not a local path and no cache_dir was provided."
        )
    model_key = "models--" + model_id_or_path.replace("/", "--")
    snapshots_dir = Path(cache_dir).expanduser().resolve() / model_key / "snapshots"
    if not snapshots_dir.exists():
        raise FileNotFoundError(
            f"No cached snapshot for {model_id_or_path!r} in {cache_dir}. "
            "Download it first or pass a local path."
        )
    snapshots = sorted(snapshots_dir.iterdir())
    if not snapshots:
        raise FileNotFoundError(f"No snapshots under: {snapshots_dir}")
    return snapshots[-1]


def load_arc2face_pipeline(
    device: torch.device,
    arc2face_model: str = DEFAULT_ARC2FACE_MODEL,
    base_model: str = DEFAULT_BASE_MODEL,
    cache_dir: Optional[str] = None,
) -> tuple[object, torch.dtype]:
    """Load the Arc2Face Stable Diffusion pipeline from local checkpoints.

    Expected layout under `arc2face_model` (default: arc2face/pretrained/):
        encoder/{config.json, pytorch_model.bin}
        arc2face/{config.json, diffusion_pytorch_model.safetensors}
    """
    from diffusers import (
        DPMSolverMultistepScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    arc2face_path = resolve_local_snapshot(arc2face_model, cache_dir)
    base_path = resolve_local_snapshot(base_model, cache_dir)

    encoder = CLIPTextModelWrapper.from_pretrained(
        str(arc2face_path), subfolder="encoder", torch_dtype=dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        str(arc2face_path), subfolder="arc2face", torch_dtype=dtype,
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        str(base_path),
        text_encoder=encoder,
        unet=unet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(device)
    pipeline.set_progress_bar_config(disable=True)
    return pipeline, dtype


@torch.no_grad()
def encode_input_image(
    recognizer: torch.nn.Module,
    img: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """Encode a PIL image with the Arc2Face arcface_webface42m_res100 recognizer.

    Returns the normalised template tensor [1, d_emb] on `device`.
    """
    from feature_extractors.arcface_webface42m_res100.inference import (
        encode_batch,
        pil_to_bgr_tensor,
    )

    x = pil_to_bgr_tensor(img).unsqueeze(0).to(device)
    template, _ = encode_batch(recognizer, x)
    return template


@torch.no_grad()
def generate_from_embedding(
    pipeline,
    dtype: torch.dtype,
    emb: torch.Tensor,
    num_steps: int,
    guidance_scale: float,
    seed: int,
) -> Image.Image:
    """Generate a face image from a face-recognition embedding via Arc2Face."""
    prompt_embeds = project_face_embs(pipeline, emb.to(dtype))
    generator = torch.Generator(device=pipeline.device).manual_seed(seed)
    return pipeline(
        prompt_embeds=prompt_embeds,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=1,
        generator=generator,
    ).images[0]
