from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.path_setup import ensure_recognizers_on_path

ensure_recognizers_on_path()

DFD_ROOT = Path(__file__).resolve().parent
SCOUT_ROOT = DFD_ROOT.parent
FEATURE_EXTRACTORS_ROOT = SCOUT_ROOT / "feature_extractors"


def bgr_to_rgb(bgr: torch.Tensor) -> torch.Tensor:
    return bgr[:, [2, 1, 0], :, :]


def bgr_to_rgb01(bgr: torch.Tensor) -> torch.Tensor:
    rgb = bgr_to_rgb(bgr)
    return (rgb + 1.0) / 2.0


def _layernorm_chw(feat: torch.Tensor) -> torch.Tensor:
    """Per-sample LayerNorm over (C, H, W). Standard normalization for all native perceptuals."""
    return F.layer_norm(feat, feat.shape[1:], eps=1e-6)


# ---------------------------------------------------------------------------
# ImageNet-pretrained VGG16 (original implementation)
# ---------------------------------------------------------------------------
class VGG16FeatureExtractor(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights

        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # relu4_3 is layer index 22 in torchvision VGG16 features
        self.features = vgg[:23].eval()
        for p in self.features.parameters():
            p.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        if device is not None:
            self.to(device)

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        pred = bgr_to_rgb01(bgr_pred)
        true = bgr_to_rgb01(bgr_true)
        pred = (pred - self.mean) / self.std
        true = (true - self.mean) / self.std
        feat_pred = self.features(pred)
        feat_true = self.features(true)
        return F.mse_loss(feat_pred, feat_true)


# ---------------------------------------------------------------------------
# VGGFace-pretrained VGG16 (face-specific features, matches the DFD paper)
# Weights must be converted once using utils/convert_vggface_weights.py
# ---------------------------------------------------------------------------
VGGFACE_WEIGHTS_PATH = DFD_ROOT / "pretrained" / "vggface_vgg16_features.pth"

# VGGFace v1 preprocessing: BGR mean subtraction (no std division)
VGGFACE_BGR_MEAN = torch.tensor([93.5940, 104.7624, 129.1863]).view(1, 3, 1, 1)


class VGGFaceFeatureExtractor(nn.Module):
    """VGG16 pretrained on VGGFace for face-specific perceptual loss.

    Matches the original DFD paper which uses keras_vggface VGG16 at conv4_3.
    Preprocessing: BGR [0, 255] with mean subtraction (no std normalization).
    """

    def __init__(self, weights_path: Optional[Path] = None, device: Optional[torch.device] = None):
        super().__init__()
        from torchvision.models import vgg16

        weights_path = Path(weights_path) if weights_path else VGGFACE_WEIGHTS_PATH
        if not weights_path.exists():
            raise FileNotFoundError(
                f"VGGFace weights not found at {weights_path}. "
                "Run: python utils/convert_vggface_weights.py"
            )

        vgg = vgg16(weights=None).features
        state = torch.load(weights_path, map_location=device, weights_only=True)
        vgg.load_state_dict(state)
        # relu4_3 = layer index 22 (same structure as ImageNet VGG16)
        self.features = vgg[:23].eval()
        for p in self.features.parameters():
            p.requires_grad = False

        # VGGFace v1 uses BGR input with mean subtraction (pixel range [0, 255])
        self.register_buffer("bgr_mean", VGGFACE_BGR_MEAN.clone())

        if device is not None:
            self.to(device)

    def _preprocess(self, bgr_m1p1: torch.Tensor) -> torch.Tensor:
        """Convert BGR [-1, 1] -> BGR [0, 255] - mean."""
        bgr255 = (bgr_m1p1 + 1.0) * 127.5  # [-1,1] -> [0,255]
        return bgr255 - self.bgr_mean

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        pred = self._preprocess(bgr_pred)
        true = self._preprocess(bgr_true)
        feat_pred = self.features(pred)
        feat_true = self.features(true)
        return F.mse_loss(feat_pred, feat_true)


# ---------------------------------------------------------------------------
# AdaFace IR-50 intermediate features as perceptual loss
# Uses the same frozen encoder already in the pipeline — zero extra deps.
# Trained on MS1MV2 (5.8M images), more modern than VGGFace.
# ---------------------------------------------------------------------------
class AdaFaceFeatureExtractor(nn.Module):
    """Extract intermediate features from a frozen AdaFace IR-50 backbone.

    IR-50 body structure (24 BasicBlockIR modules):
      body[0:3]   -> block1: 64ch,  56x56
      body[3:7]   -> block2: 128ch, 28x28
      body[7:21]  -> block3: 256ch, 14x14
      body[21:24] -> block4: 512ch, 7x7

    Default: extract after block2 (28x28) and block3 (14x14).
    The DFD paper's conv4_3 analog is block3 (14x14, 256ch).
    """

    # Block boundary indices (inclusive end of each block)
    BLOCK_ENDS = {"block1": 2, "block2": 6, "block3": 20, "block4": 23}

    def __init__(
        self,
        ckpt_path: Optional[Path] = None,
        layers: Tuple[str, ...] = ("block2", "block3"),
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.layer_indices = sorted(self.BLOCK_ENDS[l] for l in layers)

        from feature_extractors.adaface_ms1mv2_ires50 import inference as adaface_inference

        ckpt_path = ckpt_path or (
            FEATURE_EXTRACTORS_ROOT
            / "adaface_ms1mv2_ires50"
            / "pretrained"
            / "adaface_ir50_ms1mv2.ckpt"
        )
        model = adaface_inference.load_model(ckpt_path=ckpt_path, device=device)

        self.input_layer = model.input_layer
        self.body = model.body

        for p in self.parameters():
            p.requires_grad = False
        if device is not None:
            self.to(device)

    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract intermediate features at configured block boundaries."""
        x = self.input_layer(x)
        feats: List[torch.Tensor] = []
        for idx, module in enumerate(self.body):
            x = module(x)
            if idx in self.layer_indices:
                feats.append(x)
        return feats

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        """Input: BGR tensors in [-1, 1] (same as AdaFace expects)."""
        pred_feats = [_layernorm_chw(f) for f in self._extract(bgr_pred)]
        true_feats = [_layernorm_chw(f) for f in self._extract(bgr_true)]
        loss = sum(F.mse_loss(pf, tf) for pf, tf in zip(pred_feats, true_feats))
        return loss


# ---------------------------------------------------------------------------
# AdaFace ViT-B intermediate features as perceptual loss
# ---------------------------------------------------------------------------
UPSAMPLE_MODES = ("none", "bilinear")

VIT_BLOCK_MAP: Dict[str, int] = {
    "quarter1": 5,
    "third1": 7,
    "half": 11,
    "third2": 15,
    "quarter3": 17,
    "final": 23,
}

VIT_GRID_SIZE = 14


class AdaFaceViTFeatureExtractor(nn.Module):
    """Extract intermediate features from a frozen AdaFace ViT-B backbone."""

    def __init__(
        self,
        ckpt_path: Optional[Path] = None,
        layers: Tuple[str, ...] = ("third1", "third2"),
        upsample: str = "none",
        upsample_size: int = 112,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if upsample not in UPSAMPLE_MODES:
            raise ValueError(f"Unknown upsample: {upsample}. Choose from {UPSAMPLE_MODES}")
        self.upsample = upsample
        self.upsample_size = upsample_size

        self.capture_indices: List[int] = sorted(VIT_BLOCK_MAP[l] for l in layers)
        self.max_block = max(self.capture_indices)

        from feature_extractors.adaface_webface4m_vitb import inference as adaface_vit_inference

        ckpt_path = ckpt_path or (
            FEATURE_EXTRACTORS_ROOT
            / "adaface_webface4m_vitb"
            / "pretrained"
            / "adaface_webface4m_vitb.pt"
        )
        model = adaface_vit_inference.load_model(ckpt_path=ckpt_path, device=device)

        self.input_color_flip = model.input_color_flip
        self.patch_embed = model.net.patch_embed
        self.pos_embed = model.net.pos_embed
        self.pos_drop = model.net.pos_drop
        self.blocks = model.net.blocks

        for p in self.parameters():
            p.requires_grad = False
        if device is not None:
            self.to(device)

    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.input_color_flip:
            x = x.flip(1)

        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        feats: List[torch.Tensor] = []
        capture_set = set(self.capture_indices)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in capture_set:
                batch, _, channels = x.shape
                feats.append(x.transpose(1, 2).reshape(batch, channels, VIT_GRID_SIZE, VIT_GRID_SIZE))
            if i >= self.max_block:
                break
        return feats

    def _upsample_feature(self, feat: torch.Tensor) -> torch.Tensor:
        if self.upsample == "none" or feat.shape[-1] >= self.upsample_size:
            return feat
        return F.interpolate(feat, size=self.upsample_size, mode="bilinear", align_corners=False)

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        pred_feats = [_layernorm_chw(f) for f in self._extract(bgr_pred)]
        true_feats = [_layernorm_chw(f) for f in self._extract(bgr_true)]
        if not pred_feats:
            return bgr_pred.new_tensor(0.0)
        return sum(
            F.mse_loss(self._upsample_feature(pf), self._upsample_feature(tf))
            for pf, tf in zip(pred_feats, true_feats)
        )


class AdaFaceViTPerceptualLoss(nn.Module):
    def __init__(
        self,
        upsample: str = "none",
        upsample_size: int = 112,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.impl = AdaFaceViTFeatureExtractor(
            upsample=upsample,
            upsample_size=upsample_size,
            device=device,
        )

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        return self.impl(bgr_pred, bgr_true)


# ---------------------------------------------------------------------------
# ArcFace WebFace42M iResNet-100 intermediate features as perceptual loss
# ---------------------------------------------------------------------------
ARC2FACE_TAP_POINTS: Dict[str, str] = {
    "block1": "layer1/layer1/2/Add",
    "block2": "layer2/layer2/12/Add",
    "block3": "layer3/layer3/29/Add",
    "block4": "layer4/layer4/2/Add",
}


class Arc2FaceArcFaceFeatureExtractor(nn.Module):
    """Extract intermediate features from the ArcFace WebFace42M iResNet-100 model."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        layers: Tuple[str, ...] = ("block2", "block3"),
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.tap_names: List[str] = [ARC2FACE_TAP_POINTS[l] for l in layers]
        self._features: Dict[str, torch.Tensor] = {}

        from feature_extractors.arcface_webface42m_res100 import inference as arcface_inference

        model_path = model_path or (
            FEATURE_EXTRACTORS_ROOT
            / "arcface_webface42m_res100"
            / "pretrained"
            / "arc2face_arcface_r100_model.pth"
        )
        self.model = arcface_inference.load_model(model_path=model_path, device=device)

        for p in self.model.parameters():
            p.requires_grad = False

        tap_set = set(self.tap_names)
        for name, mod in self.model.named_modules():
            if name in tap_set:
                mod.register_forward_hook(self._make_hook(name))

        if device is not None:
            self.to(device)

    def _make_hook(self, name: str):
        def hook(mod, inp, out):
            self._features[name] = out

        return hook

    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        self._features.clear()
        self.model(x)
        return [self._features[n] for n in self.tap_names if n in self._features]

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        pred_feats = [_layernorm_chw(f) for f in self._extract(bgr_pred)]
        true_feats = [_layernorm_chw(f) for f in self._extract(bgr_true)]
        if not pred_feats:
            return bgr_pred.new_tensor(0.0)
        return sum(F.mse_loss(pf, tf) for pf, tf in zip(pred_feats, true_feats))


class Arc2FaceArcFacePerceptualLoss(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.impl = Arc2FaceArcFaceFeatureExtractor(device=device)

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        return self.impl(bgr_pred, bgr_true)


# ---------------------------------------------------------------------------
# SwinFace Swin-T intermediate features as perceptual loss
# ---------------------------------------------------------------------------
SWIN_STAGE_MAP: Dict[str, int] = {"stage0": 0, "stage1": 1, "stage2": 2, "stage3": 3}


class SwinFaceFeatureExtractor(nn.Module):
    """Extract intermediate features from a frozen SwinFace Swin-T backbone."""

    def __init__(
        self,
        ckpt_path: Optional[Path] = None,
        layers: Tuple[str, ...] = ("stage1", "stage2"),
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.capture_stages: List[int] = sorted(SWIN_STAGE_MAP[l] for l in layers)
        self.max_stage = max(self.capture_stages)

        from feature_extractors.swinface_ms1mv2_swint import inference as swinface_inference

        ckpt_path = ckpt_path or (
            FEATURE_EXTRACTORS_ROOT
            / "swinface_ms1mv2_swint"
            / "pretrained"
            / "checkpoint_step_79999_gpu_0.pt"
        )
        self.backbone = swinface_inference.load_model(ckpt_path=ckpt_path, device=device)

        for p in self.backbone.parameters():
            p.requires_grad = False
        if device is not None:
            self.to(device)

    def _extract(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = x[:, [2, 1, 0], :, :]
        x = self.backbone.patch_embed(x)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)

        feats: List[torch.Tensor] = []
        capture_set = set(self.capture_stages)
        for i, layer in enumerate(self.backbone.layers):
            for block in layer.blocks:
                x = block(x)

            if i in capture_set:
                batch, tokens, channels = x.shape
                side = int(tokens**0.5)
                feats.append(x.transpose(1, 2).reshape(batch, channels, side, side))

            if i >= self.max_stage:
                break

            if layer.downsample is not None:
                x = layer.downsample(x)

        return feats

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        pred_feats = [_layernorm_chw(f) for f in self._extract(bgr_pred)]
        true_feats = [_layernorm_chw(f) for f in self._extract(bgr_true)]
        if not pred_feats:
            return bgr_pred.new_tensor(0.0)
        return sum(F.mse_loss(pf, tf) for pf, tf in zip(pred_feats, true_feats))


class SwinFacePerceptualLoss(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        self.impl = SwinFaceFeatureExtractor(device=device)

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        return self.impl(bgr_pred, bgr_true)


# ---------------------------------------------------------------------------
# LPIPS (learned perceptual metric)
# ---------------------------------------------------------------------------
class LPIPSFeatureExtractor(nn.Module):
    def __init__(self, device: Optional[torch.device] = None):
        super().__init__()
        try:
            import lpips  # type: ignore
        except Exception as exc:
            raise ImportError("lpips is not installed. Add it to requirements and install.") from exc
        self.lpips = lpips.LPIPS(net="vgg").eval()
        for p in self.lpips.parameters():
            p.requires_grad = False
        if device is not None:
            self.lpips.to(device)

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        pred = bgr_to_rgb(bgr_pred)
        true = bgr_to_rgb(bgr_true)
        loss = self.lpips(pred, true)
        return loss.mean()


# ---------------------------------------------------------------------------
# Unified perceptual loss wrapper
# ---------------------------------------------------------------------------
class PerceptualLoss(nn.Module):
    MODES = (
        "vgg16",
        "vggface",
        "adaface",
        "adaface_vit",
        "arc2face_arcface",
        "swinface",
        "lpips",
        "none",
    )

    def __init__(self, mode: str = "vgg16", device: Optional[torch.device] = None):
        super().__init__()
        self.mode = mode
        if mode == "vgg16":
            self.impl = VGG16FeatureExtractor(device=device)
        elif mode == "vggface":
            self.impl = VGGFaceFeatureExtractor(device=device)
        elif mode == "adaface":
            self.impl = AdaFaceFeatureExtractor(device=device)
        elif mode == "adaface_vit":
            self.impl = AdaFaceViTPerceptualLoss(device=device)
        elif mode == "arc2face_arcface":
            self.impl = Arc2FaceArcFacePerceptualLoss(device=device)
        elif mode == "swinface":
            self.impl = SwinFacePerceptualLoss(device=device)
        elif mode == "lpips":
            self.impl = LPIPSFeatureExtractor(device=device)
        elif mode == "none":
            self.impl = None
        else:
            raise ValueError(f"Unknown perceptual mode: {mode}. Choose from {self.MODES}")

    def forward(self, bgr_pred: torch.Tensor, bgr_true: torch.Tensor) -> torch.Tensor:
        if self.impl is None:
            return bgr_pred.new_tensor(0.0)
        return self.impl(bgr_pred, bgr_true)
