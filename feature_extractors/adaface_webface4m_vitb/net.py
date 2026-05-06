from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
from timm.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[[], nn.Module] = nn.ReLU6,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_token, embed_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            num_token,
            3,
            self.num_heads,
            embed_dim // self.num_heads,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        act_layer: Callable[[], nn.Module] = nn.ReLU6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 512,
    ) -> None:
        super().__init__()
        img_size_t = to_2tuple(img_size)
        patch_size_t = to_2tuple(patch_size)
        num_patches = (img_size_t[1] // patch_size_t[1]) * (img_size_t[0] // patch_size_t[0])
        self.img_size = img_size_t
        self.patch_size = patch_size_t
        self.num_patches = num_patches
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size_t,
            stride=patch_size_t,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height != self.img_size[0] or width != self.img_size[1]:
            raise RuntimeError(
                f"Input image size ({height}*{width}) does not match model "
                f"({self.img_size[0]}*{self.img_size[1]})."
            )
        return self.proj(x).flatten(2).transpose(1, 2)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size: int = 112,
        patch_size: int = 8,
        in_channels: int = 3,
        num_classes: int = 512,
        embed_dim: int = 512,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 3.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.feature = nn.Sequential(
            nn.Linear(in_features=embed_dim * self.num_patches, out_features=embed_dim, bias=False),
            nn.BatchNorm1d(num_features=embed_dim, eps=2e-5),
            nn.Linear(in_features=embed_dim, out_features=num_classes, bias=False),
            nn.BatchNorm1d(num_features=num_classes, eps=2e-5),
        )

        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x.float())
        return torch.reshape(x, (batch_size, self.num_patches * self.embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature(self.forward_features(x))


class AdaFaceViTModel(nn.Module):
    def __init__(
        self,
        name: str = "base",
        output_dim: int = 512,
        color_space: str = "RGB",
    ) -> None:
        super().__init__()
        if name != "base":
            raise ValueError(f"Unsupported AdaFace-ViT variant: {name}")
        self.input_color_flip = color_space.upper() == "BGR"
        self.net = VisionTransformer(
            img_size=112,
            patch_size=8,
            num_classes=output_dim,
            embed_dim=512,
            depth=24,
            mlp_ratio=3.0,
            num_heads=16,
            drop_path_rate=0.1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_color_flip:
            x = x.flip(1)
        return self.net(x)


def build_model(
    variant: str = "base",
    output_dim: int = 512,
    color_space: str = "RGB",
) -> AdaFaceViTModel:
    return AdaFaceViTModel(name=variant, output_dim=output_dim, color_space=color_space)
