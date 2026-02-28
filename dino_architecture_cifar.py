"""
DINO-style architecture adapted for CIFAR-10 (32x32).

Key design choices for small images:
- Patch size = 2 -> 16x16 = 256 patch tokens (better spatial granularity than 4x4).
- Lightweight ConvStem before patchifying to improve inductive bias on small images.
- Backbone returns both CLS token and patch tokens so we can support:
  - DINO CLS loss (global+local)
  - iBOT-style patch loss (global only, same resolution)
- Separate heads for CLS and patch tokens, each using a DINOHead-like projection.
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# init utils
# -----------------------------
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_.",
            stacklevel=2,
        )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# -----------------------------
# core blocks
# -----------------------------
def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
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
        num_heads: int,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path_prob: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = (
            DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ConvStem(nn.Module):
    """
    A small conv stem to improve inductive bias on tiny images.

    Input: [B,3,H,W]
    Output: [B,stem_dim,H,W]
    """

    def __init__(self, in_chans: int = 3, stem_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_chans, stem_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
            nn.Conv2d(
                stem_dim, stem_dim, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(stem_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding (conv patchify)."""

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_chans: int = 64,
        embed_dim: int = 256,
    ):
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        # x: [B,C,H,W] -> [B, N, D]
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class CIFARViT(nn.Module):
    """
    ViT backbone returning both CLS and patch tokens.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 2,
        in_chans: int = 3,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        stem_dim: int = 64,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        self.stem = ConvStem(in_chans=in_chans, stem_dim=stem_dim)
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=stem_dim,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = torch.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path_prob=dpr[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_tokens(self, x):
        # x: [B,3,H,W]
        B, _, H, W = x.shape
        x = self.stem(x)
        x = self.patch_embed(x)  # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+N, D]

        # CIFAR crops are always 32 or 16 in your pipeline; for 16, patch_embed has different N.
        # We keep a separate pos_embed per expected size by interpolating.
        x = x + self.interpolate_pos_embed(x, H, W)
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # [B, 1+N, D]

    def interpolate_pos_embed(self, x, H, W):
        # x has shape [B, 1+N, D], H/W are input image dims
        n_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if n_patches == N:
            return self.pos_embed

        # interpolate patch positional embeddings
        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]  # [1,N,D]
        dim = patch_pos.shape[-1]
        gs = int(math.sqrt(N))
        patch_pos = patch_pos.reshape(1, gs, gs, dim).permute(0, 3, 1, 2)  # [1,D,gs,gs]

        gs_new_h = H // self.patch_embed.patch_size
        gs_new_w = W // self.patch_embed.patch_size
        patch_pos = F.interpolate(
            patch_pos, size=(gs_new_h, gs_new_w), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, gs_new_h * gs_new_w, dim)
        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x, return_patches: bool = True):
        tokens = self.forward_tokens(x)
        cls = tokens[:, 0]
        if not return_patches:
            return cls
        patches = tokens[:, 1:]
        return cls, patches


# -----------------------------
# Heads
# -----------------------------
class DINOHead(nn.Module):
    """
    Projection head used for both CLS and patch tokens (applied per token).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        nlayers: int = 3,
        hidden_dim: int = 1024,
        bottleneck_dim: int = 256,
        use_bn: bool = False,
        norm_last_layer: bool = True,
    ):
        super().__init__()
        nlayers = max(1, int(nlayers))
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)

        self.apply(self._init_weights)

        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: [B,D] or [B*N,D]
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


# -----------------------------
# Multi-crop wrapper
# -----------------------------
class MultiCropWrapper(nn.Module):
    """
    Runs the backbone per resolution, concatenates outputs, then runs heads.
    Returns dict with:
      - "cls": [B*ncrops, out_dim]
      - "patch": list (len=ncrops) of patch logits or None
      - "npatches": list of int patch counts per crop
    """

    def __init__(
        self,
        backbone: nn.Module,
        head_cls: nn.Module,
        head_patch: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head_cls = head_cls
        self.head_patch = head_patch

    def forward(
        self, x: Union[torch.Tensor, List[torch.Tensor]]
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor], List[int]]]:
        if not isinstance(x, list):
            x = [x]

        # group by resolution (same as original DINO)
        sizes = torch.tensor([inp.shape[-1] for inp in x], device=x[0].device)
        idx_crops = torch.cumsum(
            torch.unique_consecutive(sizes, return_counts=True)[1], 0
        )

        start = 0
        cls_out_all: List[torch.Tensor] = []
        patch_out_all: List[Optional[torch.Tensor]] = []
        npatches: List[int] = []

        for end in idx_crops.tolist():
            batch = torch.cat(x[start:end], dim=0)  # [B*k, C, H, W]
            cls, patches = self.backbone(batch, return_patches=True)
            cls_logits = self.head_cls(cls)
            cls_out_all.append(cls_logits)

            if self.head_patch is not None:
                # apply head per patch token
                Bk, Np, D = patches.shape
                patch_logits = self.head_patch(patches.reshape(Bk * Np, D)).reshape(
                    Bk, Np, -1
                )
                patch_out_all.append(patch_logits)
                npatches.extend([Np] * (end - start))
            else:
                patch_out_all.append(None)
                npatches.extend([0] * (end - start))

            start = end

        cls_out = torch.cat(cls_out_all, dim=0)  # [B*ncrops, out_dim]

        # flatten patch list into per-crop list (so the caller can pick global crops)
        # We currently grouped by size, so patch_out_all is per-size. We need per input crop.
        # Easiest: recompute per crop boundaries using npatches and split cls_out accordingly.
        # But for training we only need:
        #  - CLS for all crops (already concatenated)
        #  - patch logits for global crops (first 2), which are at the beginning in x ordering.
        # So we return patch logits concatenated in the same order as cls_out.
        patch_logits_cat: Optional[torch.Tensor] = None
        if self.head_patch is not None:
            # Patch token count depends on crop resolution (e.g., 32->256 patches, 16->64 patches).
            # So concatenating patch logits across different crop sizes is invalid.
            patch_logits_cat = None
            if self.head_patch is not None:
                groups = [p for p in patch_out_all if p is not None]
                if len(groups) > 0:
                    nps = [g.shape[1] for g in groups]  # number of patch tokens
                    if all(n == nps[0] for n in nps):
                        patch_logits_cat = torch.cat(groups, dim=0)
                    # else: keep patch logits grouped; do NOT cat

        return {
            "cls": cls_out,
            "patch_grouped": patch_out_all,  # list per size-group in forward order
            "npatches": npatches,
            "sizes": sizes.tolist(),
        }


def cifar_vit_small(**kwargs) -> CIFARViT:
    """
    Recommended default backbone for CIFAR-10 DINO:
    - patch_size=2 (16x16 tokens)
    - embed_dim=256, depth=6, heads=8
    """
    return CIFARViT(
        img_size=32,
        patch_size=2,
        embed_dim=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        drop_path_rate=0.1,
        stem_dim=64,
        **kwargs,
    )
