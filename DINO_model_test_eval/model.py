import math, random, os, time
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

from tslearn.datasets import UCR_UEA_datasets
from sklearn.metrics import accuracy_score

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

def sinusoidal_positional_encoding(n_tokens: int, d_model: int, device: str):
    pe = torch.zeros(n_tokens, d_model, device=device)
    position = torch.arange(0, n_tokens, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class TokenGenerator(nn.Module):
    """
    Input:  (B, C, T)
    Output: (B, N, D)
    Uses a separate Linear projection per patch_len (e.g. 16 for global, 8 for local),
    so parameter shapes never change mid-training.
    """
    def __init__(self, cfg, in_channels: int):
        super().__init__()
        self.cfg = cfg
        self.in_channels = in_channels

        N = cfg.n_patches
        patch_lens = sorted({cfg.T_global // N, cfg.T_local // N, cfg.T_base // N})
        self.proj_by_plen = nn.ModuleDict()

        for plen in patch_lens:
            feat_dim = in_channels * (2 * plen + 2)
            self.proj_by_plen[str(plen)] = nn.Linear(feat_dim, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape
        N = self.cfg.n_patches
        assert T % N == 0, f"T={T} must be divisible by n_patches={N}"
        plen = T // N
        key = str(plen)
        if key not in self.proj_by_plen:
            raise ValueError(f"Unsupported patch_len={plen}. Known: {list(self.proj_by_plen.keys())}")

        # instance norm per sample/channel
        mu = x.mean(dim=-1, keepdim=True)
        sd = x.std(dim=-1, keepdim=True) + 1e-6
        x_norm = (x - mu) / sd

        # first-order diff (pad left)
        x_diff = F.pad(x[:, :, 1:] - x[:, :, :-1], (1, 0))

        # patchify
        xn = rearrange(x_norm, "b c (n p) -> b n (c p)", n=N, p=plen)
        xd = rearrange(x_diff, "b c (n p) -> b n (c p)", n=N, p=plen)

        # patch stats on raw x
        xr = rearrange(x, "b c (n p) -> b n c p", n=N, p=plen)
        pm = xr.mean(dim=-1)   # (B, N, C)
        ps = xr.std(dim=-1)    # (B, N, C)
        stats = torch.cat([pm, ps], dim=-1)  # (B, N, 2C)

        feats = torch.cat([xn, xd, stats], dim=-1)  # (B, N, feat_dim)
        return self.proj_by_plen[key](feats)

class UticaBackbone(nn.Module):
    """
    Returns:
      cls:   (B, D)
      patch: (B, N, D)
    """
    def __init__(self, cfg: UticaConfig, in_channels: int = 1):
        super().__init__()
        self.cfg = cfg
        self.tok = TokenGenerator(cfg, in_channels=in_channels)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.mlp_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, patch_mask: Optional[torch.Tensor] = None):
        # x: (B, C, T)
        B, C, T = x.shape
        tokens = self.tok(x)  # (B, N, D)
        N = tokens.shape[1]

        if patch_mask is not None:
            # patch_mask: (B, N) True=masked
            mask_tok = self.mask_token.expand(B, N, -1)
            tokens = torch.where(patch_mask.unsqueeze(-1), mask_tok, tokens)

        cls = self.cls_token.expand(B, 1, -1)
        z = torch.cat([cls, tokens], dim=1)  # (B, 1+N, D)

        pe = sinusoidal_positional_encoding(1 + N, self.cfg.d_model, z.device).unsqueeze(0)
        z = z + pe

        z = self.encoder(z)   # (B, 1+N, D)
        cls_out = z[:, 0]
        patch_out = z[:, 1:]
        return cls_out, patch_out
    
class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, bottleneck: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.proj = nn.Linear(bottleneck, out_dim, bias=False)

    def forward(self, x: torch.Tensor):
        y = self.mlp(x)
        y = F.normalize(y, dim=-1)
        return self.proj(y)
class UticaModel(nn.Module):
    def __init__(self, cfg: UticaConfig, in_channels: int = 1):
        super().__init__()
        self.cfg = cfg
        self.backbone = UticaBackbone(cfg, in_channels=in_channels)
        self.dino_head = MLPHead(cfg.d_model, cfg.head_hidden, cfg.head_bottleneck, cfg.prototypes_k)

    def forward(self, x: torch.Tensor):
        # no patch masking, no patch head
        cls, _patch = self.backbone(x, patch_mask=None)
        dino_logits = self.dino_head(cls)   # (B, K)
        return cls, dino_logits


