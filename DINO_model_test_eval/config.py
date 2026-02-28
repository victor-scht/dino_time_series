import math, random, os, time
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class UticaConfig:
    # time / patches
    T_base: int = 512
    T_global: int = 512
    T_local: int = 256
    n_patches: int = 32  # paper uses 32 patches :contentReference[oaicite:5]{index=5}

    # model (paper backbone: L=6, D=256, H=8, MLP=512) :contentReference[oaicite:6]{index=6}
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    mlp_dim: int = 512
    dropout: float = 0.0

    # heads (paper: 3-layer MLP, hidden 2048, bottleneck 256) :contentReference[oaicite:7]{index=7}
    head_hidden: int = 2048
    head_bottleneck: int = 256
    prototypes_k: int = 1024  # set 65536 to match paper, but start smaller on Colab :contentReference[oaicite:8]{index=8}

    # crops (paper: global removal [0,0.6], local removal [0.6,0.9]) :contentReference[oaicite:9]{index=9}
    n_global: int = 2
    n_local: int = 8
    global_remove: Tuple[float, float] = (0.0, 0.6)
    local_remove: Tuple[float, float] = (0.6, 0.9)
    jitter_sigma: float = 0.2
    jitter_p: float = 0.5

    # masking (paper: mask ratio U(0.1,0.7), p=0.5) :contentReference[oaicite:10]{index=10}
    mask_ratio: Tuple[float, float] = (0.1, 0.7)

    # teacher schedules (paper: momentum cosine 0.992->1.0, temp 0.04->0.07 warmup 1.5 epochs) :contentReference[oaicite:12]{index=12}
    m_start: float = 0.992
    m_end: float = 1.0
    t_start: float = 0.04
    t_end: float = 0.04
    temp_warmup_epochs: float = 1.5

    # optimization defaults (paper) :contentReference[oaicite:13]{index=13}
    lr: float = 5e-4
    min_lr: float = 5e-7
    wd_start: float = 0.04
    wd_end: float = 0.4
    clip_grad: float = 3.0
    warmup_epochs: float = 0.5

    # DINO temperatures
    student_temp: float = 0.2  # typical DINO value
    teacher_temp_start: float = 0.04  # start of schedule
    teacher_temp_end: float = 0.07  # end of schedule
    teacher_temp_warmup_epochs: float = 1.5


if __name__ == "__main__":
    cfg = UticaConfig()
