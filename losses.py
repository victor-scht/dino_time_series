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

class DinoCenter:
    def __init__(self, K, momentum=0.9, device="cuda"):
        self.center = torch.zeros(1, K, device=device)
        self.m = momentum

    def update(self, teacher_logits):
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center = self.center * self.m + batch_center * (1 - self.m)

    def probs(self, teacher_logits, temp):
        return F.softmax((teacher_logits - self.center) / max(temp, 1e-6), dim=-1)

def teacher_temp(step: int, steps_per_epoch: int, epoch: float, cfg: UticaConfig):
    # linear warmup then constant (or you can keep warming to end)
    if epoch < cfg.teacher_temp_warmup_epochs:
        frac = epoch / max(cfg.teacher_temp_warmup_epochs, 1e-6)
        return cfg.teacher_temp_start + frac * (cfg.teacher_temp_end - cfg.teacher_temp_start)
    return cfg.teacher_temp_end

def dino_loss_multicrop_2global(student_g_logits, student_l_logits, teacher_g_probs, n_local: int, student_temp: float):
    B2, K = student_g_logits.shape
    B = B2 // 2

    sg = student_g_logits.view(2, B, K)
    tg = teacher_g_probs.view(2, B, K)
    sl = student_l_logits.view(n_local, B, K)

    log_sg = F.log_softmax(sg / student_temp, dim=-1)
    log_sl = F.log_softmax(sl / student_temp, dim=-1)

    p0 = tg[0]
    loss0 = (-(p0 * log_sg[1]).sum(-1).mean() + n_local * (-(p0[None] * log_sl).sum(-1).mean())) / (1 + n_local)

    p1 = tg[1]
    loss1 = (-(p1 * log_sg[0]).sum(-1).mean() + n_local * (-(p1[None] * log_sl).sum(-1).mean())) / (1 + n_local)

    return 0.5 * (loss0 + loss1)