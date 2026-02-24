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

def resample_to_T(x_np: np.ndarray, T: int) -> np.ndarray:
    """
    Retourne toujours un array (B, C, T).
    Gère:
      - (B, T)
      - (B, T, C)  (tslearn fréquent)
      - (B, C, T)
    """
    if x_np.ndim == 2:
        x = torch.from_numpy(x_np).float().unsqueeze(1)   # (B,1,T)
    elif x_np.ndim == 3:
        # Heuristique: si dernière dim petite => souvent (B,T,C)
        if x_np.shape[2] <= 16 and x_np.shape[1] > x_np.shape[2]:
            x = torch.from_numpy(x_np).float().permute(0, 2, 1)  # (B,C,T)
        else:
            x = torch.from_numpy(x_np).float()  # déjà (B,C,T) probable
    else:
        raise ValueError(f"UCR shape inattendue: {x_np.shape}")

    # interpolate attend (N,C,L) => ici (B,C,T)
    x = F.interpolate(x, size=T, mode="linear", align_corners=False)
    return x.numpy()  # (B,C,T)

def encode(model: UticaModel, X: torch.Tensor, batch_size=256):
    model.eval()
    feats = []
    dl = torch.utils.data.DataLoader(X, batch_size=batch_size, shuffle=False)
    for xb in dl:
        xb = xb.to(device)
        cls, _ = model(xb)     
        feats.append(cls.detach().cpu())
    return torch.cat(feats, dim=0)

class LinearProbe(nn.Module):
    def __init__(self, d_in: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(d_in, n_classes)

    def forward(self, x): return self.fc(x)

def train_linear_probe(feats_train, y_train, feats_test, y_test, lr=1e-3, epochs=50):
    n_classes = int(y_train.max().item() + 1)
    clf = LinearProbe(feats_train.shape[1], n_classes).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        clf.train()
        idx = torch.randperm(feats_train.shape[0])
        xb = feats_train[idx].to(device)
        yb = y_train[idx].to(device)
        logits = clf(xb)
        loss = loss_fn(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        pred = clf(feats_test.to(device)).argmax(dim=-1).cpu().numpy()
    acc = accuracy_score(y_test.numpy(), pred)
    return acc
