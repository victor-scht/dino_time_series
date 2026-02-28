#!/usr/bin/env python3
"""
visualization.py

Load a DINO CIFAR checkpoint, extract patch tokens for one CIFAR-10 image,
apply PCA -> 3D, map to RGB over the patch grid, upsample to 32x32, and display/save.

Usage examples:
  python3 visualization.py --ckpt ./dino_checkpoints/checkpoint_epoch5.pth
  python3 visualization.py --ckpt ./dino_checkpoints/checkpoint_epoch5.pth --index 123 --save pca_rgb.png
  python3 visualization.py --ckpt ./dino_checkpoints/checkpoint_epoch5.pth --data_dir ./data --device cpu
"""

import argparse
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Your project code
from training_loop_dino_cifar import build_student_teacher


def unwrap_tokens(out: Any) -> Optional[torch.Tensor]:
    """
    Find a likely token tensor inside nested tuples/lists/dicts.
    Prefer a 3D tensor shaped [B, T, D].
    """
    if torch.is_tensor(out):
        return out

    if isinstance(out, dict):
        # Try common keys first
        for k in ["tokens", "x", "feat", "features"]:
            v = out.get(k, None)
            if torch.is_tensor(v):
                return v
        # Otherwise recurse / pick first tensor
        for v in out.values():
            t = unwrap_tokens(v)
            if t is not None:
                return t
        return None

    if isinstance(out, (tuple, list)):
        # Prefer 3D tensor
        for v in out:
            if torch.is_tensor(v) and v.ndim == 3:
                return v
        # Otherwise recurse
        for v in out:
            t = unwrap_tokens(v)
            if t is not None:
                return t
        return None

    return None


@torch.no_grad()
def get_tokens(student: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Return token tensor from the student backbone.
    Output is expected to be either:
      - [B, N, D] patch-only tokens (common here)
      - [B, 1+N, D] tokens including CLS
    """
    bb = student.backbone
    if hasattr(bb, "forward_features"):
        out = bb.forward_features(x)
    else:
        out = bb(x)

    t = unwrap_tokens(out)
    if t is None or not torch.is_tensor(t):
        raise RuntimeError(
            f"Could not find token tensor in backbone output. Got type={type(out)}"
        )
    if t.ndim != 3:
        raise RuntimeError(
            f"Token tensor must be 3D [B,T,D], got shape={tuple(t.shape)}"
        )
    return t


def tokens_to_patch_grid(
    tokens: torch.Tensor, expected_grid: Optional[int] = None
) -> torch.Tensor:
    """
    Convert tokens to patch tokens [B, N, D] and validate N is a square.
    Handles both patch-only and CLS+patch formats.
    """
    B, T, D = tokens.shape

    # CIFAR with patch_size=2 typically yields 16x16 = 256 patches.
    # Your backbone returns patch-only [B,256,D], but handle CLS+patch as well.
    if expected_grid is not None:
        expected_patches = expected_grid * expected_grid
        if T == expected_patches:
            patch_tokens = tokens
        elif T == expected_patches + 1:
            patch_tokens = tokens[:, 1:, :]
        else:
            # fallback heuristics
            patch_tokens = (
                tokens[:, 1:, :] if int(np.sqrt(T - 1)) ** 2 == (T - 1) else tokens
            )
    else:
        # Generic handling
        if int(np.sqrt(T)) ** 2 == T:
            patch_tokens = tokens
        elif int(np.sqrt(T - 1)) ** 2 == (T - 1):
            patch_tokens = tokens[:, 1:, :]
        else:
            raise AssertionError(
                f"Neither T={T} nor T-1={T - 1} is a perfect square; cannot form a patch grid."
            )

    _, N, _ = patch_tokens.shape
    g = int(np.sqrt(N))
    if g * g != N:
        raise AssertionError(f"N={N} is not a perfect square; cannot reshape to grid.")
    return patch_tokens


def pca_rgb_from_patch_tokens(patch_tokens: torch.Tensor, grid_size: int) -> np.ndarray:
    """
    patch_tokens: [N, D] on CPU (numpy or tensor)
    returns rgb_grid: [grid_size, grid_size, 3] in [0,1]
    """
    if torch.is_tensor(patch_tokens):
        pt = patch_tokens.detach().cpu().numpy()
    else:
        pt = patch_tokens

    pca = PCA(n_components=3)
    rgb = pca.fit_transform(pt)  # [N, 3]

    # Normalize each channel to [0,1]
    rgb = rgb - rgb.min(axis=0, keepdims=True)
    rgb = rgb / (rgb.max(axis=0, keepdims=True) + 1e-8)

    return rgb.reshape(grid_size, grid_size, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    ap.add_argument(
        "--data_dir", default="./data", help="Where CIFAR-10 is/will be downloaded"
    )
    ap.add_argument(
        "--index", type=int, default=0, help="CIFAR-10 test index to visualize"
    )
    ap.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device"
    )
    ap.add_argument(
        "--out_dim", type=int, default=4096, help="Must match training out_dim"
    )
    ap.add_argument(
        "--norm_last_layer", action="store_true", help="Must match training setting"
    )
    ap.add_argument(
        "--expected_grid",
        type=int,
        default=16,
        help="Expected patch grid size (16 for CIFAR patch_size=2)",
    )
    ap.add_argument(
        "--save", default="", help="Optional path to save the PCA RGB image (PNG)"
    )
    args = ap.parse_args()

    # Choose device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print("Device:", device)

    # Build student
    models = build_student_teacher(
        out_dim=args.out_dim, norm_last_layer=args.norm_last_layer
    )
    student = models.student.to(device)

    # Load checkpoint
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    if "student" not in ckpt:
        raise KeyError(f"Checkpoint keys: {list(ckpt.keys())}. Expected 'student'.")
    student.load_state_dict(ckpt["student"], strict=True)
    student.eval()
    print("Loaded checkpoint:", args.ckpt)

    # Load one CIFAR-10 image
    os.makedirs(args.data_dir, exist_ok=True)
    tfm = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=tfm)

    idx = int(args.index) % len(ds)
    img, label = ds[idx]
    x = img.unsqueeze(0).to(device)  # [1,3,32,32]
    print(f"Sample index: {idx} | label: {label} | x: {tuple(x.shape)}")

    # Extract tokens and build patch grid
    tokens = get_tokens(student, x)
    print("tokens shape:", tuple(tokens.shape))  # [1, T, D]

    patch_tokens = tokens_to_patch_grid(tokens, expected_grid=args.expected_grid)
    B, N, D = patch_tokens.shape
    g = int(np.sqrt(N))
    print(f"patch_tokens: {tuple(patch_tokens.shape)} | grid: {g}x{g}")

    # PCA -> RGB grid
    rgb_grid = pca_rgb_from_patch_tokens(patch_tokens[0], grid_size=g)  # [g,g,3]

    # Upsample to 32x32 for nicer visualization
    rgb_t = (
        torch.from_numpy(rgb_grid).permute(2, 0, 1).unsqueeze(0).float()
    )  # [1,3,g,g]
    rgb_up = (
        F.interpolate(rgb_t, size=(32, 32), mode="bilinear", align_corners=False)[0]
        .permute(1, 2, 0)
        .numpy()
    )

    # Show
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img.permute(1, 2, 0))
    plt.title("Input")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_up)
    plt.title("Patch tokens PCA → RGB")
    plt.axis("off")

    plt.tight_layout()

    if args.save:
        plt.savefig(args.save, dpi=200, bbox_inches="tight")
        print("Saved image:", args.save)

    plt.show()


if __name__ == "__main__":
    main()
