"""
Training loop for DINO-style self-supervised learning on CIFAR-10.

This integrates the main "DINO series" losses in a modular way:
- DINO CLS loss (student vs teacher over multi-crops)
- iBOT-style patch loss (teacher global crops supervise student global crops at patch level)
- KoLeo regularizer (encourages uniform spread of embeddings on the hypersphere)

All losses are defined as separate classes/functions and then combined in the training loop.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm

from dino_data_augmentations import DataAugmentationDINO
from dino_architecture_cifar import cifar_vit_small, DINOHead, MultiCropWrapper


# -----------------------------
# schedulers / helpers
# -----------------------------
def cosine_schedule(
    base_value: float, final_value: float, total_steps: int, warmup_steps: int = 0
) -> np.ndarray:
    """Cosine schedule with optional linear warmup."""
    assert total_steps > 0
    if warmup_steps > 0:
        warmup = np.linspace(0.0, base_value, warmup_steps)
    else:
        warmup = np.array([], dtype=np.float64)

    steps = np.arange(total_steps - warmup_steps)
    cosine = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * steps / (len(steps) - 1 + 1e-12))
    )
    return np.concatenate([warmup, cosine]).astype(np.float64)


@torch.no_grad()
def update_teacher_ema(student: nn.Module, teacher: nn.Module, m: float) -> None:
    for ps, pt in zip(student.parameters(), teacher.parameters()):
        pt.data.mul_(m).add_((1.0 - m) * ps.data)


def clip_gradients(model: nn.Module, clip: float) -> float:
    if clip <= 0:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))


# -----------------------------
# Losses (modular)
# -----------------------------
class TeacherCentering(nn.Module):
    """
    Maintains an EMA center for teacher logits (for CLS and optionally patch tokens).
    """

    def __init__(self, out_dim: int, momentum: float = 0.9):
        super().__init__()
        self.momentum = float(momentum)
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update(self, teacher_logits: torch.Tensor) -> None:
        # teacher_logits: [B*k, out_dim] or [B*k*Np, out_dim]
        batch_center = teacher_logits.mean(dim=0, keepdim=True)
        self.center.mul_(self.momentum).add_((1.0 - self.momentum) * batch_center)


class DINOCrossEntropy(nn.Module):
    """
    DINO loss on CLS logits across multi-crops.

    - teacher gets ONLY 2 global crops (so 2 chunks).
    - student gets ncrops crops (2 global + local).
    - For each teacher global view i, supervise all student views v != i.
    """

    def __init__(
        self,
        out_dim: int,
        ncrops: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        student_temp: float = 0.2,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.ncrops = int(ncrops)
        self.student_temp = float(student_temp)

        self.teacher_temp_schedule = torch.cat(
            [
                torch.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
                torch.ones(total_epochs - warmup_epochs) * teacher_temp,
            ]
        )
        self.centering = TeacherCentering(out_dim=out_dim, momentum=center_momentum)

    def forward(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor, epoch: int
    ) -> torch.Tensor:
        # student_logits: [B*ncrops, out_dim]
        # teacher_logits: [B*2, out_dim]
        student_out = (student_logits / self.student_temp).chunk(self.ncrops)

        ttemp = float(self.teacher_temp_schedule[epoch].item())
        teacher_probs = (
            F.softmax((teacher_logits - self.centering.center) / ttemp, dim=-1)
            .detach()
            .chunk(2)
        )

        total_loss = 0.0
        n_terms = 0
        for i, q in enumerate(teacher_probs):
            for v, s in enumerate(student_out):
                if v == i:
                    continue
                total_loss = (
                    total_loss + torch.sum(-q * F.log_softmax(s, dim=-1), dim=-1).mean()
                )
                n_terms += 1

        total_loss = total_loss / max(1, n_terms)

        with torch.no_grad():
            self.centering.update(teacher_logits)

        return total_loss


class IBOTPatchLoss(nn.Module):
    """
    iBOT-style patch loss on global crops only (same resolution -> same #patches).

    We compute a DINO-like cross entropy over patch logits:
      teacher global view i supervises student global view j (j != i).
    """

    def __init__(
        self,
        out_dim: int,
        warmup_teacher_temp: float = 0.04,
        teacher_temp: float = 0.07,
        warmup_epochs: int = 10,
        total_epochs: int = 100,
        student_temp: float = 0.2,
        center_momentum: float = 0.9,
    ):
        super().__init__()
        self.student_temp = float(student_temp)
        self.teacher_temp_schedule = torch.cat(
            [
                torch.linspace(warmup_teacher_temp, teacher_temp, warmup_epochs),
                torch.ones(total_epochs - warmup_epochs) * teacher_temp,
            ]
        )
        self.centering = TeacherCentering(out_dim=out_dim, momentum=center_momentum)

    def forward(
        self,
        student_patch_logits: torch.Tensor,
        teacher_patch_logits: torch.Tensor,
        epoch: int,
    ) -> torch.Tensor:
        """
        student_patch_logits: [B*2, Np, out_dim] (global crops only)
        teacher_patch_logits: [B*2, Np, out_dim] (global crops only)
        """
        B2, Np, D = student_patch_logits.shape
        assert teacher_patch_logits.shape[:2] == (B2, Np)

        student = (student_patch_logits / self.student_temp).reshape(B2, Np, D)
        ttemp = float(self.teacher_temp_schedule[epoch].item())

        teacher_probs = F.softmax(
            (teacher_patch_logits - self.centering.center) / ttemp, dim=-1
        ).detach()

        # split by the two global views (DataLoader gives crop-index tensors, so we concat as [all g1; all g2])
        assert B2 % 2 == 0, (
            "Expected exactly 2 global crops concatenated on batch dimension."
        )
        B = B2 // 2
        s0, s1 = student[:B], student[B:]
        t0, t1 = teacher_probs[:B], teacher_probs[B:]
        loss01 = torch.sum(-t0 * F.log_softmax(s1, dim=-1), dim=-1).mean()
        loss10 = torch.sum(-t1 * F.log_softmax(s0, dim=-1), dim=-1).mean()
        loss = 0.5 * (loss01 + loss10)

        with torch.no_grad():
            self.centering.update(teacher_patch_logits.reshape(B2 * Np, D))

        return loss


class KoLeoLoss(nn.Module):
    """
    KoLeo (Kozachenko–Leonenko) entropy regularizer on normalized embeddings.

    Practical approximation used in DINOv2: encourage large nearest-neighbor distances
    on the unit sphere.

    Input: embeddings [B, d] (assumed not yet normalized).
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=-1, p=2)
        # pairwise cosine distance on unit sphere: dist^2 = 2 - 2*cos
        sim = x @ x.T  # [B,B]
        # exclude self
        sim.fill_diagonal_(-1.0)
        # nearest neighbor is max similarity
        nn_sim, _ = sim.max(dim=1)
        nn_dist = torch.sqrt(torch.clamp(2.0 - 2.0 * nn_sim, min=self.eps))
        return (-torch.log(nn_dist + self.eps)).mean()


# -----------------------------
# Model builder
# -----------------------------
@dataclass
class ModelBundle:
    student: MultiCropWrapper
    teacher: MultiCropWrapper


def build_student_teacher(out_dim: int, norm_last_layer: bool = True) -> ModelBundle:
    backbone_s = cifar_vit_small()
    backbone_t = cifar_vit_small()

    # CLS head
    head_cls_s = DINOHead(
        in_dim=backbone_s.embed_dim,
        out_dim=out_dim,
        nlayers=3,
        hidden_dim=1024,
        bottleneck_dim=256,
        use_bn=False,
        norm_last_layer=norm_last_layer,
    )
    head_cls_t = DINOHead(
        in_dim=backbone_t.embed_dim,
        out_dim=out_dim,
        nlayers=3,
        hidden_dim=1024,
        bottleneck_dim=256,
        use_bn=False,
        norm_last_layer=True,
    )

    # Patch head (same structure, applied per patch token)
    head_patch_s = DINOHead(
        in_dim=backbone_s.embed_dim,
        out_dim=out_dim,
        nlayers=2,
        hidden_dim=1024,
        bottleneck_dim=256,
        use_bn=False,
        norm_last_layer=norm_last_layer,
    )
    head_patch_t = DINOHead(
        in_dim=backbone_t.embed_dim,
        out_dim=out_dim,
        nlayers=2,
        hidden_dim=1024,
        bottleneck_dim=256,
        use_bn=False,
        norm_last_layer=True,
    )

    student = MultiCropWrapper(backbone_s, head_cls_s, head_patch_s)
    teacher = MultiCropWrapper(backbone_t, head_cls_t, head_patch_t)

    # init teacher
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    return ModelBundle(student=student, teacher=teacher)


# -----------------------------
# Training
# -----------------------------
def compute_total_loss(
    losses: Dict[str, nn.Module],
    student_out: Dict,
    teacher_out: Dict,
    images: List[torch.Tensor],
    epoch: int,
    w_cls: float,
    w_patch: float,
    w_koleo: float,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Combines all losses with weights and returns (total, log_dict).
    """
    log: Dict[str, float] = {}

    # CLS logits
    loss_cls = losses["dino_cls"](student_out["cls"], teacher_out["cls"], epoch)
    log["loss_dino_cls"] = float(loss_cls.item())

    # Patch logits for global crops only: patch_grouped[0] is the 32x32 group (2 global crops)
    loss_patch = torch.tensor(0.0, device=loss_cls.device)
    if w_patch > 0:
        sp = student_out["patch_grouped"][0]
        tp = teacher_out["patch_grouped"][0]
        if sp is None or tp is None:
            raise RuntimeError("Patch heads are disabled but w_patch > 0.")
        loss_patch = losses["ibot_patch"](sp, tp, epoch)
        log["loss_ibot_patch"] = float(loss_patch.item())
    else:
        log["loss_ibot_patch"] = 0.0

    # KoLeo on student global CLS *embeddings* (pre-head)
    loss_koleo = torch.tensor(0.0, device=loss_cls.device)
    if w_koleo > 0:
        with torch.no_grad():
            # keep it cheap: only use first global crop embeddings
            g1 = images[0]
        cls_emb, _ = losses["_student_backbone"](g1, return_patches=True)  # type: ignore
        loss_koleo = losses["koleo"](cls_emb)
        log["loss_koleo"] = float(loss_koleo.item())
    else:
        log["loss_koleo"] = 0.0

    total = w_cls * loss_cls + w_patch * loss_patch + w_koleo * loss_koleo
    log["loss_total"] = float(total.item())
    return total, log


def train_one_epoch(
    student: MultiCropWrapper,
    teacher: MultiCropWrapper,
    optimizer: torch.optim.Optimizer,
    data_loader: DataLoader,
    losses: Dict[str, nn.Module],
    lr_schedule: np.ndarray,
    wd_schedule: np.ndarray,
    m_schedule: np.ndarray,
    epoch: int,
    device: torch.device,
    clip_grad: float,
    w_cls: float,
    w_patch: float,
    w_koleo: float,
) -> Dict[str, float]:
    student.train()
    teacher.eval()

    metrics: Dict[str, float] = {
        "loss_total": 0.0,
        "loss_dino_cls": 0.0,
        "loss_ibot_patch": 0.0,
        "loss_koleo": 0.0,
        "lr": 0.0,
        "wd": 0.0,
        "ema_m": 0.0,
        "grad_norm": 0.0,
    }

    n = 0
    pbar = tqdm(data_loader, desc=f"Epoch {epoch + 1}", leave=True)
    for it, (crops, _) in enumerate(pbar):
        step = epoch * len(data_loader) + it

        # schedules
        lr = float(lr_schedule[step])
        wd = float(wd_schedule[step])
        m = float(m_schedule[step])

        for pg in optimizer.param_groups:
            pg["lr"] = lr
            pg["weight_decay"] = wd

        images = [im.to(device, non_blocking=True) for im in crops]

        student_out = student(images)
        with torch.no_grad():
            teacher_out = teacher(images[:2])  # only global views for teacher

        total_loss, log = compute_total_loss(
            losses=losses,
            student_out=student_out,
            teacher_out=teacher_out,
            images=images,
            epoch=epoch,
            w_cls=w_cls,
            w_patch=w_patch,
            w_koleo=w_koleo,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = clip_gradients(student, clip_grad)
        optimizer.step()

        # EMA update
        with torch.no_grad():
            update_teacher_ema(student, teacher, m)

        # accumulate
        n += 1
        for k in ["loss_total", "loss_dino_cls", "loss_ibot_patch", "loss_koleo"]:
            metrics[k] += log.get(k, 0.0)
        metrics["lr"] += lr
        metrics["wd"] += wd
        metrics["ema_m"] += m
        metrics["grad_norm"] += grad_norm

        # show
        pbar.set_postfix(
            loss=f"{log['loss_total']:.4f}",
            cls=f"{log['loss_dino_cls']:.4f}",
            patch=f"{log['loss_ibot_patch']:.4f}",
            lr=f"{lr:.2e}",
            m=f"{m:.4f}",
        )

    for k in metrics:
        metrics[k] /= max(1, n)
    return metrics


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- data ---
    transform = DataAugmentationDINO(
        global_crops_scale=(0.25, 1.0),
        local_crops_scale=(0.05, 0.25),
        local_crops_number=args.local_crops_number,
    )
    ds = datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True,
    )

    # --- models ---
    models = build_student_teacher(
        out_dim=args.out_dim, norm_last_layer=args.norm_last_layer
    )
    student, teacher = models.student.to(device), models.teacher.to(device)

    # --- losses (need total_epochs=args.epochs so if you extend epochs, schedule matches new total) ---
    dino_cls = DINOCrossEntropy(
        out_dim=args.out_dim,
        ncrops=2 + args.local_crops_number,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    ).to(device)

    ibot_patch = IBOTPatchLoss(
        out_dim=args.out_dim,
        warmup_teacher_temp=args.warmup_teacher_temp,
        teacher_temp=args.teacher_temp,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        student_temp=args.student_temp,
        center_momentum=args.center_momentum,
    ).to(device)

    koleo = KoLeoLoss().to(device)

    # --- optimizer ---
    optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- resume (optional) ---
    start_epoch = 0
    history: List[Dict[str, float]] = []

    resume_path = getattr(args, "resume", "") or ""
    if resume_path:
        ckpt = torch.load(resume_path, map_location="cpu")
        student.load_state_dict(ckpt["student"])
        teacher.load_state_dict(ckpt["teacher"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        history = ckpt.get("metrics", []) or []
        print(f"Resumed from {resume_path} at epoch {start_epoch}")

    # --- schedules (per step) ---
    steps_per_epoch = len(dl)
    total_steps = args.epochs * steps_per_epoch

    lr_schedule = cosine_schedule(
        args.lr,
        args.min_lr,
        total_steps,
        warmup_steps=args.lr_warmup_epochs * steps_per_epoch,
    )
    wd_schedule = cosine_schedule(args.weight_decay, args.weight_decay_end, total_steps)
    m_schedule = cosine_schedule(args.ema_m, 1.0, total_steps)

    # If resuming, advance schedules so step=0 corresponds to the first *new* step after resume
    start_step = start_epoch * steps_per_epoch
    if start_step > 0:
        if start_step >= len(lr_schedule):
            raise ValueError(
                f"Resume epoch {start_epoch} implies start_step={start_step}, "
                f"but total_steps={len(lr_schedule)}. Increase --epochs or resume from an earlier checkpoint."
            )
        lr_schedule = lr_schedule[start_step:]
        wd_schedule = wd_schedule[start_step:]
        m_schedule = m_schedule[start_step:]

    # --- pack losses dict (also pass student backbone for KoLeo) ---
    losses = {
        "dino_cls": dino_cls,
        "ibot_patch": ibot_patch,
        "koleo": koleo,
        "_student_backbone": student.backbone,
    }

    os.makedirs(args.output_dir, exist_ok=True)

    # --- training loop ---
    for epoch in range(start_epoch, args.epochs):
        metrics = train_one_epoch(
            student=student,
            teacher=teacher,
            optimizer=optimizer,
            data_loader=dl,
            losses=losses,
            lr_schedule=lr_schedule,
            wd_schedule=wd_schedule,
            m_schedule=m_schedule,
            epoch=epoch,
            device=device,
            clip_grad=args.clip_grad,
            w_cls=args.w_cls,
            w_patch=args.w_patch,
            w_koleo=args.w_koleo,
        )
        history.append(metrics)

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"loss={metrics['loss_total']:.4f} "
            f"(cls={metrics['loss_dino_cls']:.4f}, patch={metrics['loss_ibot_patch']:.4f}, koleo={metrics['loss_koleo']:.4f}) "
            f"lr={metrics['lr']:.2e} m={metrics['ema_m']:.4f}"
        )

        # Save checkpoints
        if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
            ckpt = {
                "epoch": epoch,
                "student": student.state_dict(),
                "teacher": teacher.state_dict(),
                "optimizer": optimizer.state_dict(),
                "metrics": history,
                "args": vars(args),
            }
            path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pth")
            torch.save(ckpt, path)
            print(f"Saved: {path}")

    # final save (include optimizer too; useful if you want to resume from "final")
    torch.save(
        {
            "epoch": args.epochs - 1,
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": history,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "dino_cifar_final.pth"),
    )
    print("Done.")


# def train(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.manual_seed(args.seed)
#     np.random.seed(args.seed)
#
#     transform = DataAugmentationDINO(
#         global_crops_scale=(0.25, 1.0),
#         local_crops_scale=(0.05, 0.25),
#         local_crops_number=args.local_crops_number,
#     )
#
#     ds = datasets.CIFAR10(
#         root=args.data_dir, train=True, download=True, transform=transform
#     )
#     dl = DataLoader(
#         ds,
#         batch_size=args.batch_size,
#         shuffle=True,
#         num_workers=args.num_workers,
#         drop_last=True,
#         pin_memory=True,
#     )
#
#     models = build_student_teacher(
#         out_dim=args.out_dim, norm_last_layer=args.norm_last_layer
#     )
#     student, teacher = models.student.to(device), models.teacher.to(device)
#
#     # losses
#     dino_cls = DINOCrossEntropy(
#         out_dim=args.out_dim,
#         ncrops=2 + args.local_crops_number,
#         warmup_teacher_temp=args.warmup_teacher_temp,
#         teacher_temp=args.teacher_temp,
#         warmup_epochs=args.warmup_epochs,
#         total_epochs=args.epochs,
#         student_temp=args.student_temp,
#         center_momentum=args.center_momentum,
#     ).to(device)
#
#     ibot_patch = IBOTPatchLoss(
#         out_dim=args.out_dim,
#         warmup_teacher_temp=args.warmup_teacher_temp,
#         teacher_temp=args.teacher_temp,
#         warmup_epochs=args.warmup_epochs,
#         total_epochs=args.epochs,
#         student_temp=args.student_temp,
#         center_momentum=args.center_momentum,
#     ).to(device)
#
#     koleo = KoLeoLoss().to(device)
#
#     # optimizer
#     optimizer = AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
#     # schedules (per step)
#     steps_per_epoch = len(dl)
#     total_steps = args.epochs * steps_per_epoch
#
#     lr_schedule = cosine_schedule(
#         args.lr,
#         args.min_lr,
#         total_steps,
#         warmup_steps=args.lr_warmup_epochs * steps_per_epoch,
#     )
#     wd_schedule = cosine_schedule(args.weight_decay, args.weight_decay_end, total_steps)
#     m_schedule = cosine_schedule(args.ema_m, 1.0, total_steps)
#
#     # pack losses dict (also pass student backbone for KoLeo)
#     losses = {
#         "dino_cls": dino_cls,
#         "ibot_patch": ibot_patch,
#         "koleo": koleo,
#         "_student_backbone": student.backbone,
#     }
#
#     os.makedirs(args.output_dir, exist_ok=True)
#
#     history: List[Dict[str, float]] = []
#     for epoch in range(args.epochs):
#         metrics = train_one_epoch(
#             student=student,
#             teacher=teacher,
#             optimizer=optimizer,
#             data_loader=dl,
#             losses=losses,
#             lr_schedule=lr_schedule,
#             wd_schedule=wd_schedule,
#             m_schedule=m_schedule,
#             epoch=epoch,
#             device=device,
#             clip_grad=args.clip_grad,
#             w_cls=args.w_cls,
#             w_patch=args.w_patch,
#             w_koleo=args.w_koleo,
#         )
#         history.append(metrics)
#
#         print(
#             f"Epoch {epoch + 1:03d}/{args.epochs} | "
#             f"loss={metrics['loss_total']:.4f} "
#             f"(cls={metrics['loss_dino_cls']:.4f}, patch={metrics['loss_ibot_patch']:.4f}, koleo={metrics['loss_koleo']:.4f}) "
#             f"lr={metrics['lr']:.2e} m={metrics['ema_m']:.4f}"
#         )
#
#         if (epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs:
#             ckpt = {
#                 "epoch": epoch,
#                 "student": student.state_dict(),
#                 "teacher": teacher.state_dict(),
#                 "optimizer": optimizer.state_dict(),
#                 "metrics": history,
#                 "args": vars(args),
#             }
#             path = os.path.join(args.output_dir, f"checkpoint_epoch{epoch + 1}.pth")
#             torch.save(ckpt, path)
#             print(f"Saved: {path}")
#
#     # final save
#     torch.save(
#         {
#             "student": student.state_dict(),
#             "teacher": teacher.state_dict(),
#             "metrics": history,
#             "args": vars(args),
#         },
#         os.path.join(args.output_dir, "dino_cifar_final.pth"),
#     )
#     print("Done.")
#


def build_argparser():
    p = argparse.ArgumentParser("DINO (CIFAR-10) with CLS + iBOT patch + KoLeo")
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--output_dir", type=str, default="./dino_checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)

    # model / loss
    p.add_argument("--out_dim", type=int, default=4096)
    p.add_argument("--local_crops_number", type=int, default=6)

    p.add_argument("--warmup_teacher_temp", type=float, default=0.04)
    p.add_argument("--teacher_temp", type=float, default=0.07)
    p.add_argument("--student_temp", type=float, default=0.2)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--center_momentum", type=float, default=0.9)
    p.add_argument("--norm_last_layer", action="store_true")
    p.add_argument(
        "--resume", type=str, default="", help="path to checkpoint .pth to resume"
    )

    # optimization
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lr_warmup_epochs", type=int, default=10)
    p.add_argument("--weight_decay", type=float, default=0.04)
    p.add_argument("--weight_decay_end", type=float, default=0.4)
    p.add_argument("--clip_grad", type=float, default=3.0)

    # EMA
    p.add_argument("--ema_m", type=float, default=0.996)

    # loss weights
    p.add_argument("--w_cls", type=float, default=1.0)
    p.add_argument("--w_patch", type=float, default=1.0)
    p.add_argument("--w_koleo", type=float, default=0.1)

    p.add_argument("--save_freq", type=int, default=20)
    return p


def main():
    args = build_argparser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
