def update_ema(teacher: nn.Module, student: nn.Module, m: float):
    for p_t, p_s in zip(teacher.parameters(), student.parameters()):
        p_t.data.mul_(m).add_(p_s.data, alpha=(1.0 - m))


def cosine_schedule(start: float, end: float, step: int, total_steps: int):
    if total_steps <= 1:
        return end
    t = step / (total_steps - 1)
    return end - (end - start) * (0.5 * (1 + math.cos(math.pi * t)))


def make_optimizer(student: UticaModel, cfg: UticaConfig):
    return torch.optim.AdamW(
        student.parameters(), lr=cfg.lr, betas=(0.9, 0.999), weight_decay=cfg.wd_start
    )


def adjust_lr_wd(optim, step, total_steps, cfg: UticaConfig):
    lr = cosine_schedule(cfg.lr, cfg.min_lr, step, total_steps)
    wd = cosine_schedule(cfg.wd_start, cfg.wd_end, step, total_steps)
    for pg in optim.param_groups:
        pg["lr"] = lr
        pg["weight_decay"] = wd
    return lr, wd


def pretrain_utica(
    cfg: UticaConfig,
    n_steps: int = 1000,
    batch_size: int = 64,
    n_samples: int = 50_000,
    in_channels: int = 1,
    seed: int = 0,
):
    set_seed(seed)
    ds = SyntheticDAGDataset(
        n_samples=n_samples, T=cfg.T_base, n_obs=in_channels, seed=seed
    )

    # IMPORTANT: num_workers=0 helps avoid Colab RAM crashes
    dl = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    student = UticaModel(cfg, in_channels=in_channels).to(device)
    teacher = UticaModel(cfg, in_channels=in_channels).to(device)
    teacher.load_state_dict(student.state_dict())
    for p in teacher.parameters():
        p.requires_grad = False

    optim = make_optimizer(student, cfg)

    # removes FutureWarning vs torch.cuda.amp.GradScaler
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    steps_per_epoch = max(1, len(dl))
    total_steps = n_steps

    it = iter(dl)
    losses = []
    teacher_entropies, teacher_top1s = [], []
    student_entropies, student_top1s = [], []

    student.train()
    teacher.eval()

    for step in tqdm(range(n_steps)):
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl)
            x = next(it)

        x = x.to(device, dtype=torch.float32)  # (B, C, T_base)
        B = x.shape[0]

        # create views
        global_views, local_views = make_multicrops(x, cfg)
        xg = torch.cat(global_views, dim=0)  # (2B, C, 512)
        xl = torch.cat(local_views, dim=0)  # (n_local*B, C, 256)

        # schedules
        lr, wd = adjust_lr_wd(optim, step, total_steps, cfg)
        epoch_float = step / steps_per_epoch
        m = cosine_schedule(cfg.m_start, cfg.m_end, step, total_steps)
        t = teacher_temp(step, steps_per_epoch, epoch_float, cfg)

        center = DinoCenter(cfg.prototypes_k, momentum=0.9, device=device)
        with torch.no_grad():
            _, t_logits = teacher(xg)
            t_dino_probs = center.probs(t_logits, temp=t)
            center.update(t_logits)

        with torch.no_grad():
            ent = -(t_dino_probs * (t_dino_probs + 1e-6).log()).sum(dim=1).mean().item()
            top1 = t_dino_probs.max(dim=1).values.mean().item()
            usage = (
                (t_dino_probs.mean(dim=0) > (1.0 / cfg.prototypes_k) * 2)
                .float()
                .mean()
                .item()
            )
            # print(f"teacher entropy={ent:.3f}  top1={top1:.4f}  proto_usage~={usage:.3f}")
            teacher_entropies.append(ent)
            teacher_top1s.append(top1)

        with torch.amp.autocast("cuda", enabled=(device == "cuda")):
            # student globals
            _, s_dino_g = student(xg)  # (2B, K)

            # student locals in chunks (memory-safe)
            s_dino_l_chunks = []
            chunk = 2 * B  # 2B at a time; reduce to B if needed
            for xl_chunk in torch.split(xl, chunk, dim=0):
                _, s_dino_chunk = student(xl_chunk)
                s_dino_l_chunks.append(s_dino_chunk)
            s_dino_l = torch.cat(s_dino_l_chunks, dim=0)  # (n_local*B, K)
            s_probs = F.softmax(s_dino_g / cfg.student_temp, dim=-1)
            s_ent = -(s_probs * (s_probs + 1e-6).log()).sum(dim=1).mean().item()
            s_top1 = s_probs.max(dim=1).values.mean().item()
            student_entropies.append(s_ent)
            student_top1s.append(s_top1)

            # DINO-only loss
            loss = dino_loss_multicrop_2global(
                s_dino_g,
                s_dino_l,
                t_dino_probs,
                cfg.n_local,
                student_temp=cfg.student_temp,
            )

        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg.clip_grad is not None and cfg.clip_grad > 0:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.clip_grad)
        scaler.step(optim)
        scaler.update()

        # EMA update
        with torch.no_grad():
            update_ema(teacher, student, m=m)

        losses.append(float(loss.detach().cpu()))

        if (step + 1) % 100 == 0:
            print(
                f"\nstep {step + 1}/{n_steps} | loss={losses[-1]:.4f} | lr={lr:.2e} wd={wd:.2e} m={m:.4f} t={t:.4f}"
            )

    return student, teacher, losses, teacher_entropies, student_entropies
