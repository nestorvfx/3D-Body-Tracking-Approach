"""Single-GPU training loop for the RTMPose3D-MobileNetV4 body model.

Designed for a 6 GB RTX 4050: batch 16, AMP, AdamW, cosine LR.
Reports MPJPE + PA-MPJPE on the val split every epoch to TensorBoard.

Usage:
    python -m training.train --dataset-dir dataset/output/synth_v1 \
        --out-dir training/runs/baseline_v1 --epochs 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ---------------------------------------------------------------------------
# DDP helpers — active only when launched via `torchrun --nproc_per_node=N`.
# When run as plain `python -m training.train`, these all no-op and we stay
# on the original single-GPU code path.
# ---------------------------------------------------------------------------
def _ddp_enabled() -> bool:
    return "LOCAL_RANK" in os.environ


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def _world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", 1))


def _is_main() -> bool:
    return _local_rank() == 0


def _ddp_print(*a, **kw):
    if _is_main():
        print(*a, **kw)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from training.data import DataConfig, SynthPoseDataset   # noqa: E402
from training.model import build_model, decode_simcc     # noqa: E402
from training.losses import KLDiscretLoss, BoneLoss, MPJPE, pa_mpjpe   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3,
                    help="peak LR after warmup.  3e-3 (empirically verified) "
                         "escapes the log(K) uniform-output local minimum; "
                         "3e-4 is too conservative for this head + KL loss "
                         "combination and the model stalls at the ceiling.")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--kl-beta", type=float, default=10.0)
    p.add_argument("--bone-weight", type=float, default=2.0)
    p.add_argument("--z-loss-weight", type=float, default=1.0,
                    help="multiplier on Z-axis KL loss (default 1.0 = "
                         "equal with X/Y; bump if Z head lags).")
    p.add_argument("--root-z-weight", type=float, default=1.0,
                    help="multiplier on root-depth L1 loss (metres). "
                         "GT root_z range ~1-10m so L1 magnitude ~1-5; "
                         "weight 1.0 puts it on par with per-axis KL.")
    p.add_argument("--warmup-iters", type=int, default=1000,
                    help="linear LR warmup iters before cosine anneal; "
                         "1k steps matches RTMPose canonical recipe.")
    p.add_argument("--backbone", default="mnv4s")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--resume", default="", help="checkpoint to resume from")
    p.add_argument("--fresh-schedule", action="store_true",
                    help="when resuming, load ONLY the model weights (ignore "
                         "saved opt/sched/scaler/epoch).  Use this to start a "
                         "new cosine cycle from an existing checkpoint once "
                         "the previous schedule has annealed to ~0 LR.")
    p.add_argument("--input-w", type=int, default=192)
    p.add_argument("--input-h", type=int, default=256)
    return p.parse_args()


def backproject_crop_to_cam3d(kps_uv_crop, depth_z, K, bbox_xywh, input_wh):
    """kps_uv_crop: [B,J,2] in cropped input frame (W,H = input_wh).
    depth_z:      [B,J] absolute Z in camera frame (metres).
    K:            [B,3,3] ORIGINAL-image intrinsics (pixels).
    bbox_xywh:    [B,4] the bbox used to make the crop (ORIGINAL image pixels).
    input_wh:     (W,H) of the model's crop input.

    Returns [B,J,3] in camera frame.  Geometry: un-warp crop pixel to
    full-image pixel with the inverse of the crop's affine, then apply
    K^-1 * depth.
    """
    W_in, H_in = input_wh
    bx = bbox_xywh[:, 0].unsqueeze(-1)
    by = bbox_xywh[:, 1].unsqueeze(-1)
    bw = bbox_xywh[:, 2].unsqueeze(-1)
    bh = bbox_xywh[:, 3].unsqueeze(-1)
    u_crop = kps_uv_crop[..., 0]
    v_crop = kps_uv_crop[..., 1]
    u_full = bx + u_crop / W_in * bw
    v_full = by + v_crop / H_in * bh
    fx = K[:, 0, 0].unsqueeze(-1)
    fy = K[:, 1, 1].unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1)
    x = (u_full - cx) / fx * depth_z
    y = (v_full - cy) / fy * depth_z
    return torch.stack([x, y, depth_z], dim=-1)


def _root_relative(pts_3d: torch.Tensor, root_indices=(11, 12)) -> torch.Tensor:
    """Subtract the pelvis (mean of left+right hip) from every joint.

    Standard 3D-pose evaluation frame: subtract the pelvis (mean of L/R
    hips) from every joint.  A single monocular RGB frame has no
    absolute-scale signal, so measuring absolute camera-frame MPJPE
    pins to a structural floor that reflects the dataset's depth spread,
    not model skill.
    """
    root = pts_3d[:, list(root_indices)].mean(dim=1, keepdim=True)   # [B, 1, 3]
    return pts_3d - root


def run_val(model, loader, device, kl_loss, bone_loss, mpjpe_metric, input_wh):
    model.eval()
    tot = {"kl": 0.0, "bone": 0.0, "mpjpe": 0.0, "pa_mpjpe": 0.0,
           "root_z_err": 0.0, "n": 0}
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            tx = batch["target_x"].to(device, non_blocking=True)
            ty = batch["target_y"].to(device, non_blocking=True)
            tz = batch["target_z"].to(device, non_blocking=True)
            gt3d = batch["kps3d"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            root_z_gt = batch["root_z"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            k_prior = batch["k_prior"].to(device, non_blocking=True)
            K = batch["camera_K"].to(device, non_blocking=True)
            bbox = batch["bbox_pre_crop"].to(device, non_blocking=True)

            out = model(img, cond, k_prior=k_prior)
            # val loss kept unweighted so numbers stay comparable to prior runs.
            kl = (
                kl_loss(out["x_logits"], tx, vis)
                + kl_loss(out["y_logits"], ty, vis)
                + kl_loss(out["z_logits"], tz, vis)
            )
            # Use the MODEL's predicted root_z (CLIFF-style) for decoding.
            # This is the number that matters at inference time — so val
            # reflects realistic deployment behaviour.
            pred_root_z = out["root_z"]
            kps2d, kps_z = decode_simcc(
                out["x_logits"], out["y_logits"], out["z_logits"],
                input_wh=input_wh, root_z=pred_root_z, mode="argmax")
            # Back-project into camera frame so we have a 3D pose to
            # subtract the pelvis from.
            pred_3d = backproject_crop_to_cam3d(
                kps2d, kps_z, K, bbox, input_wh)
            # Root-relative compare — removes the absolute-depth floor
            # that a monocular model literally cannot reach.
            pred_rel = _root_relative(pred_3d)
            gt_rel = _root_relative(gt3d)
            b = bone_loss(pred_rel, gt_rel, vis)
            m = mpjpe_metric(pred_rel, gt_rel, vis)
            pa = pa_mpjpe(pred_rel, gt_rel)
            rz_err = (pred_root_z - root_z_gt).abs().mean()
            n = img.shape[0]
            tot["kl"] += float(kl) * n
            tot["bone"] += float(b) * n
            tot["mpjpe"] += float(m) * n
            tot["pa_mpjpe"] += float(pa) * n
            tot["root_z_err"] += float(rz_err) * n
            tot["n"] += n
    return {k: (v / max(1, tot["n"])) if k != "n" else v for k, v in tot.items()}


def main():
    args = parse_args()

    # Initialize distributed process group if launched via torchrun.
    if _ddp_enabled():
        # nccl on Linux (vast.ai), gloo on Windows (stock wheels don't ship NCCL).
        backend = "nccl" if dist.is_nccl_available() else "gloo"
        dist.init_process_group(backend=backend)
        torch.cuda.set_device(_local_rank())
        device = torch.device(f"cuda:{_local_rank()}")
        _ddp_print(f"[ddp] backend={backend}  world_size={_world_size()}  "
                   f"rank={_local_rank()}  device={torch.cuda.get_device_name(device)}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise SystemExit("CUDA required; fix your driver/env.")
        print(f"[device] {torch.cuda.get_device_name(0)}")

    # TF32 matmul on Ada/Blackwell — free speedup.
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.out_dir)
    if _is_main():
        out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "tb")) if _is_main() else None

    input_wh = (args.input_w, args.input_h)

    train_cfg = DataConfig(dataset_dir=args.dataset_dir, split="train",
                            input_wh=input_wh, training=True)
    val_cfg = DataConfig(dataset_dir=args.dataset_dir, split="val",
                          input_wh=input_wh, training=False, photometric=False)
    train_ds = SynthPoseDataset(train_cfg)
    val_ds = SynthPoseDataset(val_cfg)
    _ddp_print(f"[data] train={len(train_ds)}, val={len(val_ds)}")

    # DistributedSampler under DDP — partitions the dataset across ranks
    # so each GPU processes a unique subset per epoch.
    if _ddp_enabled():
        train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
        val_sampler = DistributedSampler(val_ds, shuffle=False, drop_last=False)
    else:
        train_sampler = None
        val_sampler = None

    train_ld = DataLoader(train_ds, batch_size=args.batch,
                           shuffle=(train_sampler is None),
                           sampler=train_sampler,
                           num_workers=args.workers, pin_memory=True,
                           drop_last=True, persistent_workers=args.workers > 0,
                           prefetch_factor=4 if args.workers > 0 else None)
    val_ld = DataLoader(val_ds, batch_size=args.batch,
                         shuffle=False, sampler=val_sampler,
                         num_workers=args.workers, pin_memory=True,
                         persistent_workers=args.workers > 0)

    model = build_model(args.backbone, pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    _ddp_print(f"[model] {args.backbone}, params={n_params:,}")
    if _ddp_enabled():
        model = DDP(model, device_ids=[_local_rank()])

    # Scale LR linearly with world size (Goyal et al. rule): effective
    # batch is batch * world_size, so LR should match.  Ensures parity
    # between single-GPU and multi-GPU runs.
    effective_lr = args.lr * _world_size()
    if _world_size() > 1:
        _ddp_print(f"[lr] scaling base {args.lr:.1e} -> {effective_lr:.1e} "
                   f"(world_size={_world_size()})")
    opt = torch.optim.AdamW(model.parameters(), lr=effective_lr,
                             weight_decay=args.weight_decay)
    total_iters = args.epochs * max(1, len(train_ld))
    warmup = max(0, min(args.warmup_iters, total_iters - 1))
    if warmup > 0:
        # Linear warmup (0.01 → 1.0 of base LR) over `warmup` iters, then
        # cosine annealing to 0 for the rest.  SequentialLR chains them.
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=0.01, end_factor=1.0, total_iters=warmup)
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_iters - warmup)
        sched = torch.optim.lr_scheduler.SequentialLR(
            opt, schedulers=[warmup_sched, cosine_sched],
            milestones=[warmup])
    else:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_iters)
    # bf16 autocast on Ada/Blackwell — no GradScaler needed (bf16 has same
    # dynamic range as fp32, so no loss scaling for numerical stability).
    # fp16 needs GradScaler; we keep that path for older GPUs via --use-fp16.
    use_bf16 = getattr(args, "use_bf16", True) and args.amp
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and not use_bf16)

    kl_loss = KLDiscretLoss(beta=args.kl_beta).to(device)
    bone_loss = BoneLoss().to(device)
    mpjpe_metric = MPJPE().to(device)

    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        # Handle DDP-prefix mismatch: ckpts saved under DDP have "module."
        # prefix; ckpts saved single-GPU don't.  Unwrap.
        model_sd = ckpt["model"]
        if next(iter(model_sd)).startswith("module.") and not _ddp_enabled():
            model_sd = {k[len("module."):]: v for k, v in model_sd.items()}
        (model.module if _ddp_enabled() else model).load_state_dict(model_sd)
        if args.fresh_schedule:
            _ddp_print(f"[resume] fresh schedule — keeping only model weights "
                       f"from epoch {ckpt.get('epoch', '?')}")
        else:
            opt.load_state_dict(ckpt["opt"])
            sched.load_state_dict(ckpt["sched"])
            scaler.load_state_dict(ckpt["scaler"])
            start_epoch = ckpt["epoch"] + 1
            _ddp_print(f"[resume] from epoch {start_epoch}")

    step = start_epoch * len(train_ld)
    best_mpjpe = float("inf")
    total_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # proper reshuffle every epoch
        model.train()
        t0 = time.time()
        total_loss = 0.0
        # tqdm only on rank 0 to avoid garbled output.
        iterable = tqdm(train_ld, desc=f"epoch {epoch+1}/{args.epochs}",
                         ncols=110, leave=False, dynamic_ncols=False) \
            if _is_main() else train_ld
        for it, batch in enumerate(iterable):
            img = batch["image"].to(device, non_blocking=True)
            tx = batch["target_x"].to(device, non_blocking=True)
            ty = batch["target_y"].to(device, non_blocking=True)
            tz = batch["target_z"].to(device, non_blocking=True)
            gt3d = batch["kps3d"].to(device, non_blocking=True)
            vis = batch["vis"].to(device, non_blocking=True)
            cond = batch["cond"].to(device, non_blocking=True)
            root_z_gt = batch["root_z"].to(device, non_blocking=True)
            k_prior = batch["k_prior"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=autocast_dtype,
                                      enabled=args.amp):
                out = model(img, cond, k_prior=k_prior)
                kl_x = kl_loss(out["x_logits"], tx, vis)
                kl_y = kl_loss(out["y_logits"], ty, vis)
                kl_z = kl_loss(out["z_logits"], tz, vis)
                kl = kl_x + kl_y + args.z_loss_weight * kl_z
                # Factorized RootNet loss in LOG-GAMMA space (Moon ICCV 2019
                # + Zhang 2021).  Target: log(gamma_total) = log(root_z_gt /
                # k_prior).  Predicted: log_gamma_size + log_gamma_pose.
                # Huber / smooth-L1 in log-space gives:
                #   1. Tight, bounded targets (~log(1) = 0 for canonical,
                #      log(2) ≈ 0.7 for tall person, log(0.5) ≈ -0.7 for kid)
                #   2. Broken samples (root_z = 10000 m) map to log(gamma) ≈
                #      8, still bounded — ±1 gradient per outlier
                #   3. Multiplicative scale errors become additive (symmetric)
                log_gamma_total_gt = torch.log(
                    torch.clamp(root_z_gt, min=0.1) /
                    torch.clamp(k_prior, min=0.1))
                log_gamma_total_pred = (out["log_gamma_size"]
                                         + out["log_gamma_pose"])
                root_z_loss = torch.nn.functional.smooth_l1_loss(
                    log_gamma_total_pred, log_gamma_total_gt, beta=0.5)
                loss = kl + args.root_z_weight * root_z_loss
            if use_bf16:
                # bf16 autocast — no loss scaling required.
                loss.backward()
                opt.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            sched.step()

            total_loss += float(loss)
            running = total_loss / (it + 1)
            if _is_main():
                iterable.set_postfix(loss=f"{running:.3f}",
                                       lr=f"{opt.param_groups[0]['lr']:.1e}")
                if step % 20 == 0 and writer is not None:
                    writer.add_scalar("train/loss_kl", float(kl), step)
                    writer.add_scalar("train/loss_kl_x", float(kl_x), step)
                    writer.add_scalar("train/loss_kl_y", float(kl_y), step)
                    writer.add_scalar("train/loss_kl_z", float(kl_z), step)
                    writer.add_scalar("train/loss_root_z",
                                      float(root_z_loss), step)
                    writer.add_scalar("train/lr",
                                      opt.param_groups[0]["lr"], step)
            step += 1
        if _is_main():
            iterable.close()

        dt = time.time() - t0
        avg = total_loss / max(1, len(train_ld))

        # Validation runs only on rank 0 — deterministic, avoids
        # cross-rank reduce.  For large val sets we could DistributedSampler
        # + all_reduce but our val is 10-50 k samples, takes seconds.
        if _is_main():
            val_stats = run_val(model.module if _ddp_enabled() else model,
                                 val_ld, device, kl_loss, bone_loss,
                                 mpjpe_metric, input_wh)
            elapsed = time.time() - total_start
            remaining = elapsed / (epoch - start_epoch + 1) * (args.epochs - epoch - 1)
            mm, ss = divmod(int(remaining), 60)
            star = "  <- best" if val_stats["mpjpe"] < best_mpjpe else ""
            print(f"ep {epoch+1:>3d}/{args.epochs}  "
                  f"train_loss={avg:6.3f}  "
                  f"val_mpjpe={val_stats['mpjpe']*1000:6.1f}mm  "
                  f"val_pa={val_stats['pa_mpjpe']*1000:6.1f}mm  "
                  f"bone={val_stats['bone']*1000:5.1f}mm  "
                  f"rz_err={val_stats['root_z_err']*1000:5.0f}mm  "
                  f"{dt:4.0f}s/ep  ETA {mm:02d}:{ss:02d}{star}",
                  flush=True)
            if writer is not None:
                writer.add_scalar("val/kl", val_stats["kl"], epoch)
                writer.add_scalar("val/mpjpe_mm", val_stats["mpjpe"] * 1000, epoch)
                writer.add_scalar("val/pa_mpjpe_mm", val_stats["pa_mpjpe"] * 1000, epoch)
                writer.add_scalar("val/bone_mm", val_stats["bone"] * 1000, epoch)

            ckpt = {
                "epoch": epoch,
                "model": (model.module if _ddp_enabled() else model).state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "scaler": scaler.state_dict(),
                "val": val_stats,
                "args": vars(args),
            }
            torch.save(ckpt, out_dir / "last.pt")
            if val_stats["mpjpe"] < best_mpjpe:
                best_mpjpe = val_stats["mpjpe"]
                torch.save(ckpt, out_dir / "best.pt")

        # Sync all ranks at epoch boundary so non-0 ranks don't race ahead
        # while rank 0 runs val + saves.
        if _ddp_enabled():
            dist.barrier()

    if _is_main():
        if writer is not None:
            writer.close()
        with (out_dir / "final_metrics.json").open("w") as f:
            json.dump({
                "best_mpjpe_mm": best_mpjpe * 1000,
                "last_val": val_stats,
                "params": n_params,
                "backbone": args.backbone,
            }, f, indent=2)
        print(f"[done] best val MPJPE: {best_mpjpe * 1000:.1f}mm")

    if _ddp_enabled():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
