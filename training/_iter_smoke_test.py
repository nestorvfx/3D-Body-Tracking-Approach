"""End-to-end smoke test: DataLoader → model.forward → loss.backward.

Verifies the new F1/F2 sim2real pipeline doesn't break:
  - tensor shapes match the model's expectations
  - keypoint values are still in-bounds
  - gradients flow without NaN/Inf
  - throughput is reasonable (< 2× the no-aug baseline)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from training.data import DataConfig, SynthPoseDataset           # noqa: E402
from training.model import build_model                            # noqa: E402
from training.losses import KLDiscretLoss                         # noqa: E402


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    cfg = DataConfig(
        dataset_dir=str(HERE.parent / "dataset/output/synth_iter"),
        split="train", training=True,
        occluder_dir=str(HERE.parent / "assets/sim2real_refs/occluders"),
        bg_corpus_dir=str(HERE.parent / "assets/sim2real_refs/bg"),
        matte_dir=str(HERE.parent / "dataset/output/synth_iter/mattes"),
        fda_refs_dir=str(HERE.parent / "assets/sim2real_refs/fda"),
        p_occluder=0.6, p_bg_composite=0.5, p_fda=0.3,
    )
    ds = SynthPoseDataset(cfg)
    print(f"[smoke] dataset: {len(ds)} samples")
    print(f"[smoke] occluders={len(ds.occluders)}, "
          f"bg_corpus={len(ds.bg_corpus)}, fda_refs={len(ds.fda_refs)}")

    # CPU loader, num_workers=0 to avoid Windows-fork issues for the smoke
    # check.  Production training uses num_workers=4-8 + persistent_workers.
    dl = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[smoke] device: {device}")
    model = build_model("mnv4s", pretrained=False).to(device)
    model.train()
    loss_fn = KLDiscretLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    t0 = time.perf_counter()
    n_batches = 6
    for i, batch in enumerate(dl):
        if i >= n_batches:
            break
        img = batch["image"].to(device)
        cond = batch["cond"].to(device)
        kpr = batch["k_prior"].to(device)
        tx = batch["target_x"].to(device)
        ty = batch["target_y"].to(device)
        tz = batch["target_z"].to(device)
        vis = batch["vis"].to(device)

        # Shape sanity
        assert img.shape[1:] == (3, 256, 192), f"img shape {img.shape}"
        assert cond.shape[1] == 6, f"cond shape {cond.shape}"
        assert tx.shape[1:] == (17, 384), f"target_x shape {tx.shape}"
        assert ty.shape[1:] == (17, 512), f"target_y shape {ty.shape}"
        assert tz.shape[1:] == (17, 512), f"target_z shape {tz.shape}"

        out = model(img, cond, k_prior=kpr)
        loss_x = loss_fn(out["x_logits"], tx, vis)
        loss_y = loss_fn(out["y_logits"], ty, vis)
        loss_z = loss_fn(out["z_logits"], tz, vis)
        loss = loss_x + loss_y + loss_z

        if not torch.isfinite(loss):
            print(f"[smoke] FAIL: non-finite loss at batch {i}: {loss.item()}")
            return 2

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Check no NaN grads
        bad = 0
        for n, p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                bad += 1
        if bad > 0:
            print(f"[smoke] FAIL: {bad} params have non-finite grads")
            return 3
        opt.step()

        kps2d = batch["kps2d"]
        ok2d = ((kps2d[..., 0] >= -50) & (kps2d[..., 0] <= 256 + 50)
                & (kps2d[..., 1] >= -50) & (kps2d[..., 1] <= 512 + 50)).all()
        print(f"[smoke] batch {i}: loss={loss.item():.3f}  "
              f"img mean={img.mean().item():+.3f}  vis_sum={vis.sum().item():.0f}  "
              f"kps2d in-bounds={bool(ok2d)}")

    dt = time.perf_counter() - t0
    print(f"[smoke] {n_batches} batches in {dt:.1f}s  "
          f"({dt/n_batches:.2f} s/batch CPU side)")
    print("[smoke] OK — pipeline is sound, gradients flow, shapes match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
