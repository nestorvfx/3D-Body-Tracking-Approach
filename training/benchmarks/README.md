# External benchmark evaluators

Modular sub-system for evaluating a trained checkpoint on commercial-clean
3D-pose benchmarks.

## Design

- `base.py` — `Benchmark` protocol (duck-typed), shared utilities.
- `run.py` — generic CLI: loads a checkpoint, iterates any benchmark, writes
  a markdown report with MPJPE / PA-MPJPE / PCK and a per-subject breakdown.
- Each benchmark = one module that exposes
  `build_benchmark(data_root, **kwargs)` returning an object with `.name`,
  `.coco17_to_bench`, `.bench_joint_names`, `__len__`, `.iter_samples()`.

To add a benchmark: drop a new module in this folder. No changes to `run.py`.

## Licensing rule

This project is built for commercial deployment. **Only commercial-clean
benchmarks should live in this folder.** Research-only datasets (Human3.6M,
MPI-INF-3DHP, 3DPW, AGORA, BEDLAM, EMDB, RICH, MoYo) explicitly forbid
"use in a commercial service" or "incorporation in a commercial product"
in their licenses, and that includes using the data to validate a model
that ships in a commercial product.

Acceptable sources for new benchmark modules:

- **AIST++ annotations** — CC-BY-4.0 (joint annotations, not source videos).
- **COCO annotations** — CC-BY-4.0 (annotations; image licenses vary
  per-Flickr-photo, filter to CC-BY-licensed images).
- **Held-out subset of our own synthetic pipeline** — fully owned, any
  license you choose.
- **Internally-captured real-world clips** with signed releases.

If you want to evaluate on a research-only benchmark for an academic
paper, do it in a separate branch / fork that won't be redistributed
with the product.

## Run

```powershell
python -m training.benchmarks.run `
    --benchmark <module-name> `
    --data-root "<path-to-test-set>" `
    --ckpt training/runs/<run>/best.pt `
    --report reports/<run>_<benchmark>.md `
    --stride 5
```

`--stride 5` samples every 5th frame (faster eval). Output: markdown
report with MPJPE / PA-MPJPE / PCK plus per-subject (or per-clip)
breakdown.
