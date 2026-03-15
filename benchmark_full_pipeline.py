#!/usr/bin/env python3
"""
Full pipeline benchmark: corrections + segmentation (classical / stardist / cellpose)

Usage:
    python benchmark_full_pipeline.py [--data-dir /path/to/xqd] [--json out.json]
"""

import argparse
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.perf_counter()


def _gpu_sync():
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def _torch_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------

def _gpu_info() -> str:
    try:
        import subprocess
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip().splitlines()[0] if r.returncode == 0 else "N/A"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Load + corrections
# ---------------------------------------------------------------------------

def load_and_correct(path: Path):
    """Load XQD, apply 1st-order correction + flat (median, GPU auto)."""
    from qdseg.io import load_height_nm
    from qdseg.corrections import AFMCorrections

    height, meta = load_height_nm(path)
    corr = AFMCorrections()

    height = corr.correct_1st(height)
    corr.set_flat_method('median')
    height = corr.correct_flat(height, use_gpu=None)   # GPU auto
    height = corr.correct_baseline(height)
    return height, meta


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_corrections(files: list) -> dict:
    """Measure correction time per file."""
    from qdseg.io import load_height_nm
    from qdseg.corrections import AFMCorrections

    corr = AFMCorrections()
    corr.set_flat_method('median')

    times = []
    for p in files:
        height, meta = load_height_nm(p)
        t0 = _now()
        h = corr.correct_1st(height)
        h = corr.correct_flat(h, use_gpu=None)
        _gpu_sync()
        h = corr.correct_baseline(h)
        times.append((_now() - t0) * 1000)

    return {
        "times_ms": times,
        "mean_ms": round(float(np.mean(times)), 1),
        "median_ms": round(float(np.median(times)), 1),
        "total_ms": round(float(np.sum(times)), 1),
    }


def _bench_seg(fn, preloaded: list, warmup: bool = False) -> dict:
    """Run segmentation fn on pre-corrected images, return timing dict."""
    times = []

    if warmup and preloaded:
        # warm-up: first call (model load / JIT)
        t0 = _now()
        fn(preloaded[0][0], preloaded[0][1])
        _gpu_sync()
        _torch_sync()
        warmup_ms = (_now() - t0) * 1000
    else:
        warmup_ms = None

    for height, meta in preloaded:
        t0 = _now()
        fn(height, meta)
        _gpu_sync()
        _torch_sync()
        times.append((_now() - t0) * 1000)

    result = {
        "times_ms": times,
        "mean_ms": round(float(np.mean(times)), 1),
        "median_ms": round(float(np.median(times)), 1),
        "total_ms": round(float(np.sum(times)), 1),
    }
    if warmup_ms is not None:
        result["warmup_ms"] = round(warmup_ms, 1)
    return result


def bench_classical(preloaded: list) -> dict:
    from qdseg.segmentation import segment_advanced
    fn = lambda h, m: segment_advanced(h, m, use_gpu=None)
    return _bench_seg(fn, preloaded, warmup=False)


def bench_stardist(preloaded: list) -> dict:
    from qdseg.segmentation import segment_stardist
    fn = lambda h, m: segment_stardist(h, m, use_gpu=True)
    return _bench_seg(fn, preloaded, warmup=True)


def bench_cellpose(preloaded: list) -> dict:
    from qdseg.segmentation import segment_cellpose
    fn = lambda h, m: segment_cellpose(h, m, gpu=True)
    return _bench_seg(fn, preloaded, warmup=True)


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _bar(value: float, max_val: float, width: int = 20) -> str:
    filled = int(round(value / max_val * width)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def print_results(results: dict, n_files: int):
    print()
    print("=" * 70)
    print(f"  Full Pipeline Benchmark  —  {n_files} real XQD files")
    print(f"  GPU: {results['gpu']}")
    print(f"  Host: {results['host']}")
    print("=" * 70)

    methods = ["corrections", "classical", "stardist", "cellpose"]
    labels = {
        "corrections": "Corrections (1st+flat+base)",
        "classical":   "Classical  (advanced)      ",
        "stardist":    "StarDist                   ",
        "cellpose":    "CellPose                   ",
    }
    means = {m: results[m]["mean_ms"] for m in methods if m in results}
    max_mean = max(means.values()) if means else 1

    print()
    print(f"  {'Method':<30}  {'Mean/file':>9}  {'Median':>8}  {'Total (all)':>11}  Bar")
    print(f"  {'-'*30}  {'-'*9}  {'-'*8}  {'-'*11}  {'-'*20}")

    for m in methods:
        if m not in results:
            continue
        r = results[m]
        label = labels[m]
        mean = r["mean_ms"]
        med  = r["median_ms"]
        total = r["total_ms"]
        bar = _bar(mean, max_mean)
        extra = ""
        if "warmup_ms" in r:
            extra = f"  (1st-call: {r['warmup_ms']:.0f} ms)"
        print(f"  {label}  {mean:>8.1f}ms  {med:>7.1f}ms  {total:>9.0f}ms  {bar}{extra}")

    # Pipeline total
    seg_methods = ["classical", "stardist", "cellpose"]
    corr_mean = results.get("corrections", {}).get("mean_ms", 0)
    print()
    print(f"  {'Pipeline total (corr + seg), mean/file':}")
    for m in seg_methods:
        if m not in results:
            continue
        seg_mean = results[m]["mean_ms"]
        total_mean = corr_mean + seg_mean
        print(f"    {labels[m].strip():<28}  {total_mean:>8.1f}ms/file")

    print()
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/home/jkkwoen/qdseg_batch_data/xqd",
                        help="Directory containing .xqd files")
    parser.add_argument("--max-files", type=int, default=50)
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("*.xqd"))[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No .xqd files in {data_dir}")
    n = len(files)
    print(f"\n  Loading {n} XQD files from {data_dir} ...")

    # Pre-load and correct all files (skip corrupt ones)
    preloaded = []
    good_files = []
    for p in files:
        try:
            height, meta = load_and_correct(p)
            preloaded.append((height, meta))
            good_files.append(p)
        except Exception as e:
            print(f"  [SKIP] {p.name}: {e}")
    n = len(preloaded)
    print(f"  Loaded {n} files. Shapes: {preloaded[0][0].shape}")

    results = {
        "host": platform.node(),
        "gpu": _gpu_info(),
        "n_files": n,
    }

    # ── Corrections ──────────────────────────────────────────────────────────
    print("\n  Benchmarking corrections ...")
    results["corrections"] = bench_corrections(good_files)

    # ── Classical ────────────────────────────────────────────────────────────
    print("  Benchmarking classical (advanced) ...")
    try:
        results["classical"] = bench_classical(preloaded)
    except Exception as e:
        print(f"    [SKIP] classical failed: {e}")

    # ── StarDist ─────────────────────────────────────────────────────────────
    print("  Benchmarking StarDist (warm-up incl.) ...")
    try:
        results["stardist"] = bench_stardist(preloaded)
    except Exception as e:
        print(f"    [SKIP] stardist failed: {e}")

    # ── CellPose ─────────────────────────────────────────────────────────────
    print("  Benchmarking CellPose (warm-up incl.) ...")
    try:
        results["cellpose"] = bench_cellpose(preloaded)
    except Exception as e:
        print(f"    [SKIP] cellpose failed: {e}")

    # ── Print ─────────────────────────────────────────────────────────────────
    print_results(results, n)

    if args.json:
        out = {k: v for k, v in results.items()}
        for m in ["corrections", "classical", "stardist", "cellpose"]:
            if m in out:
                out[m] = {k: v for k, v in out[m].items() if k != "times_ms"}
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"  Results saved → {args.json}")


if __name__ == "__main__":
    main()
