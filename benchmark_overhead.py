#!/usr/bin/env python3
"""
Overhead Benchmark: Model loading vs. per-image inference.

Shows how startup overhead (model init + GPU warm-up) is amortized
when processing multiple files in a single session vs. per-file cold starts.

Usage:
    python benchmark_overhead.py [--size 512] [--max-files 50] [--json output.json]
"""

import argparse
import json
import platform
import time

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync(framework: str, device=None):
    if framework == "torch":
        import torch
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        elif device is not None and device.type == "mps":
            torch.mps.synchronize()


def _now():
    return time.perf_counter()


# ---------------------------------------------------------------------------
# StarDist overhead benchmark
# ---------------------------------------------------------------------------

def bench_stardist_overhead(hw: int, file_counts: list) -> dict:
    import numpy as np
    from stardist.models import StarDist2D

    results = {}

    # --- Model loading time ---
    t0 = _now()
    model = StarDist2D.from_pretrained("2D_versatile_fluo")
    t_load = (_now() - t0) * 1000  # ms

    # --- GPU warm-up (first inference, excluded from batch timing) ---
    warm_img = np.random.rand(hw, hw).astype(np.float32)
    t0 = _now()
    model.predict_instances(warm_img, prob_thresh=0.5, nms_thresh=0.4)
    t_warmup = (_now() - t0) * 1000  # ms

    results["model_load_ms"] = round(t_load, 1)
    results["gpu_warmup_ms"] = round(t_warmup, 1)
    results["startup_total_ms"] = round(t_load + t_warmup, 1)

    # --- Batch: measure per-file time for N files (after warm-up) ---
    max_n = max(file_counts)
    images = [np.random.rand(hw, hw).astype(np.float32) for _ in range(max_n)]

    per_file = {}
    cumulative_ms = 0.0
    for i, img in enumerate(images):
        t0 = _now()
        model.predict_instances(img, prob_thresh=0.5, nms_thresh=0.4)
        cumulative_ms += (_now() - t0) * 1000
        n = i + 1
        if n in file_counts:
            avg_infer_ms = cumulative_ms / n
            # effective cost per file if we account for startup amortized over N
            effective_cold_ms = results["startup_total_ms"] / n + avg_infer_ms
            per_file[n] = {
                "avg_infer_ms": round(avg_infer_ms, 1),
                "effective_per_file_ms": round(effective_cold_ms, 1),
                "startup_fraction_pct": round(
                    results["startup_total_ms"] / (results["startup_total_ms"] + cumulative_ms) * 100, 1
                ),
            }

    results["per_file_counts"] = per_file
    return results


# ---------------------------------------------------------------------------
# Cellpose overhead benchmark
# ---------------------------------------------------------------------------

def bench_cellpose_overhead(hw: int, file_counts: list, device_type: str) -> dict:
    import numpy as np
    from cellpose import models
    import torch

    if device_type == "cuda":
        dev = torch.device("cuda")
    elif device_type == "mps":
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    results = {}

    # --- Model loading time ---
    t0 = _now()
    model = models.CellposeModel(gpu=(dev.type != "cpu"), device=dev)
    t_load = (_now() - t0) * 1000

    # --- GPU warm-up ---
    warm_img = (np.random.rand(hw, hw).astype(np.float32) * 255)
    t0 = _now()
    model.eval(warm_img, diameter=30, flow_threshold=0.4, cellprob_threshold=0.0)
    _sync("torch", dev)
    t_warmup = (_now() - t0) * 1000

    results["model_load_ms"] = round(t_load, 1)
    results["gpu_warmup_ms"] = round(t_warmup, 1)
    results["startup_total_ms"] = round(t_load + t_warmup, 1)

    # --- Batch timing ---
    max_n = max(file_counts)
    images = [(np.random.rand(hw, hw).astype(np.float32) * 255) for _ in range(max_n)]

    per_file = {}
    cumulative_ms = 0.0
    for i, img in enumerate(images):
        t0 = _now()
        model.eval(img, diameter=30, flow_threshold=0.4, cellprob_threshold=0.0)
        _sync("torch", dev)
        cumulative_ms += (_now() - t0) * 1000
        n = i + 1
        if n in file_counts:
            avg_infer_ms = cumulative_ms / n
            effective_cold_ms = results["startup_total_ms"] / n + avg_infer_ms
            per_file[n] = {
                "avg_infer_ms": round(avg_infer_ms, 1),
                "effective_per_file_ms": round(effective_cold_ms, 1),
                "startup_fraction_pct": round(
                    results["startup_total_ms"] / (results["startup_total_ms"] + cumulative_ms) * 100, 1
                ),
            }

    results["per_file_counts"] = per_file
    return results


# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------

def print_table(name: str, r: dict, file_counts: list):
    print(f"\n  {'─'*58}")
    print(f"  {name}")
    print(f"  {'─'*58}")
    print(f"    Model load:   {r['model_load_ms']:>8.1f} ms")
    print(f"    GPU warm-up:  {r['gpu_warmup_ms']:>8.1f} ms")
    print(f"    Startup total:{r['startup_total_ms']:>8.1f} ms")
    print()
    print(f"    {'N files':>8}  {'Infer/file':>12}  {'Effective/file*':>16}  {'Startup %':>10}")
    print(f"    {'─'*8}  {'─'*12}  {'─'*16}  {'─'*10}")
    for n in file_counts:
        p = r["per_file_counts"][n]
        print(
            f"    {n:>8}  {p['avg_infer_ms']:>10.1f}ms  "
            f"{p['effective_per_file_ms']:>14.1f}ms  "
            f"{p['startup_fraction_pct']:>9.1f}%"
        )
    print(f"\n    * Effective/file = startup/N + avg_infer  (cold-start amortized)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512, help="Image size HxH (default: 512)")
    parser.add_argument("--max-files", type=int, default=50, help="Max number of files (default: 50)")
    parser.add_argument("--json", help="Save results to JSON file")
    args = parser.parse_args()

    hw = args.size
    file_counts = [n for n in [1, 2, 5, 10, 20, 50] if n <= args.max_files]
    if args.max_files not in file_counts:
        file_counts.append(args.max_files)
    file_counts = sorted(set(file_counts))

    # Device detection
    device_type = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device_type = "cuda"
            device_name = torch.cuda.get_device_name(0)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_type = "mps"
            device_name = "Apple MPS"
        else:
            device_name = "CPU"
    except ImportError:
        device_name = "CPU"

    print("=" * 60)
    print("  Overhead Benchmark — model init vs. per-file inference")
    print("=" * 60)
    print(f"\n  Host:   {platform.node()}")
    print(f"  GPU:    {device_name}")
    print(f"  Size:   {hw}x{hw}  |  N = {file_counts}")

    all_results = {
        "system": {"hostname": platform.node(), "device": device_name, "image_size": hw},
        "file_counts": file_counts,
    }

    # StarDist
    try:
        r = bench_stardist_overhead(hw, file_counts)
        all_results["stardist"] = r
        print_table(f"StarDist (TensorFlow)  —  {hw}x{hw}", r, file_counts)
    except Exception as e:
        print(f"\n  StarDist ERROR: {e}")
        all_results["stardist"] = {"error": str(e)}

    # Cellpose
    try:
        r = bench_cellpose_overhead(hw, file_counts, device_type)
        all_results["cellpose"] = r
        print_table(f"Cellpose (PyTorch)  —  {hw}x{hw}", r, file_counts)
    except Exception as e:
        print(f"\n  Cellpose ERROR: {e}")
        all_results["cellpose"] = {"error": str(e)}

    print(f"\n{'='*60}\n  Done.\n{'='*60}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.json}")


if __name__ == "__main__":
    main()
