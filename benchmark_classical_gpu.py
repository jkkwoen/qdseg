#!/usr/bin/env python3
"""
Classical Segmentation CPU vs GPU Benchmark.

Measures per-file processing time for advanced, watershed, and thresholding
with use_gpu=False (CPU) and use_gpu=True (GPU via cuCIM+CuPy), across two
image sizes and several batch sizes.

Usage:
    python benchmark_classical_gpu.py [--sizes 256 512] [--max-files 50] [--json out.json]
"""

import argparse
import json
import platform
import time

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now() -> float:
    return time.perf_counter()


def _make_images(hw: int, n: int) -> list:
    rng = np.random.default_rng(42)
    return [rng.uniform(0, 10, size=(hw, hw)).astype(np.float32) for _ in range(n)]


META = {"pixel_nm": (1.0, 1.0)}


def _gpu_sync():
    """Synchronize CUDA stream for accurate GPU timing."""
    try:
        import cupy as cp
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_cpu(method_fn, images: list, file_counts: list) -> dict:
    """Benchmark CPU path (use_gpu=False). No warm-up needed."""
    results: dict = {}
    cumulative_ms = 0.0
    max_n = max(file_counts)

    for i, img in enumerate(images[:max_n]):
        t0 = _now()
        method_fn(img, META, use_gpu=False)
        cumulative_ms += (_now() - t0) * 1000
        n = i + 1
        if n in file_counts:
            results[n] = round(cumulative_ms / n, 1)

    return results


def bench_gpu(method_fn, images: list, file_counts: list) -> dict:
    """
    Benchmark GPU path (use_gpu=True).

    First call (warm-up) is timed separately to capture JIT / kernel compile cost.
    Subsequent calls measure steady-state throughput.
    """
    results: dict = {}
    max_n = max(file_counts)

    # --- warm-up call (first image) ---
    t0 = _now()
    method_fn(images[0], META, use_gpu=True)
    _gpu_sync()
    warmup_ms = (_now() - t0) * 1000

    # --- steady-state ---
    cumulative_ms = warmup_ms  # include warm-up in N=1 result
    for i, img in enumerate(images[:max_n]):
        if i == 0:
            # warm-up already done above
            n = 1
            if n in file_counts:
                results[n] = {
                    "avg_ms": round(warmup_ms, 1),
                    "warmup_ms": round(warmup_ms, 1),
                    "note": "warm-up only",
                }
            continue

        t0 = _now()
        method_fn(img, META, use_gpu=True)
        _gpu_sync()
        cumulative_ms += (_now() - t0) * 1000
        n = i + 1
        if n in file_counts:
            results[n] = {
                "avg_ms": round(cumulative_ms / n, 1),
                "warmup_ms": round(warmup_ms, 1),
                "note": "",
            }

    return results


# ---------------------------------------------------------------------------
# Table printing
# ---------------------------------------------------------------------------

def print_comparison_table(name: str, cpu_r: dict, gpu_r: dict, file_counts: list):
    avail = bool(gpu_r)
    print(f"\n  {'─'*72}")
    print(f"  {name}")
    print(f"  {'─'*72}")
    if not avail:
        print(f"    GPU not available — CPU results only")
        print(f"\n    {'N':>6}  {'CPU ms/file':>12}")
        print(f"    {'─'*6}  {'─'*12}")
        for n in file_counts:
            print(f"    {n:>6}  {cpu_r.get(n, 0):>10.1f}ms")
        return

    print(f"\n    {'N':>6}  {'CPU ms/file':>12}  {'GPU ms/file':>12}  {'Speedup':>9}  {'GPU warmup':>12}")
    print(f"    {'─'*6}  {'─'*12}  {'─'*12}  {'─'*9}  {'─'*12}")
    for n in file_counts:
        cpu_ms = cpu_r.get(n, 0.0)
        gpu_entry = gpu_r.get(n, {})
        gpu_ms = gpu_entry.get("avg_ms", 0.0) if isinstance(gpu_entry, dict) else 0.0
        warmup_ms = gpu_entry.get("warmup_ms", 0.0) if isinstance(gpu_entry, dict) else 0.0
        note = gpu_entry.get("note", "") if isinstance(gpu_entry, dict) else ""
        speedup = (cpu_ms / gpu_ms) if gpu_ms > 0 else float("nan")
        speedup_str = f"{speedup:.2f}x" if not (speedup != speedup) else "  n/a"
        note_tag = " *" if note else ""
        print(f"    {n:>6}  {cpu_ms:>10.1f}ms  {gpu_ms:>10.1f}ms  {speedup_str:>9}  {warmup_ms:>10.1f}ms{note_tag}")
    print(f"    (* N=1 GPU includes warm-up / kernel compile time)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CPU vs GPU classical segmentation benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512],
                        help="Image sizes to test, e.g. --sizes 256 512")
    parser.add_argument("--max-files", type=int, default=50,
                        help="Maximum number of images per run (default: 50)")
    parser.add_argument("--json", help="Save results to JSON file")
    args = parser.parse_args()

    file_counts_base = [1, 5, 10, 50]
    file_counts = sorted({n for n in file_counts_base if n <= args.max_files} | {args.max_files})

    print("=" * 74)
    print("  Classical Segmentation  CPU vs GPU Benchmark")
    print("=" * 74)
    print(f"\n  Host:    {platform.node()}")
    print(f"  Sizes:   {args.sizes}")
    print(f"  N files: {file_counts}")

    # Check GPU availability
    try:
        from qdseg._classical_gpu import is_gpu_available
        gpu_ok = is_gpu_available()
    except Exception:
        gpu_ok = False

    if gpu_ok:
        try:
            import cupy as cp
            dev = cp.cuda.Device(0)
            try:
                gpu_name = cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode()
            except Exception:
                gpu_name = "unknown"
            print(f"  GPU:     {gpu_name} (device 0)")
        except Exception:
            print(f"  GPU:     available (name unavailable)")
    else:
        print(f"  GPU:     not available — CPU-only results")

    from qdseg.segmentation import segment_advanced, segment_watershed, segment_thresholding

    methods = [
        ("advanced",     segment_advanced,     "Advanced    (Otsu+DT+DBSCAN+Voronoi)"),
        ("watershed",    segment_watershed,    "Watershed   (Gaussian+Sobel+WS)"),
        ("thresholding", segment_thresholding, "Thresholding (Otsu)"),
    ]

    all_results: dict = {
        "system": {
            "hostname": platform.node(),
            "gpu_available": gpu_ok,
        },
        "file_counts": file_counts,
        "sizes": args.sizes,
    }

    for hw in args.sizes:
        print(f"\n{'='*74}")
        print(f"  Image size: {hw}×{hw}")
        print(f"{'='*74}")

        images = _make_images(hw, max(file_counts))
        size_results: dict = {}

        for key, fn, label in methods:
            label_full = f"{label}  [{hw}×{hw}]"
            try:
                cpu_r = bench_cpu(fn, images, file_counts)
            except Exception as e:
                print(f"\n  {key} CPU ERROR: {e}")
                cpu_r = {}

            gpu_r: dict = {}
            if gpu_ok:
                try:
                    gpu_r = bench_gpu(fn, images, file_counts)
                except Exception as e:
                    print(f"\n  {key} GPU ERROR: {e}")

            print_comparison_table(label_full, cpu_r, gpu_r, file_counts)

            size_results[key] = {"cpu": cpu_r, "gpu": gpu_r}

        all_results[f"size_{hw}"] = size_results

    print(f"\n{'='*74}\n  Done.\n{'='*74}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved → {args.json}")


if __name__ == "__main__":
    main()
