#!/usr/bin/env python3
"""
Classical Segmentation Benchmark: advanced, watershed, thresholding.

Classical methods have no model-loading overhead (pure CPU scipy/skimage),
so this focuses on per-image processing time and N-file batch scaling.

Usage:
    python benchmark_classical.py [--size 512] [--max-files 50] [--json output.json]
"""

import argparse
import json
import platform
import time

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now():
    return time.perf_counter()


def _make_images(hw: int, n: int) -> list:
    """Synthetic height maps (uniform random, nm-scale values)."""
    rng = np.random.default_rng(42)
    return [rng.uniform(0, 10, size=(hw, hw)).astype(np.float32) for _ in range(n)]


META = {"pixel_nm": (1.0, 1.0)}


# ---------------------------------------------------------------------------
# Per-method benchmark
# ---------------------------------------------------------------------------

def bench_method(method_fn, images: list, file_counts: list) -> dict:
    """
    Run method_fn on each image sequentially, record cumulative time.
    No warm-up needed (no GPU, no model load).
    """
    results = {"startup_total_ms": 0.0}  # Classical: no startup overhead

    per_file = {}
    cumulative_ms = 0.0
    max_n = max(file_counts)

    for i, img in enumerate(images[:max_n]):
        t0 = _now()
        method_fn(img, META)
        cumulative_ms += (_now() - t0) * 1000
        n = i + 1
        if n in file_counts:
            avg_ms = cumulative_ms / n
            per_file[n] = {
                "avg_infer_ms": round(avg_ms, 1),
                "effective_per_file_ms": round(avg_ms, 1),  # no startup to amortize
                "startup_fraction_pct": 0.0,
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
    print(f"    Startup overhead:  none (CPU-only, no model load)")
    print()
    print(f"    {'N files':>8}  {'Avg ms/file':>12}")
    print(f"    {'─'*8}  {'─'*12}")
    for n in file_counts:
        p = r["per_file_counts"][n]
        print(f"    {n:>8}  {p['avg_infer_ms']:>10.1f}ms")


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

    print("=" * 60)
    print("  Classical Segmentation Benchmark")
    print("=" * 60)
    print(f"\n  Host:   {platform.node()}")
    print(f"  Size:   {hw}x{hw}  |  N = {file_counts}")
    print(f"  Note:   No GPU used. No model-load overhead.")

    from qdseg.segmentation import segment_advanced, segment_watershed, segment_thresholding

    images = _make_images(hw, max(file_counts))

    all_results = {
        "system": {"hostname": platform.node(), "image_size": hw},
        "file_counts": file_counts,
    }

    methods = [
        ("advanced",      segment_advanced,      f"Advanced (Otsu+DT+DBSCAN+Voronoi)    — {hw}x{hw}"),
        ("watershed",     segment_watershed,     f"Watershed                              — {hw}x{hw}"),
        ("thresholding",  segment_thresholding,  f"Thresholding (Otsu)                    — {hw}x{hw}"),
    ]

    for key, fn, label in methods:
        try:
            r = bench_method(fn, images, file_counts)
            all_results[key] = r
            print_table(label, r, file_counts)
        except Exception as e:
            print(f"\n  {key} ERROR: {e}")
            all_results[key] = {"error": str(e)}

    print(f"\n{'='*60}\n  Done.\n{'='*60}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.json}")


if __name__ == "__main__":
    main()
