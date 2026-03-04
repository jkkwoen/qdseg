#!/usr/bin/env python3
"""
GPU Benchmark: Compute primitives + StarDist + Cellpose inference.
Self-contained script — does not require qdseg package.

Usage:
    python benchmark_gpu.py [--json output.json]
"""

import json
import platform
import statistics
import sys
import time


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sync_device(framework: str, device=None):
    """Ensure GPU operations are complete before timing."""
    if framework == "torch":
        import torch
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize()
        elif device is not None and device.type == "mps":
            torch.mps.synchronize()
    elif framework == "tf":
        import tensorflow as tf
        # TF ops are synchronous by default in eager mode after .numpy()
        pass


def _bench(fn, warmup: int = 1, repeats: int = 5, framework: str = "torch", device=None):
    """Run *fn* with warmup, return list of elapsed seconds."""
    for _ in range(warmup):
        fn()
        _sync_device(framework, device)

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        _sync_device(framework, device)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return times


def _fmt(times):
    """Return median ± std as string (ms)."""
    med = statistics.median(times) * 1000
    if len(times) > 1:
        sd = statistics.stdev(times) * 1000
        return f"{med:.1f} ± {sd:.1f} ms"
    return f"{med:.1f} ms"


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------

def get_system_info() -> dict:
    info = {
        "hostname": platform.node(),
        "arch": platform.machine(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }

    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["torch_device"] = torch.cuda.get_device_name(0)
            info["torch_device_type"] = "cuda"
            info["vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            info["torch_device"] = "Apple MPS"
            info["torch_device_type"] = "mps"
        else:
            info["torch_device"] = "CPU"
            info["torch_device_type"] = "cpu"
    except ImportError:
        info["torch_version"] = "N/A"

    # TensorFlow
    try:
        import tensorflow as tf
        info["tf_version"] = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError:
                    pass
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                info["tf_device"] = "Apple Metal"
            else:
                info["tf_device"] = gpus[0].name
        else:
            info["tf_device"] = "CPU"
    except ImportError:
        info["tf_version"] = "N/A"
    except Exception as e:
        info["tf_version"] = f"error: {e}"

    return info


# ---------------------------------------------------------------------------
# PyTorch benchmarks
# ---------------------------------------------------------------------------

def bench_torch(device_type: str) -> dict:
    import torch

    if device_type == "cuda":
        dev = torch.device("cuda")
    elif device_type == "mps":
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    results = {}

    # Matmul
    for N in [1024, 4096, 8192]:
        A = torch.randn(N, N, device=dev, dtype=torch.float32)
        B = torch.randn(N, N, device=dev, dtype=torch.float32)
        times = _bench(lambda: A @ B, framework="torch", device=dev)
        results[f"matmul_{N}"] = {"median_ms": round(statistics.median(times) * 1000, 2), "raw": _fmt(times)}
        del A, B

    # Conv2d
    import torch.nn.functional as F
    for batch, ch, hw in [(16, 64, 128), (4, 128, 256)]:
        x = torch.randn(batch, ch, hw, hw, device=dev, dtype=torch.float32)
        w = torch.randn(ch, ch, 3, 3, device=dev, dtype=torch.float32)
        times = _bench(lambda: F.conv2d(x, w, padding=1), framework="torch", device=dev)
        results[f"conv2d_{batch}x{ch}x{hw}"] = {"median_ms": round(statistics.median(times) * 1000, 2), "raw": _fmt(times)}
        del x, w

    return results


# ---------------------------------------------------------------------------
# TensorFlow benchmarks
# ---------------------------------------------------------------------------

def bench_tf() -> dict:
    import tensorflow as tf

    results = {}

    # Matmul
    for N in [1024, 4096, 8192]:
        A = tf.random.normal([N, N])
        B = tf.random.normal([N, N])
        times = _bench(lambda: tf.matmul(A, B).numpy(), framework="tf")
        results[f"matmul_{N}"] = {"median_ms": round(statistics.median(times) * 1000, 2), "raw": _fmt(times)}
        del A, B

    # Conv2d
    for batch, ch, hw in [(16, 64, 128), (4, 128, 256)]:
        x = tf.random.normal([batch, hw, hw, ch])
        w = tf.random.normal([3, 3, ch, ch])
        times = _bench(lambda: tf.nn.conv2d(x, w, strides=1, padding="SAME").numpy(), framework="tf")
        results[f"conv2d_{batch}x{ch}x{hw}"] = {"median_ms": round(statistics.median(times) * 1000, 2), "raw": _fmt(times)}
        del x, w

    return results


# ---------------------------------------------------------------------------
# StarDist benchmark
# ---------------------------------------------------------------------------

def bench_stardist() -> dict:
    import numpy as np
    try:
        from stardist.models import StarDist2D
    except ImportError:
        return {"error": "stardist not installed"}

    results = {}
    model = StarDist2D.from_pretrained("2D_versatile_fluo")

    for hw in [256, 512]:
        img = np.random.rand(hw, hw).astype(np.float32)
        times = _bench(
            lambda: model.predict_instances(img, prob_thresh=0.5, nms_thresh=0.4),
            warmup=1, repeats=5, framework="tf"
        )
        results[f"stardist_{hw}x{hw}"] = {"median_ms": round(statistics.median(times) * 1000, 2), "raw": _fmt(times)}

    return results


# ---------------------------------------------------------------------------
# Cellpose benchmark
# ---------------------------------------------------------------------------

def bench_cellpose(device_type: str) -> dict:
    import numpy as np
    try:
        from cellpose import models
    except ImportError:
        return {"error": "cellpose not installed"}

    import torch
    if device_type == "cuda":
        dev = torch.device("cuda")
    elif device_type == "mps":
        dev = torch.device("mps")
    else:
        dev = torch.device("cpu")

    results = {}
    model = models.CellposeModel(gpu=(dev.type != "cpu"), device=dev)

    for hw in [256, 512]:
        img = np.random.rand(hw, hw).astype(np.float32) * 255
        times = _bench(
            lambda: model.eval(img, diameter=30, flow_threshold=0.4, cellprob_threshold=0.0),
            warmup=1, repeats=5, framework="torch", device=dev
        )
        results[f"cellpose_{hw}x{hw}"] = {"median_ms": round(statistics.median(times) * 1000, 2), "raw": _fmt(times)}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", help="Save results to JSON file")
    args = parser.parse_args()

    print("=" * 60)
    print("  GPU Benchmark — qdseg segmentation workloads")
    print("=" * 60)

    info = get_system_info()
    print(f"\n  Host:       {info['hostname']}")
    print(f"  Arch:       {info['arch']}")
    print(f"  OS:         {info['os']}")
    print(f"  Python:     {info['python']}")
    print(f"  PyTorch:    {info.get('torch_version', 'N/A')}  [{info.get('torch_device', 'N/A')}]")
    print(f"  TensorFlow: {info.get('tf_version', 'N/A')}  [{info.get('tf_device', 'N/A')}]")

    all_results = {"system": info}
    device_type = info.get("torch_device_type", "cpu")

    # --- PyTorch compute ---
    print(f"\n{'─' * 60}")
    print("  PyTorch compute primitives")
    print(f"{'─' * 60}")
    try:
        r = bench_torch(device_type)
        all_results["torch_compute"] = r
        for k, v in r.items():
            print(f"    {k:30s}  {v['raw']}")
    except Exception as e:
        print(f"    ERROR: {e}")
        all_results["torch_compute"] = {"error": str(e)}

    # --- TF compute ---
    print(f"\n{'─' * 60}")
    print("  TensorFlow compute primitives")
    print(f"{'─' * 60}")
    try:
        r = bench_tf()
        all_results["tf_compute"] = r
        for k, v in r.items():
            print(f"    {k:30s}  {v['raw']}")
    except Exception as e:
        print(f"    ERROR: {e}")
        all_results["tf_compute"] = {"error": str(e)}

    # --- StarDist ---
    print(f"\n{'─' * 60}")
    print("  StarDist inference (TensorFlow)")
    print(f"{'─' * 60}")
    try:
        r = bench_stardist()
        all_results["stardist"] = r
        for k, v in r.items():
            print(f"    {k:30s}  {v.get('raw', v.get('error', ''))}")
    except Exception as e:
        print(f"    ERROR: {e}")
        all_results["stardist"] = {"error": str(e)}

    # --- Cellpose ---
    print(f"\n{'─' * 60}")
    print("  Cellpose inference (PyTorch)")
    print(f"{'─' * 60}")
    try:
        r = bench_cellpose(device_type)
        all_results["cellpose"] = r
        for k, v in r.items():
            print(f"    {k:30s}  {v.get('raw', v.get('error', ''))}")
    except Exception as e:
        print(f"    ERROR: {e}")
        all_results["cellpose"] = {"error": str(e)}

    print(f"\n{'=' * 60}")
    print("  Benchmark complete.")
    print(f"{'=' * 60}")

    # JSON output
    json_str = json.dumps(all_results, indent=2, ensure_ascii=False)
    if args.json:
        with open(args.json, "w") as f:
            f.write(json_str)
        print(f"\n  Results saved to {args.json}")
    else:
        print(f"\n--- JSON ---\n{json_str}")


if __name__ == "__main__":
    main()
