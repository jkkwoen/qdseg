# GPU Docker Deployment Guide

qdseg GPU Docker images for NVIDIA GPUs. Two images are maintained for different
GPU architectures.

---

## Architecture overview

| Image | Target GPU | Compute | CUDA | Base |
|-------|-----------|---------|------|------|
| `qdseg-gpu-blackwell` | RTX PRO 6000, RTX 5090 (sm_120) | Blackwell | 12.8 | NGC TF 25.02 |
| `qdseg-gpu-turing` | RTX 2080 Ti (sm_75) | Turing | 12.2 | nvidia/cuda runtime |

Both images contain:
- PyTorch (CellPose cpsam model pre-baked)
- TensorFlow (StarDist 2D_versatile_fluo model pre-baked)
- CuPy + cuCIM (GPU-accelerated classical segmentation)
- qdseg installed from source

---

## Build

### Blackwell (RTX PRO 6000 / RTX 5090, sm_120)

```bash
docker build -f Dockerfile.gpu -t qdseg-gpu-blackwell .
```

### Turing (RTX 2080 Ti, sm_75)

```bash
docker build -f Dockerfile.gpu.turing -t qdseg-gpu-turing .
```

Build time: ~10–20 min (downloads PyTorch, TF, model weights).

**Model cache invalidation**: bump `MODEL_CACHE_DATE` to force re-download:

```bash
docker build --build-arg MODEL_CACHE_DATE=$(date +%Y-%m-%d) \
    -f Dockerfile.gpu.turing -t qdseg-gpu-turing .
```

---

## Run

```bash
# Blackwell
docker run --rm --gpus all \
    -v /path/to/xqd:/data/xqd \
    qdseg-gpu-blackwell \
    python3 /app/qdseg_src/benchmark_full_pipeline.py \
    --data-dir /data/xqd --max-files 50

# Turing
docker run --rm --gpus all \
    -v /path/to/xqd:/data/xqd \
    qdseg-gpu-turing \
    python3 /app/qdseg_src/benchmark_full_pipeline.py \
    --data-dir /data/xqd --max-files 50
```

---

## Benchmark results (2026-03-05, 512×512 XQD files)

Median per-file times (excludes first-call JIT/model-load warm-up).

| Method | Blackwell (RTX PRO 6000) | Turing (RTX 2080 Ti) |
|--------|--------------------------|----------------------|
| Corrections (flat median GPU) | ~112 ms | ~600 ms |
| Classical rule_based (GPU) | ~8 ms | ~15 ms |
| StarDist 2D_versatile_fluo | ~111 ms | ~150 ms |
| CellPose cpsam | ~161 ms | ~1 140 ms |

**First-call warm-up** (model load + CUDA JIT, included in mean but not median):

| | Blackwell | Turing |
|---|---|---|
| StarDist 1st call | ~1.8 s | ~4.5 s |
| CellPose 1st call | ~1.9 s | ~3.7 s |
| CuPy corrections 1st call | included in 1st file | included in 1st file |

Notes:
- Corrections mean is higher than median due to CuPy JIT on 1st file (~2 s on Turing).
- Classical mean is similarly elevated (CuPy JIT on 1st file ~10 s on Turing).
- Blackwell StarDist is GPU-accelerated (TF 2.17 + NGC 25.02 supports sm_120).
- CellPose cpsam on Turing is slower because sm_75 has less FP32 throughput than sm_120.

---

## Configuration

### `QDSEG_TF_MEMORY_MB` (Turing only)

TensorFlow's BFC allocator retains allocated VRAM and can prevent CellPose (PyTorch)
from loading the cpsam model on GPUs with limited VRAM (≤ 16 GB).

Default in `Dockerfile.gpu.turing`: **5120 MB** (5 GB).

| GPU VRAM | Recommended value |
|----------|------------------|
| 11 GB (RTX 2080 Ti) | 5120 |
| 16 GB (RTX 3080/4080) | 6144 |
| 24 GB+ | 0 (unlimited, or omit) |

Override at runtime:

```bash
docker run --rm --gpus all -e QDSEG_TF_MEMORY_MB=6144 \
    -v /path/to/xqd:/data/xqd qdseg-gpu-turing python3 ...
```

The Blackwell image does **not** set `QDSEG_TF_MEMORY_MB` (96 GB VRAM — no limit needed).

---

## Architecture-specific notes

### Blackwell (sm_120, `Dockerfile.gpu`)

- Base: `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` — the only public TF image with
  working Blackwell PTX compilation as of 2026-03.
- CuPy and StarDist both work natively; no extra CUDA header patching needed.
- CellPose uses **cpsam** (4.x default, ~1.15 GB).

### Turing (sm_75, `Dockerfile.gpu.turing`)

- Base: `nvidia/cuda:12.2.2-cudnn8-runtime` (leaner, no full CUDA dev kit).
- **CuPy JIT header fix** (Step 10): `nvidia-cuda-nvcc-cu12` provides `nvcc` but
  only ships 3 header files. `nvidia-cuda-runtime-cu12` (pulled in by CuPy) has
  the full runtime headers (`vector_types.h` etc.). They are merged into
  `cuda_nvcc/include/` at build time so `CUDA_PATH` points to a complete set:
  ```dockerfile
  RUN cp -rn .../cuda_runtime/include/* .../cuda_nvcc/include/
  ```
- **TF XLA ptxas path** (ENV): TF 2.17 on a runtime-only image cannot find the
  system `ptxas`. `XLA_FLAGS` and `PATH` point to the pip-installed nvcc binary.
- **Legacy Docker builder**: Turing server's Docker daemon does not support BuildKit
  heredoc syntax (`RUN python3 - <<'EOF'`). Model download steps use single-line
  `-c` strings instead.
- **`HEREDOC` vs `-c`**: If you rebuild on a BuildKit-capable host, heredoc is fine;
  on legacy builder always use single-line `-c`.
- CellPose uses **cpsam** (4.x, ~1.15 GB) — same model as Blackwell.

---

## Troubleshooting

### `vector_types.h: No such file or directory` (CuPy JIT)

CuPy fails to JIT-compile CUDA kernels; flat correction falls back to
`scipy.ndimage.median_filter` (~3.5 s/file instead of ~0.6 s/file).

**Cause**: `CUDA_PATH` points to `cuda_nvcc/` which lacks runtime headers.

**Fix**: already applied in `Dockerfile.gpu.turing` Step 10.

If you see this in a custom image, add:
```dockerfile
RUN cp -rn /usr/local/lib/python3.12/dist-packages/nvidia/cuda_runtime/include/* \
           /usr/local/lib/python3.12/dist-packages/nvidia/cuda_nvcc/include/
```

### CellPose `CUDA out of memory` (Turing)

Occurs when StarDist (TF) runs before CellPose (PyTorch) in the same process.
TF's BFC allocator fragments VRAM, leaving no contiguous block for cpsam.

**Fix**: set `QDSEG_TF_MEMORY_MB=5120` (already in `Dockerfile.gpu.turing`).

### StarDist warm-up 4–5 s (Turing)

TF XLA compiles the graph on first inference. This is a one-time cost per container
invocation and is not included in the median benchmark figure.

### `no seeds found in get_masks_torch` (CellPose)

CellPose finds no cells in the image. Normal for AFM grain images with unusual
contrast — not an error. The benchmark still measures inference time correctly.

### `._filename.xqd` files skipped

macOS resource-fork files (Apple Double format) created when copying data from macOS
to Linux. They sort before real files alphabetically.

`benchmark_full_pipeline.py` skips them automatically (wrong file size).
Use `--max-files N` large enough to load past these hidden files:

```bash
# Check how many ._ files are present:
ls -a /path/to/xqd/*.xqd | grep '^\._' | wc -l
# Then set --max-files = (count of ._ files) + desired real files
python3 benchmark_full_pipeline.py --max-files 20 ...
```

---

## Image contents summary

| Component | Blackwell | Turing |
|-----------|-----------|--------|
| Python | 3.12 (NGC) | 3.12 (deadsnakes PPA) |
| CUDA | 12.8 | 12.2 |
| PyTorch | NGC bundled | 2.3.1+cu121 |
| TensorFlow | 2.17 (NGC) | 2.17.0 (pip) |
| cellpose | 4.0.8 | 4.0.8 |
| stardist | 0.9.2 | 0.9.2 |
| CuPy | 13.3.0 | 13.3.0 |
| cuCIM | 24.10.0 | 24.10.0 |
| CellPose model | cpsam (~1.15 GB) | cpsam (~1.15 GB) |
| StarDist model | 2D_versatile_fluo (~50 MB) | 2D_versatile_fluo (~50 MB) |
