# QDSeg

**Quantum Dot Segmentation and Analysis for AFM/XQD images**
Version 0.3.2 | Python 3.12 | MIT License

> Contact: [jk.kwoen@gmail.com](mailto:jk.kwoen@gmail.com) | [GitHub](https://github.com/jkkwoen/qdseg)

---

## Overview

QDSeg is a **pure analysis library** for detecting and measuring quantum dots (QDs) in atomic force microscopy (AFM) images stored as XQD files.

| Property | Description |
|----------|-------------|
| Stateless | No database or filesystem state |
| Single-image | Processes one image at a time |
| Reusable | Designed as a dependency for larger pipelines |

```
AFMData (load XQD)
    → Corrections (tilt / scan-line / flat / baseline)
        → Segmentation (rule-based / StarDist / Cellpose / Cellulus)
            → Statistics (per-grain + aggregate)
```

---

## Installation

### Minimal (rule-based only)

```bash
pip install git+https://github.com/jkkwoen/qdseg.git
```

### With deep-learning backends

```bash
# StarDist (TensorFlow)
pip install "git+https://github.com/jkkwoen/qdseg.git#egg=qdseg[stardist]"

# Cellpose
pip install "git+https://github.com/jkkwoen/qdseg.git#egg=qdseg[cellpose]"

# Everything
pip install "git+https://github.com/jkkwoen/qdseg.git#egg=qdseg[all]"
```

### Development

```bash
git clone https://github.com/jkkwoen/qdseg.git
cd qdseg
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

> **Python 3.12 is required.** TensorFlow (used by StarDist) does not yet support Python 3.13+.

---

## Quick start

```python
from qdseg import AFMData, segment_rule_based, calculate_grain_statistics

# 1. Load and correct
data = AFMData("path/to/file.xqd")
data.first_correction().second_correction().third_correction()
data.align_rows(method='median')
data.flat_correction("line_by_line").baseline_correction("min_to_zero")

# 2. Segment
height = data.get_data()
meta   = data.get_meta()
labels = segment_rule_based(height, meta)

# 3. Statistics
stats  = calculate_grain_statistics(labels, height, meta)
print(f"QDs detected : {stats['num_grains']}")
print(f"Mean diameter: {stats['mean_diameter_nm']:.1f} nm")
```

### Reset corrections

```python
data = AFMData("file.xqd")
data.first_correction().second_correction()
data.reset()          # discard all corrections
data.first_correction()   # start over
```

### High-level single-file analysis

```python
from pathlib import Path
from qdseg import analyze_single_file_with_grain_data

success, grains, stats, pdf_path = analyze_single_file_with_grain_data(
    xqd_file=Path("file.xqd"),
    output_dir=Path("output"),
    method="cellpose",   # or "rule_based", "stardist", "cellulus"
)
print(f"Saved PDF: {pdf_path}")
```

---

## Segmentation methods

| Method | Description | Extra install |
|--------|-------------|---------------|
| `rule_based` | Otsu → distance transform → DBSCAN peaks → Voronoi | — |
| `watershed` | Local maxima → Watershed on Sobel gradient | — |
| `thresholding` | Simple threshold + connected components | — |
| `stardist` | Star-convex polygon DL (pre-trained `2D_versatile_fluo`) | `[stardist]` |
| `cellpose` | Gradient-flow DL (Cellpose-SAM `cpsam`, v4+) | `[cellpose]` |
| `cellulus` | Unsupervised embedding DL (requires custom training) | `[cellulus]` |

---

## AFM corrections

Corrections are applied in the following recommended order:

```python
data.first_correction()       # remove planar tilt (1st-order)
data.second_correction()      # remove quadratic background
data.third_correction()       # remove cubic background
data.align_rows(method='median')  # remove scan-line artefacts
data.flat_correction("line_by_line")
data.baseline_correction("min_to_zero")
```

`first_correction` / `second_correction` / `third_correction` accept
`method='polynomial'` (full 2-D fit via `numpy.linalg.lstsq`, default) or
`method='simple'` (separable row + column fit, faster).

---

## Statistics returned

`calculate_grain_statistics` returns a dict with:

| Key | Description |
|-----|-------------|
| `num_grains` | Total grain count |
| `mean_diameter_nm` / `std_diameter_nm` | Diameter (nm) |
| `mean_area_nm2` / `std_area_nm2` | Area (nm²) |
| `mean_height_nm` / `std_height_nm` | Mean height per grain |
| `mean_height_peak_nm` | Peak height per grain |
| `mean_volume_nm3` | Volume (nm³) |
| `mean_eccentricity` / `mean_solidity` / `mean_aspect_ratio` | Shape |
| `grain_density` / `area_fraction` | Coverage |
| `orientations_rad` | Per-grain orientation (radians) |
| `areas_nm2`, `diameters_nm`, … | Per-grain arrays |

---

## GPU support

Automatically detected:

| Hardware | Backend |
|----------|---------|
| NVIDIA GPU | CUDA (PyTorch / TensorFlow) |
| Apple Silicon | MPS (PyTorch) / Metal (TensorFlow) |
| CPU | Fallback |

```python
from qdseg import print_gpu_info
print_gpu_info()
```

---

## Dependencies

### Required

- `numpy >= 1.24`
- `scipy >= 1.10`
- `scikit-image >= 0.20`
- `scikit-learn >= 1.3` (DBSCAN in rule-based method)
- `matplotlib >= 3.7`

### Optional

| Extra | Packages |
|-------|----------|
| `[stardist]` | `stardist >= 0.8`, `tensorflow >= 2.10` |
| `[cellpose]` | `cellpose >= 4.0` |
| `[cellulus]` | `torch >= 2.0`, `zarr >= 2.16`, `tqdm >= 4.65` |
| `[training]` | above + `python-dotenv >= 1.0` |
| `[all]` | all of the above |

---

## Project structure

```
qdseg/
├── qdseg/
│   ├── __init__.py          # public API exports
│   ├── io.py                # XQD file reader
│   ├── afm_data_wrapper.py  # AFMData class
│   ├── corrections.py       # AFMCorrections (numpy-only, no sklearn)
│   ├── segmentation.py      # all segmentation algorithms
│   ├── statistics.py        # grain statistics
│   ├── analyze.py           # high-level pipeline
│   ├── utils.py             # GPU detection utilities
│   └── training/            # Cellulus / Cellpose / StarDist trainers
├── tests/
│   ├── test_core.py         # unit tests (no XQD files needed)
│   └── example_usage.py
├── setup.py
└── README.md
```

---

## License

MIT — see [LICENSE](LICENSE).

Copyright (c) 2026 jkkwoen
Contact: [jk.kwoen@gmail.com](mailto:jk.kwoen@gmail.com)
