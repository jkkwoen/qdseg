# QDSeg

Quantum dot segmentation and grain statistics for AFM height images.

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19424802.svg)](https://doi.org/10.5281/zenodo.19424802)

QDSeg loads AFM height data, applies common AFM background corrections, separates quantum dots, and returns both summary statistics and per-grain measurements. It is designed as a Python package for reproducible analysis scripts rather than an interactive GUI.

![QDSeg demo: AFM height map and segmentation result](docs/demo.png)

## Features

- Load SII/Nanonavi XQD/XQF and Bruker/Veeco NanoScope height files.
- Apply plane, polynomial, row-alignment, flat, and baseline corrections.
- Segment quantum dots with thresholding, watershed, and an advanced rule-based pipeline.
- Optionally run StarDist or Cellpose when deep-learning backends are installed.
- Calculate area, diameter, height, volume, centroid, peak position, perimeter, and shape descriptors.
- Use CPU by default, with optional CUDA, CuPy/cuCIM, TensorFlow, PyTorch, or Apple Silicon acceleration where available.

## Installation

QDSeg currently targets Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/jkkwoen/qdseg.git
```

For local development:

```bash
git clone https://github.com/jkkwoen/qdseg.git
cd qdseg
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional deep-learning extras:

```bash
pip install "qdseg[stardist] @ git+https://github.com/jkkwoen/qdseg.git"
pip install "qdseg[cellpose] @ git+https://github.com/jkkwoen/qdseg.git"
pip install "qdseg[all] @ git+https://github.com/jkkwoen/qdseg.git"
```

Apple Silicon users can install the bundled Mac GPU extras:

```bash
pip install "qdseg[mac-gpu] @ git+https://github.com/jkkwoen/qdseg.git"
```

## Quick Start

```python
from qdseg import AFMData

data = AFMData("sample.xqd")

data.first_correction()
data.second_correction()
data.third_correction()
data.align_rows(method="median")
data.flat_correction("line_by_line")
data.baseline_correction("min_to_zero")

labels = data.segment(method="advanced")
stats = data.stats()
grains = data.grains()

print(f"Detected grains: {stats['num_grains']}")
print(f"Mean diameter: {stats['mean_diameter_nm']:.2f} nm")
print(f"Mean height: {stats['mean_height_nm']:.2f} nm")
```

`labels` is a 2D integer array with `0` as background and positive integers as grain IDs. The same array is also stored as `data.labels`.

## Working With Arrays

If you already have a height image and metadata, use the functional API:

```python
from qdseg import segment, calculate_grain_statistics, get_individual_grains

meta = {
    "pixel_nm": (2.0, 2.0),
    "scan_size_nm": (1024.0, 1024.0),
}

labels = segment(height_nm, meta, method="advanced")
stats = calculate_grain_statistics(labels, height_nm, meta)
grains = get_individual_grains(labels, height_nm, meta)
```

## Supported File Formats

| Format | Extensions | Notes |
| --- | --- | --- |
| SII/Nanonavi XQD/XQF | `.xqd`, `.xqf` | Reads native header metadata and raw height counts. |
| Bruker/Veeco NanoScope | `.spm`, `.000`, `.001`, `.002`, etc. | Reads common image channels from the text header and binary image block. |

NanoScope loading uses header fields such as `Data offset`, `Data length`, `Samps/line`, `Number of lines`, `Bytes/pixel`, and `Z scale`. The loader prefers a channel containing `Height` and falls back to the first image channel.

## Corrections

Corrections are chainable and modify the current in-memory height array:

```python
data = (
    AFMData("sample.000")
    .crop_px(x_min=100, x_max=400, y_min=80, y_max=360)
    .first_correction()
    .second_correction()
    .align_rows(method="median")
    .flat_correction("line_by_line")
    .baseline_correction("min_to_zero")
)
```

Available correction methods:

| Method | Purpose |
| --- | --- |
| `first_correction(method="polynomial")` | Remove first-order plane tilt. |
| `second_correction(method="polynomial")` | Remove second-order curved background. |
| `third_correction(method="polynomial")` | Remove third-order background curvature. |
| `align_rows(method="median")` | Reduce scan-line offsets. |
| `flat_correction("line_by_line")` | Remove residual line-wise or global background. |
| `baseline_correction("min_to_zero")` | Shift the height baseline. |
| `reset()` | Restore the originally loaded height image. |

`crop_px()` and `crop_nm()` can be used before correction or segmentation. Crop ranges are half-open: `x_min <= x < x_max` and `y_min <= y < y_max`.

## Segmentation

The main dispatcher is `segment(height, meta, method=...)`, and `AFMData.segment()` forwards to the same implementation.

| Method | Description | Main parameters |
| --- | --- | --- |
| `advanced` | Otsu threshold, distance transform, peak clustering, and Voronoi assignment. | `gaussian_sigma`, `min_area_px`, `min_peak_separation_nm`, `use_gpu` |
| `watershed` | Gaussian smoothing, Otsu mask, local maxima, and watershed. | `gaussian_sigma`, `min_distance`, `min_area_px`, `use_gpu` |
| `thresholding` | Global threshold and connected components, with optional distance separation. | `threshold_method`, `threshold_value`, `min_area_px`, `use_distance_separation`, `min_distance` |
| `stardist` | StarDist instance segmentation. | `model_name`, `model_path`, `prob_thresh`, `nms_thresh`, `use_gpu` |
| `cellpose` | Cellpose v4/Cellpose-SAM instance segmentation. | `diameter`, `flow_threshold`, `cellprob_threshold`, `model_path`, `gpu` |

Thresholding supports `otsu`, `isodata`, `li`, `triangle`, `yen`, `minimum`, and `manual`.

```python
labels = data.segment(method="thresholding", threshold_method="li")

labels = data.segment(
    method="thresholding",
    threshold_method="manual",
    threshold_value=8.0,
    use_distance_separation=True,
    min_distance=3,
)

labels = data.segment(method="watershed", min_distance=8)
```

Start with `advanced` for standard quantum-dot AFM images. Use `thresholding` when you need explicit threshold control, and `watershed` or distance separation when connected grains need to be split.

## Statistics

`data.stats()` and `calculate_grain_statistics()` return a dictionary with summary values:

| Key group | Examples |
| --- | --- |
| Counts and coverage | `num_grains`, `grain_density`, `area_fraction` |
| Area | `mean_area_px`, `mean_area_nm2`, `std_area_nm2`, `min_area_nm2`, `max_area_nm2` |
| Diameter and perimeter | `mean_diameter_nm`, `mean_diameter_px`, `mean_perimeter_nm`, `mean_perimeter_px` |
| Height and volume | `mean_height_nm`, `mean_height_peak_nm`, `mean_height_centroid_nm`, `mean_volume_nm3` |
| Shape | `mean_eccentricity`, `mean_solidity`, `mean_aspect_ratio`, `orientations_rad` |
| Per-grain arrays | `areas_nm2`, `diameters_nm`, `perimeters_nm`, `major_axis_nm`, `minor_axis_nm` |

`data.grains()` and `get_individual_grains()` return one dictionary per grain, including:

- `grain_id`
- `area_px`, `area_nm2`
- `diameter_px`, `diameter_nm`, `equivalent_radius_nm`
- `centroid_x_px`, `centroid_y_px`, `centroid_x_nm`, `centroid_y_nm`
- `peak_x_px`, `peak_y_px`, `peak_x_nm`, `peak_y_nm`
- `height_mean_nm`, `height_std_nm`, `height_peak_nm`, `height_centroid_nm`
- `peak_to_centroid_dist_nm`, `volume_nm3`
- `major_axis_nm`, `minor_axis_nm`, `perimeter_px`, `perimeter_nm`
- `eccentricity`, `solidity`, `aspect_ratio`, `orientation_deg`

## Saving Results

```python
from pathlib import Path
from qdseg import save_results

save_results(
    stats,
    grains,
    stats_path=Path("output/sample_stats.json"),
    grains_path=Path("output/sample_grains.csv"),
)
```

## GPU Notes

QDSeg automatically uses supported acceleration paths when their dependencies are present and falls back to CPU otherwise.

| Workload | Acceleration path |
| --- | --- |
| Advanced, watershed, thresholding | Optional CuPy/cuCIM GPU path |
| StarDist | TensorFlow GPU or Apple Metal when installed |
| Cellpose | PyTorch CUDA, MPS, or CPU |

Check the detected backend:

```python
from qdseg import print_gpu_info

print_gpu_info()
```

For NVIDIA Docker deployment notes, see [GPU_DOCKER.md](GPU_DOCKER.md).

## Development

```bash
source .venv/bin/activate
pip install -e .
python -m pytest tests/test_core.py -v
python tests/example_usage.py
python benchmark_classical.py
```

The test data directory may be kept outside version control for private or large AFM files. Synthetic tests in `tests/test_core.py` cover the public correction, segmentation, statistics, and wrapper behavior.

## Project Layout

```text
qdseg/
├── qdseg/
│   ├── __init__.py
│   ├── io.py
│   ├── nanoscope.py
│   ├── afm_data_wrapper.py
│   ├── corrections.py
│   ├── segmentation.py
│   ├── statistics.py
│   ├── analyze.py
│   └── utils.py
├── qdseg/training/
├── tests/
├── docs/demo.png
├── setup.py
└── README.md
```

## License

MIT. See [LICENSE](LICENSE).

Copyright (c) 2026 jkkwoen.
