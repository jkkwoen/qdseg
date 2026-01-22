# QDSeg

A Python package for Quantum Dot (QD) segmentation and analysis from AFM/XQD images (Version 0.2.4)

> ⚠️ **Python Version Requirement**: This package requires **Python 3.12**. TensorFlow (required for StarDist) does not yet support Python 3.13+.

## Design Philosophy

QDSeg is designed as a **pure analysis library** with the following characteristics:

| Characteristic | Description |
|----------------|-------------|
| **Stateless** | No database or file system state management |
| **Single-file focused** | Processes one image at a time |
| **Reusable** | Can be used as a dependency in other projects |

This design allows QDSeg to be easily integrated into larger workflows (e.g., batch processing systems, databases, web applications) while keeping the core analysis logic clean and testable.

```
┌─────────────────────────────────────────────────────────────┐
│                         qdseg                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ Data Load   │→ │ Corrections │→ │ Segmentation        │  │
│  │ (AFMData)   │  │             │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
│                                              ↓              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ PDF Report  │← │ Individual  │← │ Statistics          │  │
│  │             │  │ Grains      │  │                     │  │
│  └─────────────┘  └─────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Features

- **Multiple Segmentation Methods**
  - Rule-based: Otsu thresholding + Distance Transform + Voronoi
  - StarDist: Deep learning-based segmentation (optional)
  - CellPose: Deep learning-based segmentation (optional)
  - Cellulus: Custom deep learning model (optional)

- **Statistics Calculation**: Detailed statistics for each quantum dot (area, diameter, centroid, peak position, etc.)
- **High-level Analysis**: Single file or batch analysis
- **GPU Acceleration Support**: Automatic detection for NVIDIA CUDA, Apple Silicon MPS/Metal
- **Model Training**: Cellulus model training utilities

## Installation

### Basic Installation

```bash
pip install .
```

Or install directly from GitHub:

```bash
pip install git+https://github.com/jkkwoen/qdseg.git
```

### Development Installation

If you plan to modify the code:

```bash
# Create virtual environment with Python 3.12 (REQUIRED)
python3.12 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Verify Python version
python --version  # Should show Python 3.12.x

# Install in editable mode
pip install -e .
```

> 💡 **Why Python 3.12?**: TensorFlow (used by StarDist for deep learning segmentation) does not support Python 3.13 or 3.14. Python 3.12 is the latest fully supported version for all dependencies.

### Environment Variables

The package uses environment variables for flexible path configuration. Create a `.env` file in your project root:

```bash
# Copy the example file
cp .env.example .env

# Edit the .env file with your paths
# QDSEG_DATA_DIR=./tests/input_data/xqd
# QDSEG_OUTPUT_DIR=./tests/model_data
# QDSEG_MODEL_TYPE=cellulus
```

If `.env` file is not provided, the package will use default paths relative to the project root.

### Optional Dependencies

To use deep learning-based segmentation methods:

```bash
# For StarDist
pip install "qdseg[stardist]"

# For CellPose
pip install "qdseg[cellpose]"

# Install all optional dependencies
pip install "qdseg[all]"
```

## Usage

### Basic Example

```python
from pathlib import Path
from qdseg import analyze_single_file_with_grain_data

xqd_file = Path("path/to/your/file.xqd")
output_dir = Path("output")

success, individual_grain_data, grain_stats, pdf_path = analyze_single_file_with_grain_data(
    xqd_file, output_dir
)

if success:
    print(f"PDF saved: {pdf_path}")
    print(f"Number of quantum dots: {grain_stats['num_grains']}")
    print(f"First QD area: {individual_grain_data[0]['area_nm2']} nm²")
```

### Choosing Segmentation Method

```python
from qdseg import (
    segment_rule_based,
    segment_stardist,
    segment_cellpose,
    segment_cellulus,
    calculate_grain_statistics,
    get_individual_grains
)
from qdseg import AFMData

# Read XQD file and apply corrections
data = AFMData("path/to/file.xqd")

# Apply corrections (recommended order)
data.first_correction().second_correction().third_correction()
data.align_rows(method='median')  # Scan Line Artefacts correction (before flat correction)
data.flat_correction("line_by_line").baseline_correction("min_to_zero")

# Get corrected data
height = data.get_data()
meta = data.get_meta()

# Choose segmentation method
labels = segment_rule_based(height, meta)  # Default method
# labels = segment_stardist(height, meta)  # StarDist (deep learning)
# labels = segment_cellpose(height, meta)  # CellPose (deep learning)
# labels = segment_cellulus(height, meta)  # Cellulus (custom model)

# Calculate statistics
stats = calculate_grain_statistics(labels, height, meta)
individual_grains = get_individual_grains(labels, height, meta)
```

### Scan Line Artefacts Correction

```python
from qdseg import AFMData

data = AFMData("path/to/file.xqd")
data.first_correction().second_correction().third_correction()

# Apply Scan Line Artefacts correction (align rows)
# Available methods: 'median', 'mean', 'polynomial', 'median_difference', 'trimmed_mean'
data.align_rows(method='median')  # Basic method (recommended)
# data.align_rows(method='mean')  # Mean value subtraction
# data.align_rows(method='polynomial', poly_degree=1)  # Remove linear slopes
# data.align_rows(method='median_difference')  # Better preserves large features
# data.align_rows(method='trimmed_mean', trim_fraction=0.1)  # Robust against outliers

data.flat_correction("line_by_line").baseline_correction("min_to_zero")
```
```

### GPU Status Check

```python
from qdseg import print_gpu_info, setup_gpu_environment

# Setup GPU environment
setup_gpu_environment()

# Print GPU information
print_gpu_info()
```

### Model Training (Cellulus)

```python
from qdseg import CellulusTrainer, TrainingConfig, setup_environment

# Setup environment
setup_environment()

# Training configuration
config = TrainingConfig(
    data_dir="path/to/training/data",
    output_dir="path/to/output",
    # ... other settings
)

# Run training
trainer = CellulusTrainer(config)
trainer.train()
```

## Output

- **PDF File**: Plot containing original height data and grain mask overlay
- **Individual Grain Data**: Detailed information for each grain (area, diameter, centroid, peak position, etc.)
- **Grain Statistics**: Statistical information for all grains

## Dependencies

### Python Version

- **Python 3.12** (Required)
  - Python 3.13+ is NOT supported due to TensorFlow compatibility
  - Use `python3.12 -m venv .venv` to create a compatible environment

### Required Dependencies

- numpy>=1.24.0
- matplotlib>=3.7.0
- scipy>=1.10.0
- scikit-learn>=1.3.0
- scikit-image>=0.20.0
- python-dotenv>=1.0.0

### Optional Dependencies

- **StarDist**: `stardist>=0.8.0`, `tensorflow>=2.10.0`
- **CellPose**: `cellpose>=3.0.0`
- **Training**: `torch>=2.0.0`, `zarr>=2.16.0`, `tqdm>=4.65.0`

## GPU Support

The package automatically detects and supports various GPU environments:

- **NVIDIA GPU**: CUDA (PyTorch/TensorFlow)
- **Apple Silicon**: MPS (PyTorch) / Metal (TensorFlow)
- **Others**: Automatically falls back to CPU mode

## Project Structure

```
qdseg/
├── qdseg/
│   ├── __init__.py              # Package initialization and main exports
│   ├── afm_data_wrapper.py      # AFMData - XQD file loading and data access
│   ├── io.py                    # Low-level XQD file reading
│   ├── corrections.py           # Data corrections (flat, baseline, etc.)
│   ├── segmentation.py          # Segmentation algorithms
│   │                            # - segment_rule_based (Classical/Watershed)
│   │                            # - segment_stardist (TensorFlow)
│   │                            # - segment_cellpose (PyTorch)
│   │                            # - segment_cellulus (PyTorch, requires training)
│   ├── statistics.py            # Grain statistics calculation
│   ├── analyze.py               # High-level analysis API
│   ├── utils.py                 # GPU utilities (CUDA, MPS, Metal detection)
│   ├── grain_analysis.py        # (Deprecated) Legacy compatibility
│   ├── train_model.py           # Model training utilities
│   └── training/                # Training modules
│       ├── cellulus_trainer.py
│       ├── cellpose_trainer.py
│       └── stardist_trainer.py
├── requirements.txt
├── setup.py
└── README.md
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `AFMData` | XQD file loading and data access |
| `corrections.py` | Data correction (first/second/third, flat, baseline) |
| `segmentation.py` | Segmentation algorithms (Classical, StarDist, CellPose, Cellulus) |
| `statistics.py` | Grain statistics calculation (stats, individual grains) |
| `analyze.py` | High-level analysis API and PDF report generation |
| `utils.py` | GPU utilities and environment setup |

## Main API

### Segmentation

- `segment_rule_based(height, meta, **kwargs)`: Rule-based segmentation
- `segment_stardist(height, meta, **kwargs)`: StarDist segmentation
- `segment_cellpose(height, meta, **kwargs)`: CellPose segmentation
- `segment_cellulus(height, meta, **kwargs)`: Cellulus segmentation

### Statistics

- `calculate_grain_statistics(labels, height, meta)`: Calculate overall statistics
- `get_individual_grains(labels, height, meta)`: Extract individual grain data

### Analysis

- `analyze_grains(data, method='classical', **kwargs)`: Analyze grains
- `analyze_single_file_with_grain_data(xqd_file, output_dir, **kwargs)`: Analyze single file

### GPU Utilities

- `setup_gpu_environment()`: Setup GPU environment
- `get_torch_device()`: Get PyTorch device
- `check_tensorflow_gpu()`: Check TensorFlow GPU
- `print_gpu_info()`: Print GPU information

### Training

- `TrainingConfig`: Training configuration class
- `CellulusTrainer`: Cellulus model trainer
- `get_hardware_info()`: Get hardware information
- `print_hardware_info()`: Print hardware information
- `setup_environment()`: Setup training environment

## License

Please check the repository for license information.

## Contributing

Bug reports and feature suggestions are welcome through GitHub Issues.
