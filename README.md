# QDSeg

A Python package for Quantum Dot (QD) segmentation and analysis from AFM/XQD images (Version 0.2.0)

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
# Create virtual environment (optional)
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install in editable mode
pip install -e .
```

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
from qdseg.afm_data_wrapper import AFMData

# Read XQD file
data = AFMData.from_xqd("path/to/file.xqd")
height = data.height
meta = data.meta

# Choose segmentation method
labels = segment_rule_based(height, meta)  # Default method
# labels = segment_stardist(height, meta)  # StarDist (deep learning)
# labels = segment_cellpose(height, meta)  # CellPose (deep learning)
# labels = segment_cellulus(height, meta)  # Cellulus (custom model)

# Calculate statistics
stats = calculate_grain_statistics(labels, height, meta)
individual_grains = get_individual_grains(labels, height, meta)
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

### Required Dependencies

- Python 3.8+
- numpy>=1.24.0
- matplotlib>=3.7.0
- scipy>=1.10.0
- scikit-learn>=1.3.0
- scikit-image>=0.20.0

### Optional Dependencies

- **StarDist**: `stardist>=0.8.0`, `tensorflow>=2.10.0`
- **CellPose**: `cellpose>=3.0.0`

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
│   ├── io.py                    # XQD file reading
│   ├── corrections.py           # Correction functions
│   ├── grain_analysis.py        # Grain analysis functions
│   ├── segmentation.py          # Segmentation methods
│   ├── statistics.py            # Statistics calculation
│   ├── utils.py                 # Utility functions (GPU support included)
│   ├── afm_data_wrapper.py      # AFMData wrapper class
│   ├── analyze.py               # High-level analysis functions
│   └── train_model.py           # Model training utilities
├── requirements.txt
├── setup.py
└── README.md
```

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
