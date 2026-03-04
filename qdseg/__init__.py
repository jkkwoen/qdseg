"""
QDSeg (Quantum Dot Segmentation) Package

A pure analysis library for quantum dot detection and analysis in AFM images.

Key Features:
- Pure analysis library (Stateless) - no database or file system state management
- Focused on single file/image processing
- Reusable general-purpose package

Module Structure:
- AFMData: XQD file loading and data access
- corrections: Data corrections (flat, baseline, etc.)
- segmentation: Segmentation algorithms
  - segment_rule_based: Otsu + Distance + DBSCAN + Voronoi (recommended)
  - segment_watershed: Watershed-based
  - segment_thresholding: Thresholding-based
  - segment_stardist: StarDist (TensorFlow)
  - segment_cellpose: CellPose (PyTorch)
  - segment_cellulus: Cellulus (PyTorch, requires training)
- statistics: Grain statistics calculation
- analyze: High-level analysis API
- utils: GPU utilities

GPU Acceleration (auto-detected based on environment):
- NVIDIA GPU: CUDA
- Apple Silicon: MPS (PyTorch) / Metal (TensorFlow)
- Other: CPU

Usage Example:
    >>> from qdseg import AFMData, segment_rule_based, calculate_grain_statistics
    >>>
    >>> # 1. Load data
    >>> data = AFMData("path/to/file.xqd")
    >>>
    >>> # 2. Apply corrections
    >>> data.first_correction().second_correction().third_correction()
    >>> data.align_rows(method='median')  # Scan line artefact correction (before flat)
    >>> data.flat_correction("line_by_line").baseline_correction("min_to_zero")
    >>>
    >>> # 3. Segmentation
    >>> labels = segment_rule_based(data.get_data(), data.get_meta())
    >>>
    >>> # 4. Calculate statistics
    >>> stats = calculate_grain_statistics(labels, data.get_data(), data.get_meta())
    >>> print(f"Found {stats['num_grains']} quantum dots")
"""

# Segmentation functions
from .segmentation import (
    segment_rule_based,
    segment_watershed,
    segment_thresholding,
    segment_stardist,
    segment_cellpose,
    segment_cellulus,
)

# Statistics functions
from .statistics import (
    calculate_grain_statistics,
    get_individual_grains,
)

# High-level analysis
from .analyze import (
    analyze_grains,
    analyze_single_file_with_grain_data,
)

# Data wrapper
from .afm_data_wrapper import AFMData

# GPU utilities
from .utils import (
    setup_gpu_environment,
    get_torch_device,
    check_tensorflow_gpu,
    print_gpu_info,
)

# Training utilities
from .training import (
    TrainingConfig,
    CellulusTrainer,
    load_afm_data,
    get_device,
    get_hardware_info,
    print_hardware_info,
    setup_environment,
)

__version__ = "0.3.2"

__all__ = [
    # Segmentation
    "segment_rule_based",
    "segment_watershed",
    "segment_thresholding",
    "segment_stardist",
    "segment_cellpose",
    "segment_cellulus",
    # Statistics
    "calculate_grain_statistics",
    "get_individual_grains",
    # Analysis
    "analyze_grains",
    "analyze_single_file_with_grain_data",
    # Data
    "AFMData",
    # GPU utilities
    "setup_gpu_environment",
    "get_torch_device",
    "check_tensorflow_gpu",
    "print_gpu_info",
    # Training
    "TrainingConfig",
    "CellulusTrainer",
    "load_afm_data",
    "get_device",
    "get_hardware_info",
    "print_hardware_info",
    "setup_environment",
]
