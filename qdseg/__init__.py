"""
QDSeg (Quantum Dot Segmentation) Package

A pure analysis library for quantum dot detection and analysis in AFM images.

Key Features:
- Pure analysis library (Stateless) - no database or file system state management
- Focused on single file/image processing
- Reusable general-purpose package

Module Structure:
- AFMData: XQD file loading, corrections, segmentation, and statistics
  - .segment()  — run segmentation (stores result internally, returns labels)
  - .stats()    — overall grain statistics dict
  - .grains()   — per-grain measurement list
  - .labels     — property: last segmentation result
- segmentation: segment() dispatcher + internal method functions
- statistics: Grain statistics calculation
- analyze: save_results(), _filter_small_labels(), _create_grain_analysis_pdf()
- utils: GPU utilities

GPU Acceleration (auto-detected based on environment):
- NVIDIA GPU: CUDA
- Apple Silicon: MPS (PyTorch) / Metal (TensorFlow)
- Other: CPU

Usage Example:
    >>> from qdseg import AFMData
    >>>
    >>> data = AFMData("path/to/file.xqd")
    >>> data.first_correction().second_correction().third_correction()
    >>> data.flat_correction("line_by_line").baseline_correction("min_to_zero")
    >>>
    >>> data.segment()                     # default: method='advanced'
    >>> stats = data.stats()
    >>> print(f"Found {stats['num_grains']} quantum dots")
    >>>
    >>> # Direct height/meta access (custom pipelines)
    >>> from qdseg import segment, calculate_grain_statistics
    >>> labels = segment(height, meta, method='watershed')
    >>> stats = calculate_grain_statistics(labels, height, meta)
"""

# Unified segmentation dispatcher
from .segmentation import segment

# Statistics functions
from .statistics import (
    calculate_grain_statistics,
    get_individual_grains,
)

# save_results
from .analyze import save_results

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
    StarDistConfig,
    StarDistTrainer,
    create_stardist_labels,
    CellposeConfig,
    CellposeTrainer,
    create_cellpose_labels,
)

__version__ = "0.4.1"

__all__ = [
    # Data wrapper (primary API)
    "AFMData",
    # Segmentation dispatcher (for custom pipelines)
    "segment",
    # Statistics
    "calculate_grain_statistics",
    "get_individual_grains",
    # Results I/O
    "save_results",
    # GPU utilities
    "setup_gpu_environment",
    "get_torch_device",
    "check_tensorflow_gpu",
    "print_gpu_info",
    # Training
    "StarDistConfig",
    "StarDistTrainer",
    "create_stardist_labels",
    "CellposeConfig",
    "CellposeTrainer",
    "create_cellpose_labels",
]
