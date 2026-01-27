"""
QDSeg (Quantum Dot Segmentation) Package

AFM 이미지에서 Quantum Dot(양자점) 검출 및 분석을 위한 순수 분석 라이브러리

핵심 특성:
- 순수 분석 라이브러리 (Stateless) - 데이터베이스나 파일 시스템 상태 관리 없음
- 단일 파일/이미지 처리에 집중
- 재사용 가능한 범용 패키지

모듈 구조:
- AFMData: XQD 파일 로드 및 데이터 접근
- corrections: 데이터 보정 (flat, baseline 등)
- segmentation: 세그멘테이션 알고리즘
  - segment_rule_based: Otsu + Distance + DBSCAN + Voronoi (권장)
  - segment_watershed: Watershed 기반
  - segment_thresholding: Thresholding 기반
  - segment_stardist: StarDist (TensorFlow)
  - segment_cellpose: CellPose (PyTorch)
  - segment_cellulus: Cellulus (PyTorch, 학습 필요)
- statistics: Grain 통계 계산
- analyze: 고수준 분석 API
- utils: GPU 유틸리티

GPU 가속 (환경에 따라 자동 감지):
- NVIDIA GPU: CUDA
- Apple Silicon: MPS (PyTorch) / Metal (TensorFlow)
- 그 외: CPU

사용 예시:
    >>> from qdseg import AFMData, segment_rule_based, calculate_grain_statistics
    >>>
    >>> # 1. 데이터 로드
    >>> data = AFMData("path/to/file.xqd")
    >>>
    >>> # 2. 보정 적용
    >>> data.first_correction().second_correction().third_correction()
    >>> data.align_rows(method='median')  # Scan Line Artefacts 보정 (flat 전)
    >>> data.flat_correction("line_by_line").baseline_correction("min_to_zero")
    >>>
    >>> # 3. 세그멘테이션
    >>> labels = segment_rule_based(data.get_data(), data.get_meta())
    >>>
    >>> # 4. 통계 계산
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

__version__ = "0.3.1"

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
