"""
Utility functions for grain analysis

GPU/Device 자동 감지 유틸리티 포함:
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)
- Metal (TensorFlow on Apple Silicon)
- CPU (fallback)
"""

import os
import warnings
import numpy as np
from typing import Tuple, Optional
from .corrections import AFMCorrections


# ============================================================
# GPU/Device 자동 감지 유틸리티
# ============================================================

def setup_gpu_environment():
    """
    GPU 환경 초기화 및 최적화 설정
    
    Apple Silicon MPS의 경우 메모리 설정 최적화
    프로그램 시작 시 한 번 호출 권장
    """
    # TensorFlow 로그 레벨 조정 (불필요한 경고 숨김)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
    
    # PyTorch MPS fallback 경고 숨김
    warnings.filterwarnings('ignore', message='.*MPS.*')
    warnings.filterwarnings('ignore', message='.*channels deprecated.*')


def get_torch_device(verbose: bool = True) -> "torch.device":
    """
    PyTorch용 최적 디바이스 자동 감지 (Cellpose용)
    
    우선순위: CUDA > MPS > CPU
    
    Parameters
    ----------
    verbose : bool
        디바이스 정보 출력 여부 (default: True)
    
    Returns
    -------
    torch.device
        사용할 디바이스
    
    Examples
    --------
    >>> device = get_torch_device()
    ✓ Apple Silicon MPS 가속 사용
    >>> device
    device(type='mps')
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch가 설치되지 않았습니다. 설치: pip install torch")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            device_name = torch.cuda.get_device_name(0)
            print(f"   ✓ CUDA GPU 감지: {device_name}")
        return device
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print("   ✓ Apple Silicon MPS 가속 사용")
        return device
    
    else:
        if verbose:
            print("   ⚠️ GPU를 찾을 수 없음, CPU 모드로 실행")
        return torch.device('cpu')


def check_tensorflow_gpu(verbose: bool = True) -> Tuple[bool, str]:
    """
    TensorFlow GPU 가속 상태 확인 (StarDist용)
    
    Parameters
    ----------
    verbose : bool
        상태 메시지 출력 여부 (default: True)
    
    Returns
    -------
    Tuple[bool, str]
        (GPU 사용 가능 여부, 설명 메시지)
    
    Examples
    --------
    >>> gpu_available, msg = check_tensorflow_gpu()
    >>> print(msg)
    ✓ TensorFlow Metal GPU 사용 (Apple Silicon)
    """
    try:
        import tensorflow as tf
    except ImportError:
        msg = "⚠️ TensorFlow가 설치되지 않음"
        if verbose:
            print(f"   {msg}")
        return False, msg
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        # GPU 메모리 증가 허용 (OOM 방지)
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        
        # Metal (Apple Silicon) vs CUDA 구분
        gpu_name = gpus[0].name if gpus else "Unknown"
        
        # Apple Silicon Metal 감지
        try:
            # tensorflow-metal이 설치되어 있으면 Metal GPU로 인식
            if any('GPU' in str(gpu) for gpu in gpus):
                # macOS에서 실행 중인지 확인
                import platform
                if platform.system() == 'Darwin' and platform.processor() == 'arm':
                    msg = "✓ TensorFlow Metal GPU 사용 (Apple Silicon)"
                else:
                    msg = f"✓ TensorFlow CUDA GPU 사용: {gpu_name}"
            else:
                msg = f"✓ TensorFlow GPU 사용: {gpu_name}"
        except Exception:
            msg = f"✓ TensorFlow GPU 사용: {gpu_name}"
        
        if verbose:
            print(f"   {msg}")
        return True, msg
    else:
        msg = "⚠️ TensorFlow GPU를 찾을 수 없음, CPU 모드로 실행"
        if verbose:
            print(f"   {msg}")
        return False, msg


def print_gpu_info():
    """
    GPU 환경 정보 출력
    
    PyTorch와 TensorFlow 모두의 GPU 상태를 확인하고 출력
    
    Examples
    --------
    >>> print_gpu_info()
    
    🖥️  GPU 환경 정보:
    --------------------------------------------------
       PyTorch 버전: 2.9.1
       ✓ Apple Silicon MPS 가속 사용
       PyTorch 디바이스: mps
       TensorFlow 버전: 2.20.0
       ✓ TensorFlow Metal GPU 사용 (Apple Silicon)
    --------------------------------------------------
    """
    print("\n🖥️  GPU 환경 정보:")
    print("-" * 50)
    
    # PyTorch
    try:
        import torch
        print(f"   PyTorch 버전: {torch.__version__}")
        device = get_torch_device(verbose=True)
        print(f"   PyTorch 디바이스: {device}")
    except ImportError:
        print("   PyTorch: 설치되지 않음")
    
    print()
    
    # TensorFlow
    try:
        import tensorflow as tf
        print(f"   TensorFlow 버전: {tf.__version__}")
        check_tensorflow_gpu(verbose=True)
    except ImportError:
        print("   TensorFlow: 설치되지 않음")
    
    print("-" * 50)


def nm2_to_px_area(area_nm2: float, pixel_nm: Tuple[float, float]) -> int:
    """Convert area in nm² to pixels
    
    Parameters
    ----------
    area_nm2 : float
        Area in nm²
    pixel_nm : Tuple[float, float]
        Pixel size in nm (x, y)
    
    Returns
    -------
    int
        Area in pixels
    """
    px_area_nm2 = float(pixel_nm[0]) * float(pixel_nm[1])
    if px_area_nm2 <= 0:
        return int(max(1.0, area_nm2))
    return max(1, int(round(area_nm2 / px_area_nm2)))


def apply_grain_excluded_flat_correction(
    height_data: np.ndarray, 
    grain_mask: np.ndarray,
    grain_labels: np.ndarray
) -> np.ndarray:
    """
    Grain 영역을 제외하고 flat 보정을 적용
    
    Parameters
    ----------
    height_data : np.ndarray
        보정할 높이 데이터 (after_slope_correction)
    grain_mask : np.ndarray
        Grain 영역을 나타내는 마스크 (boolean)
    grain_labels : np.ndarray
        Grain 라벨링된 데이터
        
    Returns
    -------
    np.ndarray
        Grain 영역 제외 flat 보정된 데이터
    """
    height_corrected = height_data.copy()
    
    # Grain 영역이 아닌 배경 영역만 사용
    background_mask = ~grain_mask
    
    if np.sum(background_mask) < height_data.size * 0.1:  # 배경이 너무 작으면
        print("⚠️  Warning: Background area too small, using standard flat correction")
        corrector = AFMCorrections()
        corrector.set_flat_method("line_by_line")
        return corrector.correct_flat(height_corrected)
    
    # 외곽선 경계에서 기준값 계산 (grain 제외)
    boundary_mask = np.zeros_like(grain_mask)
    boundary_width = max(2, min(height_data.shape) // 50)
    
    # 상하좌우 외곽선 생성
    boundary_mask[:boundary_width, :] = True  # 상단
    boundary_mask[-boundary_width:, :] = True  # 하단
    boundary_mask[:, :boundary_width] = True  # 좌측
    boundary_mask[:, -boundary_width:] = True  # 우측
    
    # Grain 제외된 외곽선만 사용
    reference_mask = background_mask & boundary_mask
    
    if np.sum(reference_mask) > 0:
        reference_level = np.mean(height_data[reference_mask])
        height_corrected = height_data - reference_level
    else:
        # 외곽선에서 참조를 구할 수 없으면 전체 배경 사용
        reference_level = np.mean(height_data[background_mask])
        height_corrected = height_data - reference_level
    
    # 라인별 최종 평면 보정 (배경 영역만 적용)
    for i in range(height_data.shape[0]):
        row_mask = background_mask[i, :]
        if np.sum(row_mask) > 0:
            row_avg = np.mean(height_corrected[i, row_mask])
            if not np.isnan(row_avg):
                height_corrected[i, :] -= row_avg
    
    for j in range(height_data.shape[1]):
        col_mask = background_mask[:, j]
        if np.sum(col_mask) > 0:
            col_avg = np.mean(height_corrected[:, col_mask])
            if not np.isnan(col_avg):
                height_corrected[:, j] -= col_avg
    
    return height_corrected

