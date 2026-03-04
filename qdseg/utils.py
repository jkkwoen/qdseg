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
    """Print GPU environment information for PyTorch and TensorFlow.

    Examples
    --------
    >>> print_gpu_info()
    GPU Environment:
    --------------------------------------------------
    PyTorch version: 2.9.1
    PyTorch device: mps
    TensorFlow version: 2.20.0
    --------------------------------------------------
    """
    print("\nGPU Environment:")
    print("-" * 50)

    # PyTorch
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        device = get_torch_device(verbose=True)
        print(f"  PyTorch device: {device}")
    except ImportError:
        print("  PyTorch: not installed")

    print()

    # TensorFlow
    try:
        import tensorflow as tf
        print(f"  TensorFlow version: {tf.__version__}")
        check_tensorflow_gpu(verbose=True)
    except ImportError:
        print("  TensorFlow: not installed")

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
    grain_labels: np.ndarray,
) -> np.ndarray:
    """Apply line-by-line flat correction, excluding grain regions.

    .. deprecated::
        Use ``AFMData.flat_correction(method='line_by_line', mask=grain_mask)``
        instead, which is equivalent and avoids the row/column interference
        that the previous implementation produced.

    Parameters
    ----------
    height_data : np.ndarray
        Height data after slope correction.
    grain_mask : np.ndarray
        Boolean mask — ``True`` where grains are.
    grain_labels : np.ndarray
        Label image (not used; kept for backward-compatibility).

    Returns
    -------
    np.ndarray
        Flat-corrected height data.
    """
    import warnings
    warnings.warn(
        "apply_grain_excluded_flat_correction is deprecated. "
        "Use AFMData.flat_correction(method='line_by_line', mask=grain_mask) instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    corrector = AFMCorrections()
    corrector.set_flat_method("line_by_line")
    return corrector.correct_flat(height_data, mask=grain_mask.astype(int))

