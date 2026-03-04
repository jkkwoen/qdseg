"""
Utility functions for grain analysis

Includes GPU/Device auto-detection utilities:
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
# GPU/Device auto-detection utilities
# ============================================================

def setup_gpu_environment():
    """
    Initialize and optimize GPU environment settings.

    Optimizes memory settings for Apple Silicon MPS.
    Recommended to call once at program startup.
    """
    # Adjust TensorFlow log level (suppress unnecessary warnings)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

    # Suppress PyTorch MPS fallback warnings
    warnings.filterwarnings('ignore', message='.*MPS.*')
    warnings.filterwarnings('ignore', message='.*channels deprecated.*')


def get_torch_device(verbose: bool = True) -> "torch.device":
    """
    Auto-detect the optimal device for PyTorch (for Cellpose).

    Priority: CUDA > MPS > CPU

    Parameters
    ----------
    verbose : bool
        Whether to print device information (default: True).

    Returns
    -------
    torch.device
        The device to use.

    Examples
    --------
    >>> device = get_torch_device()
    ✓ Using Apple Silicon MPS acceleration
    >>> device
    device(type='mps')
    """
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. Install with: pip install torch")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        if verbose:
            device_name = torch.cuda.get_device_name(0)
            print(f"   ✓ CUDA GPU detected: {device_name}")
        return device
    
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        if verbose:
            print("   ✓ Using Apple Silicon MPS acceleration")
        return device
    
    else:
        if verbose:
            print("   ⚠️ No GPU found, running in CPU mode")
        return torch.device('cpu')


def check_tensorflow_gpu(verbose: bool = True) -> Tuple[bool, str]:
    """
    Check TensorFlow GPU acceleration status (for StarDist).

    Parameters
    ----------
    verbose : bool
        Whether to print status messages (default: True).

    Returns
    -------
    Tuple[bool, str]
        (whether GPU is available, description message).

    Examples
    --------
    >>> gpu_available, msg = check_tensorflow_gpu()
    >>> print(msg)
    ✓ Using TensorFlow Metal GPU (Apple Silicon)
    """
    try:
        import tensorflow as tf
    except ImportError:
        msg = "⚠️ TensorFlow is not installed"
        if verbose:
            print(f"   {msg}")
        return False, msg
    
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        # Allow GPU memory growth (prevent OOM)
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass
        
        # Distinguish between Metal (Apple Silicon) and CUDA
        gpu_name = gpus[0].name if gpus else "Unknown"
        
        # Detect Apple Silicon Metal
        try:
            # Recognized as Metal GPU if tensorflow-metal is installed
            if any('GPU' in str(gpu) for gpu in gpus):
                # Check if running on macOS
                import platform
                if platform.system() == 'Darwin' and platform.processor() == 'arm':
                    msg = "✓ Using TensorFlow Metal GPU (Apple Silicon)"
                else:
                    msg = f"✓ Using TensorFlow CUDA GPU: {gpu_name}"
            else:
                msg = f"✓ Using TensorFlow GPU: {gpu_name}"
        except Exception:
            msg = f"✓ Using TensorFlow GPU: {gpu_name}"

        if verbose:
            print(f"   {msg}")
        return True, msg
    else:
        msg = "⚠️ No TensorFlow GPU found, running in CPU mode"
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

