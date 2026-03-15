"""
Grain Segmentation Methods

Each function follows the same input/output format:
    Input: height (2D array), meta (dict)
    Output: labels (2D int array, 0=background, 1,2,3...=grains)

Usage examples:
    >>> from qdseg.segmentation import segment
    >>> labels = segment(height, meta)                        # default: advanced
    >>> labels = segment(height, meta, method='watershed')
    >>> labels = segment(height, meta, method='stardist')
"""

import warnings
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import ndimage as ndi
from skimage.filters import gaussian, threshold_otsu
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.feature import peak_local_max

# Module-level model cache for DL methods (avoids reloading on every call)
_model_cache: Dict[str, object] = {}


# ── TensorFlow GPU helpers ────────────────────────────────────────────────


def _ensure_tf_device(use_gpu: bool = True) -> None:
    """Configure TensorFlow to use GPU or CPU.

    When *use_gpu* is True, runs the normal GPU setup.  When False, sets
    an internal flag so that ``_tf_cpu_context()`` can be used to force
    operations onto CPU via ``tf.device('/CPU:0')``.

    This avoids ``CUDA_ERROR_INVALID_PTX`` on architectures not yet
    supported by the installed TF build (e.g. Blackwell sm_120 with
    TF <= 2.22).
    """
    global _force_tf_cpu, _tf_gpu_initialized
    if use_gpu:
        _force_tf_cpu = False
        if not _tf_gpu_initialized:
            try:
                from .utils import setup_gpu_environment, check_tensorflow_gpu
                setup_gpu_environment()
                check_tensorflow_gpu(verbose=False)
                _tf_gpu_initialized = True
            except ImportError:
                pass
    else:
        _force_tf_cpu = True


_force_tf_cpu: bool = False
_tf_gpu_initialized: bool = False


class _tf_cpu_context:
    """Context manager that forces TF ops onto CPU when ``_force_tf_cpu`` is set."""

    def __enter__(self):
        if _force_tf_cpu:
            import tensorflow as tf
            self._ctx = tf.device('/CPU:0')
            self._ctx.__enter__()
        else:
            self._ctx = None
        return self

    def __exit__(self, *exc):
        if self._ctx is not None:
            return self._ctx.__exit__(*exc)
        return False


def _is_gpu_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a CUDA / GPU compatibility error."""
    msg = str(exc).lower()
    markers = [
        'cuda_error',
        'invalid_ptx',
        'invalid_handle',
        'culaunchkernel',
        'cumoduleloaddata',
        'gpu',
    ]
    return any(m in msg for m in markers)


def _segment_advanced(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    gaussian_sigma: float = 1.0,
    min_area_px: int = 10,
    min_peak_separation_nm: float = 10.0,
    use_gpu: Optional[bool] = None,
) -> np.ndarray:
    """
    Advanced segmentation: Otsu + Distance Transform + DBSCAN + Voronoi

    Steps:
    1. Gaussian smoothing + Otsu thresholding
    2. Distance transform + peak detection
    3. DBSCAN clustering (merge nearby peaks)
    4. Voronoi segmentation from representative peaks

    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units)
    meta : dict, optional
        Metadata with 'pixel_nm' key for pixel size in nm
    gaussian_sigma : float
        Gaussian smoothing sigma (default: 1.0)
    min_area_px : int
        Minimum grain area in pixels (default: 10)
    min_peak_separation_nm : float
        Minimum separation between peaks in nm (default: 10.0)

    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3...=grains)

    Examples
    --------
    >>> labels = segment_advanced(height, meta, gaussian_sigma=1.5)
    >>> num_grains = labels.max()
    """
    pixel_nm = meta.get("pixel_nm", (1.0, 1.0)) if meta else (1.0, 1.0)
    xp_nm, yp_nm = float(pixel_nm[0]), float(pixel_nm[1])

    # GPU dispatch — use_gpu=None means "try GPU silently, fall back to CPU"
    _try_gpu = use_gpu if use_gpu is not None else True
    if _try_gpu:
        try:
            from ._classical_gpu import is_gpu_available, segment_advanced_gpu
            if is_gpu_available():
                return segment_advanced_gpu(
                    height, meta,
                    gaussian_sigma=gaussian_sigma,
                    min_area_px=min_area_px,
                    min_peak_separation_nm=min_peak_separation_nm,
                )
            elif use_gpu is True:
                warnings.warn(
                    "use_gpu=True but cuCIM/CuPy not available. Falling back to CPU.",
                    RuntimeWarning, stacklevel=2,
                )
        except Exception as exc:
            if use_gpu is True:
                warnings.warn(
                    f"GPU segmentation failed ({exc}). Falling back to CPU.",
                    RuntimeWarning, stacklevel=2,
                )

    # 1. Smoothing + Thresholding
    h_smooth = gaussian(height, sigma=gaussian_sigma, preserve_range=True)
    binary = h_smooth > threshold_otsu(h_smooth)

    # Remove small objects (suppress deprecation warning for min_size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        binary = remove_small_objects(binary, min_size=min_area_px)

    if not np.any(binary):
        return np.zeros(height.shape, dtype=np.int32)

    # 2. Distance transform and peak detection
    distance = ndi.distance_transform_edt(binary)
    avg_px_nm = float(np.mean(pixel_nm)) if np.all(np.isfinite(pixel_nm)) else 1.0
    min_dist_px = max(1, int(round((min_peak_separation_nm / avg_px_nm) / 2.0)))

    coords = peak_local_max(
        distance,
        labels=binary,
        min_distance=min_dist_px,
        exclude_border=False,
    )

    if coords.size == 0:
        return np.zeros(height.shape, dtype=np.int32)

    # 3. Cluster nearby peaks using DBSCAN
    try:
        from sklearn.cluster import DBSCAN

        coords_nm = np.column_stack([coords[:, 0] * yp_nm, coords[:, 1] * xp_nm])
        clustering = DBSCAN(eps=min_peak_separation_nm, min_samples=1).fit(coords_nm)

        rep_coords = []
        for lab in np.unique(clustering.labels_):
            idxs = np.where(clustering.labels_ == lab)[0]
            best = idxs[np.argmax(distance[coords[idxs, 0], coords[idxs, 1]])]
            rep_coords.append(coords[best])
        rep_coords = np.array(rep_coords)
    except ImportError:
        # If sklearn is not available, use all coordinates
        rep_coords = coords

    # 4. Voronoi segmentation from markers
    labels = _voronoi_from_markers(height.shape, rep_coords, binary, pixel_nm)

    return labels


def _segment_watershed(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    gaussian_sigma: float = 1.0,
    min_distance: int = 5,
    min_area_px: int = 20,
    use_gpu: Optional[bool] = None,
) -> np.ndarray:
    """
    Watershed-based grain segmentation

    Steps:
    1. Remove noise with Gaussian blur
    2. Detect local maxima as markers
    3. Apply watershed algorithm
    4. Filter small regions

    Parameters
    ----------
    height : np.ndarray
        Height map (2D numpy array)
    meta : dict, optional
        Metadata (optional)
    gaussian_sigma : float
        Gaussian blur sigma (default: 1.0)
    min_distance : int
        Minimum distance between local maxima in pixels (default: 5)
    min_area_px : int
        Minimum grain area in pixels (default: 20)

    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3,...=grain IDs)

    Examples
    --------
    >>> labels = segment_watershed(height)
    >>> labels = segment_watershed(height, min_distance=10)
    """
    # GPU dispatch
    _try_gpu = use_gpu if use_gpu is not None else True
    if _try_gpu:
        try:
            from ._classical_gpu import is_gpu_available, segment_watershed_gpu
            if is_gpu_available():
                return segment_watershed_gpu(
                    height, meta,
                    gaussian_sigma=gaussian_sigma,
                    min_distance=min_distance,
                    min_area_px=min_area_px,
                )
            elif use_gpu is True:
                warnings.warn(
                    "use_gpu=True but cuCIM/CuPy not available. Falling back to CPU.",
                    RuntimeWarning, stacklevel=2,
                )
        except Exception as exc:
            if use_gpu is True:
                warnings.warn(
                    f"GPU segmentation failed ({exc}). Falling back to CPU.",
                    RuntimeWarning, stacklevel=2,
                )

    from skimage import filters, morphology, segmentation

    # 1. Remove noise with Gaussian blur
    height_smoothed = filters.gaussian(height, sigma=gaussian_sigma, preserve_range=True)

    # 2. Otsu thresholding — background mask
    binary = height_smoothed > threshold_otsu(height_smoothed)
    binary = morphology.remove_small_objects(binary, min_size=min_area_px)

    # 3. Detect local maxima (grain peaks) within binary mask
    local_max_coords = peak_local_max(
        height_smoothed,
        min_distance=min_distance,
        labels=binary,
    )

    # Convert coordinates to boolean mask
    local_max = np.zeros(height_smoothed.shape, dtype=bool)
    if len(local_max_coords) > 0:
        local_max[tuple(local_max_coords.T)] = True

    # 4. Create markers
    markers = ndi.label(local_max)[0]

    # 5. Watershed segmentation with binary mask (background excluded)
    gradient = filters.sobel(height_smoothed)
    labeled_image = segmentation.watershed(gradient, markers, mask=binary)

    # 5. Remove small regions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        labeled_image = morphology.remove_small_objects(
            labeled_image.astype(int),
            min_size=min_area_px
        )

    # 6. Relabel sequentially (starting from 1)
    labeled_image = morphology.label(labeled_image > 0)

    return labeled_image.astype(np.int32)


def _segment_thresholding(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    threshold_method: str = 'otsu',
    threshold_value: Optional[float] = None,
    min_area_px: int = 20,
    closing_size: int = 3,
    use_distance_separation: bool = False,
    min_distance: int = 5,
    use_gpu: Optional[bool] = None,
) -> np.ndarray:
    """
    Thresholding-based grain segmentation

    Steps (use_distance_separation=False):
    1. Determine threshold (Otsu, Isodata, Manual, etc.)
    2. Binary thresholding
    3. Morphological closing (fill holes)
    4. Connected component labeling
    5. Filter small regions

    Steps (use_distance_separation=True):
    1. Determine threshold and binarize
    2. Compute Distance Transform
    3. Detect local peaks
    4. Separate touching grains with Watershed
    5. Filter small regions

    Parameters
    ----------
    height : np.ndarray
        Height map (2D numpy array)
    meta : dict, optional
        Metadata (optional)
    threshold_method : str
        Threshold method (default: 'otsu')
        - 'otsu': Otsu's method (histogram-based, class variance maximization)
        - 'isodata': Iterative Isodata method (midpoint of two class means)
        - 'li': Li's minimum cross-entropy method
        - 'triangle': Triangle algorithm (good for unimodal distributions)
        - 'yen': Yen's method (entropy-based)
        - 'minimum': Minimum method (histogram valley)
        - 'manual': User-specified threshold value
    threshold_value : float, optional
        Manual threshold value (nm) (used if method='manual')
    min_area_px : int
        Minimum grain area in pixels (default: 20)
    closing_size : int
        Morphological closing kernel size (default: 3)
    use_distance_separation : bool
        Whether to separate touching grains using Distance Transform + Local Peaks (default: False)
    min_distance : int
        Minimum distance between local peaks in pixels (only used when use_distance_separation=True) (default: 5)

    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3,...=grain IDs)

    Examples
    --------
    >>> labels = segment_thresholding(height)
    >>> labels = segment_thresholding(height, threshold_method='isodata')
    >>> labels = segment_thresholding(height, threshold_method='manual', threshold_value=5.0)
    >>> labels = segment_thresholding(height, use_distance_separation=True, min_distance=10)
    """
    # GPU dispatch
    _try_gpu = use_gpu if use_gpu is not None else True
    if _try_gpu:
        try:
            from ._classical_gpu import is_gpu_available, segment_thresholding_gpu
            if is_gpu_available():
                return segment_thresholding_gpu(
                    height, meta,
                    threshold_method=threshold_method,
                    threshold_value=threshold_value,
                    min_area_px=min_area_px,
                    closing_size=closing_size,
                    use_distance_separation=use_distance_separation,
                    min_distance=min_distance,
                )
            elif use_gpu is True:
                warnings.warn(
                    "use_gpu=True but cuCIM/CuPy not available. Falling back to CPU.",
                    RuntimeWarning, stacklevel=2,
                )
        except Exception as exc:
            if use_gpu is True:
                warnings.warn(
                    f"GPU segmentation failed ({exc}). Falling back to CPU.",
                    RuntimeWarning, stacklevel=2,
                )

    from skimage import filters, morphology, segmentation

    # 1. Determine threshold
    if threshold_method == 'otsu':
        threshold = filters.threshold_otsu(height)
    elif threshold_method == 'isodata':
        threshold = filters.threshold_isodata(height)
    elif threshold_method == 'li':
        threshold = filters.threshold_li(height)
    elif threshold_method == 'triangle':
        threshold = filters.threshold_triangle(height)
    elif threshold_method == 'yen':
        threshold = filters.threshold_yen(height)
    elif threshold_method == 'minimum':
        threshold = filters.threshold_minimum(height)
    elif threshold_method == 'manual':
        if threshold_value is None:
            raise ValueError("threshold_value must be provided when method='manual'")
        threshold = threshold_value
    else:
        raise ValueError(
            f"Unknown threshold_method: {threshold_method}. "
            f"Supported methods: 'otsu', 'isodata', 'li', 'triangle', 'yen', 'minimum', 'manual'"
        )

    # 2. Binarize
    binary = height > threshold

    # 3. Morphological closing (fill small holes)
    if closing_size > 0:
        selem = morphology.disk(closing_size)
        binary = morphology.closing(binary, selem)

    # 4. Separate using Distance Transform + Local Peaks (optional)
    if use_distance_separation:
        # Compute Distance Transform
        distance = ndi.distance_transform_edt(binary)

        # Detect local maxima
        local_max_coords = peak_local_max(
            distance,
            min_distance=min_distance,
            labels=binary,
        )

        # Convert coordinates to boolean mask
        local_max = np.zeros(binary.shape, dtype=bool)
        if len(local_max_coords) > 0:
            local_max[tuple(local_max_coords.T)] = True

        # Create markers
        markers = ndi.label(local_max)[0]

        # Separate with Watershed
        labeled_image = segmentation.watershed(-distance, markers, mask=binary)
    else:
        # 4. Connected component labeling (default)
        labeled_image = morphology.label(binary)

    # 5. Remove small regions
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        labeled_image = morphology.remove_small_objects(
            labeled_image,
            min_size=min_area_px
        )

    # 6. Relabel sequentially
    labeled_image = morphology.label(labeled_image > 0)

    return labeled_image.astype(np.int32)


def _segment_stardist(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    model_name: str = "2D_versatile_fluo",
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    model_path: Optional[str] = None,
    normalize_input: bool = True,
    use_gpu: bool = True,
) -> np.ndarray:
    """
    StarDist-based segmentation using deep learning
    
    Uses star-convex polygon detection for grain segmentation.
    Requires stardist and tensorflow to be installed.
    
    GPU acceleration is auto-detected based on the environment:
    - NVIDIA GPU: CUDA (requires tensorflow-gpu)
    - Apple Silicon: Metal (requires tensorflow-metal)
    - Otherwise: CPU
    
    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units)
    meta : dict, optional
        Metadata (not used, for API consistency)
    model_name : str
        Pretrained model name (default: '2D_versatile_fluo')
        Options: '2D_versatile_fluo', '2D_versatile_he', '2D_paper_dsb2018'
    prob_thresh : float
        Probability threshold for detection (0-1, default: 0.5)
    nms_thresh : float
        Non-maximum suppression threshold (0-1, default: 0.4)
    model_path : str, optional
        Path to custom trained model directory
    normalize_input : bool
        Whether to normalize input image (default: True)
    use_gpu : bool
        Whether to use GPU if available (default: True)
        GPU type is auto-detected (CUDA/Metal/CPU)
    
    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3...=grains)
    
    Raises
    ------
    ImportError
        If stardist or tensorflow is not installed
    
    Examples
    --------
    >>> labels = segment_stardist(height, prob_thresh=0.6)
    >>> # With custom model
    >>> labels = segment_stardist(height, model_path="./my_model")
    >>> # CPU only
    >>> labels = segment_stardist(height, use_gpu=False)
    
    Notes
    -----
    - AFM height images may require custom trained models for best results
    - Pretrained models are optimized for fluorescence microscopy images
    - Install with: pip install stardist tensorflow
    - For Apple Silicon: pip install tensorflow-metal
    - For NVIDIA GPU: pip install tensorflow[cuda]
    
    References
    ----------
    StarDist: https://github.com/stardist/stardist
    """
    # Lazy import - other functions remain available if StarDist is not installed
    try:
        from stardist.models import StarDist2D
    except ImportError:
        raise ImportError(
            "StarDist is not installed. "
            "Install with: pip install stardist tensorflow"
        )

    # GPU setup — detect incompatible GPU and fallback to CPU automatically
    _ensure_tf_device(use_gpu)

    def _load_model():
        """Load or retrieve cached StarDist model."""
        cache_key = f"stardist:{model_path or model_name}"
        if cache_key in _model_cache:
            return _model_cache[cache_key], cache_key
        if model_path:
            import os
            model_dir = os.path.dirname(model_path) or '.'
            m = StarDist2D(None, name=os.path.basename(model_path), basedir=model_dir)
        else:
            m = StarDist2D.from_pretrained(model_name)
        _model_cache[cache_key] = m
        return m, cache_key

    def _run(img):
        """Load model + predict, all within the current TF device context."""
        m, _ = _load_model()
        lbl, _ = m.predict_instances(img, prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        return lbl

    # Normalize input
    img_norm = _normalize_percentile(height, 1, 99.8) if normalize_input else height

    # Try GPU first, fallback to CPU on CUDA errors
    try:
        with _tf_cpu_context():
            labels = _run(img_norm)
    except Exception as exc:
        if use_gpu and _is_gpu_error(exc):
            warnings.warn(
                f"StarDist failed on GPU ({type(exc).__name__}). "
                "Falling back to CPU.",
                RuntimeWarning,
                stacklevel=2,
            )
            # Clear cached model (was built for GPU), force CPU
            cache_key = f"stardist:{model_path or model_name}"
            _model_cache.pop(cache_key, None)
            _ensure_tf_device(use_gpu=False)
            with _tf_cpu_context():
                labels = _run(img_norm)
        else:
            raise

    return labels.astype(np.int32)


def _segment_cellpose(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    model_path: Optional[str] = None,
    normalize_input: bool = True,
    gpu: bool = True,
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Cellpose-based segmentation using deep learning (Cellpose v4 / cpsam).

    Uses the Cellpose-SAM (cpsam) model introduced in Cellpose v4.
    Requires cellpose >= 4.0 to be installed.

    GPU acceleration is auto-detected:
    - NVIDIA GPU: CUDA
    - Apple Silicon: MPS (Metal Performance Shaders)
    - Otherwise: CPU

    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units)
    meta : dict, optional
        Metadata with 'pixel_nm' key (used for diameter estimation)
    diameter : float, optional
        Expected grain diameter in pixels. If None, auto-estimated.
    flow_threshold : float
        Flow error threshold (default: 0.4). Lower = fewer masks.
    cellprob_threshold : float
        Cell probability threshold (default: 0.0). Higher = stricter.
    model_path : str, optional
        Path to a custom trained Cellpose model file.
        If provided, the built-in cpsam model is not used.
    normalize_input : bool
        Whether to normalize input image (default: True)
    gpu : bool
        Whether to use GPU if available (default: True)
    device : torch.device, optional
        Specific device to use. If None, auto-detected.

    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3...=grains)

    Raises
    ------
    ImportError
        If cellpose is not installed

    Examples
    --------
    >>> labels = segment_cellpose(height, diameter=30)
    >>> # With custom trained model
    >>> labels = segment_cellpose(height, model_path="./models/afm_model")
    >>> # CPU only
    >>> labels = segment_cellpose(height, gpu=False)

    Notes
    -----
    - Cellpose v4 uses only the 'cpsam' (Cellpose-SAM) built-in model.
      The model_type argument from v3 (cyto3, cyto2, nuclei, ...) is no
      longer supported; pass a custom model path via model_path if needed.
    - Install with: pip install "cellpose>=4.0"

    References
    ----------
    Cellpose: https://github.com/mouseland/cellpose
    """
    # Lazy import - other functions remain available if Cellpose is not installed
    try:
        from cellpose import models
    except ImportError:
        raise ImportError(
            "Cellpose is not installed. Install with: pip install 'cellpose>=4.0'"
        )

    # Auto-detect device
    if device is None and gpu:
        try:
            from .utils import setup_gpu_environment, get_torch_device
            setup_gpu_environment()
            device = get_torch_device(verbose=False)
            gpu = device.type != 'cpu'
        except ImportError:
            pass

    # Build model kwargs (Cellpose v4: only pretrained_model and device matter)
    cache_key = f"cellpose:{model_path or 'cpsam'}:{gpu}"
    if cache_key in _model_cache:
        model = _model_cache[cache_key]
    else:
        model_kwargs: dict = {'gpu': gpu}
        if device is not None:
            model_kwargs['device'] = device
        if model_path:
            model_kwargs['pretrained_model'] = model_path
        model = models.CellposeModel(**model_kwargs)
        _model_cache[cache_key] = model
    
    # Normalize input
    if normalize_input:
        img_norm = _normalize_percentile(height, 1, 99.8)
        # Cellpose expects 0-255 range for better performance
        img_input = (img_norm * 255).astype(np.float32)
    else:
        img_input = height.astype(np.float32)
    
    # Run segmentation (Cellpose 4.x returns 3 values: masks, flows, styles)
    result = model.eval(
        img_input,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    # result[0] is masks
    masks = result[0]
    
    return masks.astype(np.int32)


def _voronoi_from_markers(
    shape: Tuple[int, int],
    markers: np.ndarray,
    mask: np.ndarray,
    pixel_nm: Tuple[float, float],
) -> np.ndarray:
    """
    Voronoi segmentation from marker coordinates
    
    Uses distance transform with anisotropic metric (nm units)
    to assign each pixel to nearest marker.
    
    Parameters
    ----------
    shape : Tuple[int, int]
        Output shape (height, width)
    markers : np.ndarray
        Marker coordinates (N, 2) as (row, col)
    mask : np.ndarray
        Binary mask limiting segmentation region
    pixel_nm : Tuple[float, float]
        Pixel size (x_nm, y_nm)
    
    Returns
    -------
    labels : np.ndarray
        Label image
    """
    h, w = shape
    labels = np.zeros((h, w), dtype=np.int32)
    
    if markers.size == 0:
        return labels
    
    xp_nm, yp_nm = float(pixel_nm[0]), float(pixel_nm[1])
    
    # Create seed mask
    seeds = np.zeros((h, w), dtype=bool)
    valid_markers = []
    for r, c in np.asarray(markers, dtype=int):
        if 0 <= r < h and 0 <= c < w:
            seeds[r, c] = True
            valid_markers.append((r, c))
    
    if not valid_markers:
        return labels
    
    # Distance transform with nm metric (anisotropic)
    sampling = (yp_nm, xp_nm)
    _, (iy, ix) = ndi.distance_transform_edt(
        ~seeds, sampling=sampling, return_indices=True
    )
    
    # Assign labels based on nearest marker
    seed_labels = np.zeros_like(labels)
    vm = np.asarray(valid_markers)
    seed_labels[vm[:, 0], vm[:, 1]] = np.arange(1, len(valid_markers) + 1)
    
    labels = seed_labels[iy, ix]
    
    # Apply mask
    labels = np.where(mask, labels, 0)
    
    return labels


def _normalize_percentile(
    img: np.ndarray,
    pmin: float = 1,
    pmax: float = 99.8,
) -> np.ndarray:
    """
    Percentile-based normalization to [0, 1]
    
    Parameters
    ----------
    img : np.ndarray
        Input image
    pmin : float
        Lower percentile (default: 1)
    pmax : float
        Upper percentile (default: 99.8)
    
    Returns
    -------
    np.ndarray
        Normalized image in [0, 1]
    """
    mi = np.percentile(img, pmin)
    ma = np.percentile(img, pmax)
    
    if ma - mi < 1e-10:
        return np.zeros_like(img, dtype=np.float32)
    
    return np.clip((img - mi) / (ma - mi), 0, 1).astype(np.float32)


def segment_cellulus(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    model_path: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    normalize_input: bool = True,
    gpu: bool = True,
    device: Optional["torch.device"] = None,
    use_official: bool = True,
    # Official Cellulus parameters
    num_fmaps: int = 24,
    fmap_inc_factor: int = 3,
    p_salt_pepper: float = 0.01,
    num_infer_iterations: int = 16,
    bandwidth: Optional[float] = None,
    clustering: str = "meanshift",
    grow_distance: int = 3,
    shrink_distance: int = 6,
    min_size: Optional[int] = None,
) -> np.ndarray:
    """
    Cellulus-based segmentation using unsupervised deep learning
    
    Cellulus is an unsupervised learning-based instance segmentation method.
    It learns object-centric embeddings to perform segmentation without labels.

    This function supports two model formats:
    1. Official Cellulus model (use_official=True, requires cellulus package)
    2. Simplified model (use_official=False, requires PyTorch only)

    GPU acceleration is auto-detected based on the environment:
    - NVIDIA GPU: CUDA
    - Apple Silicon: MPS
    - Otherwise: CPU
    
    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units)
    meta : dict, optional
        Metadata (not used, for API consistency)
    model_path : str, optional
        Path to trained Cellulus model directory
    checkpoint_path : str, optional
        Path to model checkpoint file (.pth)
    normalize_input : bool
        Whether to normalize input image (default: True)
    gpu : bool
        Whether to use GPU if available (default: True)
    device : torch.device, optional
        Specific device to use. If None, auto-detected.
    use_official : bool
        Whether to use official Cellulus model (default: True)
    num_fmaps : int
        Number of feature maps in first layer (default: 24)
    fmap_inc_factor : int
        Feature map increase factor (default: 3)
    p_salt_pepper : float
        Salt-pepper noise probability for uncertainty (default: 0.01)
    num_infer_iterations : int
        Number of inference iterations (default: 16)
    bandwidth : float, optional
        MeanShift bandwidth. If None, auto-detected.
    clustering : str
        Clustering method: "meanshift" or "greedy" (default: "meanshift")
    grow_distance : int
        Morphological grow distance (default: 3)
    shrink_distance : int
        Morphological shrink distance (default: 6)
    min_size : int, optional
        Minimum object size filter. If None, no filtering.
    
    Returns
    -------
    labels : np.ndarray
        Label image (int32, 0=background, 1,2,3...=grains)
    
    Raises
    ------
    ImportError
        If neither cellulus nor torch is installed
    ValueError
        If no model path or checkpoint is provided
    
    Examples
    --------
    >>> # With official Cellulus model
    >>> labels = segment_cellulus(height, checkpoint_path="./models/best_loss.pth")
    
    >>> # With simple model (faster)
    >>> labels = segment_cellulus(height, checkpoint_path="./models/best_loss.pth", use_official=False)
    
    Notes
    -----
    - Train your model with: python train_model.py or cellulus CLI
    - Cellulus requires training on your own data (unsupervised)
    - Best suited for microscopy images with consistent object appearance
    
    References
    ----------
    Cellulus: https://github.com/funkelab/cellulus
    Paper: "Unsupervised Learning of Object-Centric Embeddings for Cell 
           Instance Segmentation in Microscopy Images" (ICCV 2023)
    """
    import torch
    
    if checkpoint_path is None and model_path is None:
        raise ValueError(
            "Cellulus requires a trained model. "
            "Please specify checkpoint_path. "
            "Train a model with: python train_model.py"
        )
    
    # Auto-detect device
    if device is None and gpu:
        try:
            from .utils import setup_gpu_environment, get_torch_device
            setup_gpu_environment()
            device = get_torch_device(verbose=False)
        except ImportError:
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
    elif device is None:
        device = torch.device('cpu')
    
    # Normalize input
    if normalize_input:
        img_norm = _normalize_percentile(height, 1, 99.8)
    else:
        img_norm = height.astype(np.float32)
    
    # Try official Cellulus
    if use_official:
        try:
            device_str = str(device) if device else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cuda:0" if torch.cuda.is_available() else "cpu")
            
            labels = _segment_with_official_cellulus(
                img_norm,
                checkpoint_path,
                device_str,
                num_fmaps=num_fmaps,
                fmap_inc_factor=fmap_inc_factor,
                p_salt_pepper=p_salt_pepper,
                num_infer_iterations=num_infer_iterations,
                bandwidth=bandwidth,
                clustering=clustering,
                grow_distance=grow_distance,
                shrink_distance=shrink_distance,
                min_size=min_size,
            )
            if labels is not None:
                return labels.astype(np.int32)
        except ImportError:
            pass  # If official Cellulus not available, try simplified model
        except Exception as e:
            import traceback
            traceback.print_exc()
            pass  # On failure, try simplified model
    
    # Try simplified model (PyTorch checkpoint)
    labels = _segment_with_simple_cellulus(img_norm, checkpoint_path, device)
    if labels is not None:
        return labels.astype(np.int32)
    
    raise RuntimeError(
        "Failed to load model. "
        "Train a model with: python train_model.py"
    )


def _segment_with_official_cellulus(
    img_norm: np.ndarray,
    checkpoint_path: str,
    device: str,
    *,
    num_fmaps: int = 24,
    fmap_inc_factor: int = 3,
    crop_size: int = 252,
    p_salt_pepper: float = 0.01,
    num_infer_iterations: int = 16,
    bandwidth: Optional[float] = None,
    num_bandwidths: int = 1,
    reduction_probability: float = 0.1,
    clustering: str = "meanshift",
    post_processing: str = "cell",
    grow_distance: int = 3,
    shrink_distance: int = 6,
    min_size: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Segmentation using the official Cellulus CLI.

    Calls the official Cellulus infer pipeline:
    1. Convert input image to zarr format
    2. Create ExperimentConfig
    3. Call infer_experiment (predict -> detect -> segment)
    4. Load segmentation from result zarr

    Parameters
    ----------
    img_norm : np.ndarray
        Normalized input image (H, W), value range [0, 1]
    checkpoint_path : str
        Path to trained Cellulus model checkpoint
    device : str
        Device ("cuda:0", "mps", "cpu")
    num_fmaps : int
        Number of feature maps in the first layer
    fmap_inc_factor : int
        Feature map increase factor between layers
    crop_size : int
        Crop size used for inference
    p_salt_pepper : float
        Salt-pepper noise probability
    num_infer_iterations : int
        Number of inference iterations
    bandwidth : float, optional
        MeanShift bandwidth (auto-detected if None)
    num_bandwidths : int
        Number of bandwidths to use
    reduction_probability : float
        Fraction of pixels used for clustering
    clustering : str
        Clustering method ("meanshift" or "greedy")
    post_processing : str
        Post-processing method ("cell" or "nucleus")
    grow_distance : int
        Morphological grow distance
    shrink_distance : int
        Morphological shrink distance
    min_size : int, optional
        Minimum object size (no filtering if None)

    Returns
    -------
    labels : np.ndarray or None
        Segmentation result (H, W), None on failure
    """
    import os
    import tempfile
    import shutil
    
    try:
        import zarr
        import toml
        from cellulus.cli import ExperimentConfig, infer_experiment
    except ImportError:
        return None
    
    try:
        # Verify and reshape input
        # (H, W) -> (1, 1, H, W)
        images = img_norm[np.newaxis, np.newaxis, ...].astype(np.float32)
        
        # Use parent directory of model path as working directory
        checkpoint_path_abs = os.path.abspath(checkpoint_path)
        model_dir = os.path.dirname(os.path.dirname(checkpoint_path_abs))  # parent of models/
        
        # Create temporary directory (inside model folder)
        temp_dir = tempfile.mkdtemp(prefix="cellulus_infer_", dir=model_dir)
        zarr_path = os.path.join(temp_dir, "data.zarr")
        
        # Relative path of model checkpoint (relative to working directory)
        checkpoint_rel = os.path.relpath(checkpoint_path_abs, model_dir)
        
        try:
            # Create zarr file
            root = zarr.open(zarr_path, mode='w')
            root['train/raw'] = images
            root['train/raw'].attrs['resolution'] = (1, 1)
            root['train/raw'].attrs['axis_names'] = ('s', 'c', 'y', 'x')
            
            # Convert zarr path to relative path
            zarr_rel = os.path.relpath(zarr_path, model_dir)
            
            # Create TOML configuration (dict)
            config = {
                'normalization_factor': 1.0,
                'model_config': {
                    'num_fmaps': num_fmaps,
                    'fmap_inc_factor': fmap_inc_factor,
                    'checkpoint': checkpoint_rel,
                },
                'inference_config': {
                    'crop_size': [crop_size, crop_size],
                    'device': device,
                    'p_salt_pepper': p_salt_pepper,
                    'num_infer_iterations': num_infer_iterations,
                    'num_bandwidths': num_bandwidths,
                    'reduction_probability': reduction_probability,
                    'clustering': clustering,
                    'post_processing': post_processing,
                    'grow_distance': grow_distance,
                    'shrink_distance': shrink_distance,
                    'dataset_config': {
                        'container_path': zarr_rel,
                        'dataset_name': 'train/raw',
                    },
                    'prediction_dataset_config': {
                        'container_path': zarr_rel,
                        'dataset_name': 'embeddings',
                    },
                    'detection_dataset_config': {
                        'container_path': zarr_rel,
                        'dataset_name': 'detection',
                        'secondary_dataset_name': 'embeddings',
                    },
                    'segmentation_dataset_config': {
                        'container_path': zarr_rel,
                        'dataset_name': 'segmentation',
                        'secondary_dataset_name': 'detection',
                    },
                },
            }
            
            if bandwidth is not None:
                config['inference_config']['bandwidth'] = bandwidth
            if min_size is not None:
                config['inference_config']['min_size'] = min_size
            
            # Change working directory and run Cellulus
            original_cwd = os.getcwd()
            os.chdir(model_dir)
            
            try:
                # Load ExperimentConfig and run inference
                experiment_config = ExperimentConfig(**config)
                infer_experiment(experiment_config)
            finally:
                os.chdir(original_cwd)
            
            # Load results
            root = zarr.open(zarr_path, mode='r')
            labels = np.array(root['segmentation'][:])  # (1, 1, H, W)
            labels = labels[0, 0, :, :]  # (H, W)
            
            return labels.astype(np.int32)
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None


def _segment_with_simple_cellulus(
    img_norm: np.ndarray,
    checkpoint_path: str,
    device: "torch.device",
) -> Optional[np.ndarray]:
    """
    Segmentation with simplified Cellulus model.

    Uses the simplified model trained with train_model.py.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from scipy import ndimage as ndi_local
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max as plm
    
    # Define simplified U-Net model
    class SimpleUNet(nn.Module):
        def __init__(self, in_channels=1, num_embeddings=8, num_fmaps=32):
            super().__init__()
            
            self.enc1 = self._block(in_channels, num_fmaps)
            self.enc2 = self._block(num_fmaps, num_fmaps * 2)
            self.enc3 = self._block(num_fmaps * 2, num_fmaps * 4)
            
            self.dec2 = self._block(num_fmaps * 4 + num_fmaps * 2, num_fmaps * 2)
            self.dec1 = self._block(num_fmaps * 2 + num_fmaps, num_fmaps)
            
            self.out = nn.Conv2d(num_fmaps, num_embeddings, 1)
            
            self.pool = nn.MaxPool2d(2)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        def _block(self, in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            
            d2 = self.dec2(torch.cat([self.up(e3), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
            
            return self.out(d1)
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

        # Check for simple model format
        if 'model_state_dict' not in checkpoint:
            return None  # Not a simple Cellulus model

        # Build model and load weights
        model = SimpleUNet().to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Build input tensor
        img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0)
        img_tensor = img_tensor.to(device)

        # Extract embeddings
        with torch.no_grad():
            embeddings = model(img_tensor)  # (1, C, H, W)

        # Embeddings -> instance labels
        embeddings = embeddings.squeeze(0).cpu().numpy()  # (C, H, W)
        labels = _embeddings_to_labels(embeddings, img_norm)

        return labels

    except Exception as e:
        import warnings
        warnings.warn(
            f"Simple Cellulus segmentation failed: {e}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _embeddings_to_labels(embeddings: np.ndarray, img: np.ndarray) -> np.ndarray:
    """
    Convert object-centric embeddings to instance labels.

    Groups similar pixels in embedding space.
    """
    from scipy import ndimage as ndi_local
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max as plm
    
    c, h, w = embeddings.shape
    
    # Normalize embeddings
    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=0, keepdims=True) + 1e-10)
    
    # Compute embedding gradient for each pixel (boundary detection)
    gradient_x = np.abs(np.diff(embeddings_norm, axis=2))
    gradient_y = np.abs(np.diff(embeddings_norm, axis=1))
    
    # Padding
    gradient_x = np.pad(gradient_x, ((0, 0), (0, 0), (0, 1)), mode='edge')
    gradient_y = np.pad(gradient_y, ((0, 0), (0, 1), (0, 0)), mode='edge')
    
    # Boundary strength
    boundary = np.sqrt(np.sum(gradient_x**2, axis=0) + np.sum(gradient_y**2, axis=0))
    
    # Create markers based on boundaries
    smooth = ndi_local.gaussian_filter(1 - boundary, sigma=2)
    
    # Find local maxima
    coords = plm(smooth, min_distance=5, threshold_abs=0.1)
    
    if len(coords) == 0:
        return np.zeros((h, w), dtype=np.int32)
    
    # Create marker image
    markers = np.zeros((h, w), dtype=np.int32)
    for i, (r, c) in enumerate(coords, start=1):
        markers[r, c] = i
    
    # Watershed segmentation
    labels = watershed(boundary, markers)
    
    return labels.astype(np.int32)



# ── Public dispatcher ──────────────────────────────────────────────────────────


def segment(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    method: str = "advanced",
    **kwargs
) -> np.ndarray:
    """Segment grains in an AFM height image.

    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units).
    meta : dict, optional
        Metadata with 'pixel_nm' key.
    method : str
        Segmentation method.  One of:
        - ``'advanced'``     (default) Otsu + Distance + DBSCAN + Voronoi
        - ``'watershed'``    Watershed-based
        - ``'thresholding'`` Height threshold based
        - ``'stardist'``     StarDist deep learning (requires ``stardist``)
        - ``'cellpose'``     CellPose deep learning (requires ``cellpose``)
    **kwargs
        Additional keyword arguments forwarded to the selected method.

    Returns
    -------
    np.ndarray
        Label image (int32, 0 = background, 1 … N = grain IDs).

    Examples
    --------
    >>> from qdseg import segment
    >>> labels = segment(height, meta)
    >>> labels = segment(height, meta, method='watershed')
    >>> labels = segment(height, meta, method='stardist', prob_thresh=0.6)
    """
    _dispatch = {
        "advanced":     _segment_advanced,
        "watershed":    _segment_watershed,
        "thresholding": _segment_thresholding,
        "stardist":     _segment_stardist,
        "cellpose":     _segment_cellpose,
    }
    if method not in _dispatch:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {list(_dispatch)}"
        )
    return _dispatch[method](height, meta, **kwargs)
