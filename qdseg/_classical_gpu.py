"""
GPU-Accelerated Classical Segmentation (cuCIM + CuPy)

Drop-in GPU replacements for the three classical segmentation methods.
Called automatically from segmentation.py when use_gpu is enabled and
cuCIM/CuPy are available.

CPU → GPU library mapping:
    skimage.filters.gaussian              → cucim.skimage.filters.gaussian
    skimage.filters.threshold_otsu        → cucim.skimage.filters.threshold_otsu
    skimage.feature.peak_local_max        → cucim.skimage.feature.peak_local_max
    skimage.filters.sobel                 → cucim.skimage.filters.sobel
    skimage.segmentation.watershed        → CPU only (not in cucim 24.10)
    skimage.morphology.remove_small_objects → cucim.skimage.morphology.remove_small_objects
    skimage.morphology.label              → cupyx.scipy.ndimage.label (cucim has no label)
    skimage.morphology.closing            → cucim.skimage.morphology.closing
    scipy.ndimage.distance_transform_edt  → cupyx.scipy.ndimage.distance_transform_edt
    scipy.ndimage.label                   → cupyx.scipy.ndimage.label
    sklearn.cluster.DBSCAN                → CPU only (small data, GPU transfer overhead > gain)
    non-Otsu thresholds (li, yen, ...)    → CPU threshold value → GPU comparison
"""

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np

# Module-level cache: None = unchecked, True/False = result
_gpu_available: Optional[bool] = None


def is_gpu_available() -> bool:
    """Return True if cuCIM and CuPy are importable and a GPU device is accessible."""
    global _gpu_available
    if _gpu_available is not None:
        return _gpu_available
    try:
        import cupy as cp
        import cupyx.scipy.ndimage  # noqa: F401
        import cucim.skimage.filters  # noqa: F401
        import cucim.skimage.morphology  # noqa: F401
        import cucim.skimage.feature  # noqa: F401
        # Smoke test: allocate a small array and sync
        _ = cp.zeros((4, 4), dtype=cp.float32)
        cp.cuda.Stream.null.synchronize()
        _gpu_available = True
    except Exception:
        _gpu_available = False
    return _gpu_available


def _to_numpy(arr) -> np.ndarray:
    """Convert cupy array or any array-like to a numpy ndarray."""
    try:
        return arr.get()  # cupy ndarray
    except AttributeError:
        return np.asarray(arr)


# ---------------------------------------------------------------------------
# Rule-based GPU
# ---------------------------------------------------------------------------

def segment_advanced_gpu(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    gaussian_sigma: float = 1.0,
    min_area_px: int = 10,
    min_peak_separation_nm: float = 10.0,
) -> np.ndarray:
    """GPU version of segment_advanced (Otsu + DT + DBSCAN + Voronoi)."""
    import cupy as cp
    import cupyx.scipy.ndimage as cpndi
    from cucim.skimage.filters import gaussian as cu_gaussian
    from cucim.skimage.filters import threshold_otsu as cu_threshold_otsu
    from cucim.skimage.morphology import remove_small_objects as cu_remove_small_objects
    from cucim.skimage.feature import peak_local_max as cu_peak_local_max

    pixel_nm = meta.get("pixel_nm", (1.0, 1.0)) if meta else (1.0, 1.0)
    xp_nm, yp_nm = float(pixel_nm[0]), float(pixel_nm[1])

    # Transfer to GPU
    height_gpu = cp.asarray(height, dtype=cp.float32)

    # 1. Gaussian smoothing + Otsu thresholding
    h_smooth_gpu = cu_gaussian(height_gpu, sigma=gaussian_sigma, preserve_range=True)
    threshold = float(cu_threshold_otsu(h_smooth_gpu))
    binary_gpu = h_smooth_gpu > threshold

    # Remove small objects
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        binary_gpu = cu_remove_small_objects(binary_gpu, min_size=min_area_px)

    if not cp.any(binary_gpu):
        return np.zeros(height.shape, dtype=np.int32)

    # 2. Distance transform + peak detection on GPU
    distance_gpu = cpndi.distance_transform_edt(binary_gpu)

    avg_px_nm = float(np.mean(pixel_nm)) if np.all(np.isfinite(pixel_nm)) else 1.0
    min_dist_px = max(1, int(round((min_peak_separation_nm / avg_px_nm) / 2.0)))

    # cucim peak_local_max may return cupy or numpy array depending on version
    coords_raw = cu_peak_local_max(
        distance_gpu,
        labels=binary_gpu,
        min_distance=min_dist_px,
        exclude_border=False,
    )
    coords = _to_numpy(coords_raw)

    if coords.size == 0:
        return np.zeros(height.shape, dtype=np.int32)

    # 3. DBSCAN clustering — kept on CPU (input is small ~10-1000 coords)
    distance_cpu = _to_numpy(distance_gpu)
    try:
        from sklearn.cluster import DBSCAN

        coords_nm = np.column_stack([coords[:, 0] * yp_nm, coords[:, 1] * xp_nm])
        clustering = DBSCAN(eps=min_peak_separation_nm, min_samples=1).fit(coords_nm)

        rep_coords = []
        for lab in np.unique(clustering.labels_):
            idxs = np.where(clustering.labels_ == lab)[0]
            best = idxs[np.argmax(distance_cpu[coords[idxs, 0], coords[idxs, 1]])]
            rep_coords.append(coords[best])
        rep_coords = np.array(rep_coords)
    except ImportError:
        rep_coords = coords

    # 4. Voronoi segmentation from markers on GPU
    binary_cpu = _to_numpy(binary_gpu)
    labels = _voronoi_from_markers_gpu(height.shape, rep_coords, binary_cpu, pixel_nm)

    return labels


# ---------------------------------------------------------------------------
# Watershed GPU
# ---------------------------------------------------------------------------

def segment_watershed_gpu(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    gaussian_sigma: float = 1.0,
    min_distance: int = 5,
    min_area_px: int = 20,
) -> np.ndarray:
    """GPU version of segment_watershed."""
    import cupy as cp
    import cupyx.scipy.ndimage as cpndi
    from cucim.skimage.filters import gaussian as cu_gaussian
    from cucim.skimage.filters import sobel as cu_sobel
    from cucim.skimage.feature import peak_local_max as cu_peak_local_max
    from cucim.skimage.morphology import remove_small_objects as cu_remove_small_objects
    from skimage.segmentation import watershed as cpu_watershed  # cucim has no watershed

    height_gpu = cp.asarray(height, dtype=cp.float32)

    # 1. Gaussian smoothing
    height_smoothed_gpu = cu_gaussian(height_gpu, sigma=gaussian_sigma, preserve_range=True)

    # 2. Local maxima on GPU
    coords_raw = cu_peak_local_max(height_smoothed_gpu, min_distance=min_distance)
    local_max_coords = _to_numpy(coords_raw)

    local_max_gpu = cp.zeros(height_smoothed_gpu.shape, dtype=bool)
    if len(local_max_coords) > 0:
        local_max_gpu[local_max_coords[:, 0], local_max_coords[:, 1]] = True

    # 3. Label local maxima → markers (cupyx.scipy.ndimage.label works; cucim has no label)
    markers_gpu = cpndi.label(local_max_gpu)[0]

    # 4. Sobel gradient on GPU
    gradient_gpu = cu_sobel(height_smoothed_gpu)

    # 5. Watershed — cucim has no watershed; transfer to CPU, then back
    gradient_cpu = _to_numpy(gradient_gpu)
    markers_cpu = _to_numpy(markers_gpu)
    labeled_cpu = cpu_watershed(gradient_cpu, markers_cpu)
    labeled_gpu = cp.asarray(labeled_cpu)

    # 6. Remove small objects
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        labeled_gpu = cu_remove_small_objects(
            labeled_gpu.astype(cp.int64), min_size=min_area_px
        )

    # 7. Relabel sequentially
    labeled_gpu = cpndi.label(labeled_gpu > 0)[0]

    return _to_numpy(labeled_gpu).astype(np.int32)


# ---------------------------------------------------------------------------
# Thresholding GPU
# ---------------------------------------------------------------------------

def segment_thresholding_gpu(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    threshold_method: str = 'otsu',
    threshold_value: Optional[float] = None,
    min_area_px: int = 20,
    closing_size: int = 3,
    use_distance_separation: bool = False,
    min_distance: int = 5,
) -> np.ndarray:
    """GPU version of segment_thresholding."""
    import cupy as cp
    import cupyx.scipy.ndimage as cpndi
    from cucim.skimage.filters import threshold_otsu as cu_threshold_otsu
    from cucim.skimage.feature import peak_local_max as cu_peak_local_max
    from cucim.skimage.morphology import (
        remove_small_objects as cu_remove_small_objects,
        closing as cu_closing,
        disk as cu_disk,
    )
    from skimage.segmentation import watershed as cpu_watershed  # cucim has no watershed

    height_gpu = cp.asarray(height, dtype=cp.float32)

    # 1. Determine threshold
    if threshold_method == 'otsu':
        threshold = float(cu_threshold_otsu(height_gpu))
    elif threshold_method == 'manual':
        if threshold_value is None:
            raise ValueError("threshold_value must be provided when method='manual'")
        threshold = float(threshold_value)
    else:
        # Non-Otsu methods: compute scalar threshold on CPU, apply comparison on GPU
        from skimage import filters as cpu_filters
        height_cpu = _to_numpy(height_gpu)
        _threshold_map = {
            'isodata':  cpu_filters.threshold_isodata,
            'li':       cpu_filters.threshold_li,
            'triangle': cpu_filters.threshold_triangle,
            'yen':      cpu_filters.threshold_yen,
            'minimum':  cpu_filters.threshold_minimum,
        }
        if threshold_method not in _threshold_map:
            raise ValueError(
                f"Unknown threshold_method: {threshold_method}. "
                f"Supported: 'otsu', 'isodata', 'li', 'triangle', 'yen', 'minimum', 'manual'"
            )
        threshold = float(_threshold_map[threshold_method](height_cpu))

    # 2. Binarize on GPU
    binary_gpu = height_gpu > threshold

    # 3. Morphological closing
    if closing_size > 0:
        try:
            selem_gpu = cu_disk(closing_size)
        except Exception:
            # fallback: build disk on CPU, transfer to GPU
            from skimage.morphology import disk as cpu_disk_fn
            selem_gpu = cp.asarray(cpu_disk_fn(closing_size))
        binary_gpu = cu_closing(binary_gpu, selem_gpu)

    # 4. Separate touching grains (optional)
    if use_distance_separation:
        distance_gpu = cpndi.distance_transform_edt(binary_gpu)

        coords_raw = cu_peak_local_max(
            distance_gpu,
            min_distance=min_distance,
            labels=binary_gpu,
        )
        local_max_coords = _to_numpy(coords_raw)

        local_max_gpu = cp.zeros(binary_gpu.shape, dtype=bool)
        if len(local_max_coords) > 0:
            local_max_gpu[local_max_coords[:, 0], local_max_coords[:, 1]] = True

        markers_gpu = cpndi.label(local_max_gpu)[0]
        # Watershed: cucim has no watershed; transfer to CPU, then back
        distance_cpu = _to_numpy(distance_gpu)
        markers_cpu = _to_numpy(markers_gpu)
        binary_cpu = _to_numpy(binary_gpu)
        labeled_cpu = cpu_watershed(-distance_cpu, markers_cpu, mask=binary_cpu)
        labeled_gpu = cp.asarray(labeled_cpu)
    else:
        labeled_gpu = cpndi.label(binary_gpu)[0]

    # 5. Remove small objects
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        labeled_gpu = cu_remove_small_objects(labeled_gpu, min_size=min_area_px)

    # 6. Relabel sequentially
    labeled_gpu = cpndi.label(labeled_gpu > 0)[0]

    return _to_numpy(labeled_gpu).astype(np.int32)


# ---------------------------------------------------------------------------
# Voronoi from markers (GPU core)
# ---------------------------------------------------------------------------

def _voronoi_from_markers_gpu(
    shape: Tuple[int, int],
    markers: np.ndarray,
    mask: np.ndarray,
    pixel_nm: Tuple[float, float],
) -> np.ndarray:
    """
    GPU-accelerated Voronoi segmentation from marker coordinates.

    Uses distance_transform_edt with return_indices=True so that each pixel
    is assigned the label of its nearest marker, restricted to the binary mask.

    Parameters
    ----------
    shape : (H, W)
    markers : (N, 2) array of (row, col) coordinates
    mask : (H, W) boolean array
    pixel_nm : (x_nm, y_nm)

    Returns
    -------
    labels : (H, W) int32 ndarray
    """
    import cupy as cp
    import cupyx.scipy.ndimage as cpndi

    h, w = shape
    if markers.size == 0:
        return np.zeros((h, w), dtype=np.int32)

    xp_nm, yp_nm = float(pixel_nm[0]), float(pixel_nm[1])

    # Build seed mask (CPU), collect valid markers
    seeds = np.zeros((h, w), dtype=bool)
    valid_markers: List[Tuple[int, int]] = []
    for r, c in np.asarray(markers, dtype=int):
        if 0 <= r < h and 0 <= c < w:
            seeds[r, c] = True
            valid_markers.append((r, c))

    if not valid_markers:
        return np.zeros((h, w), dtype=np.int32)

    seeds_gpu = cp.asarray(seeds)
    mask_gpu = cp.asarray(mask)

    # Distance transform: for each pixel find nearest seed (return_indices)
    _, indices_gpu = cpndi.distance_transform_edt(
        ~seeds_gpu, sampling=(yp_nm, xp_nm), return_indices=True
    )
    # indices_gpu shape: (2, H, W)

    # Assign sequential labels to seed positions
    vm = np.asarray(valid_markers)          # (N, 2) numpy — used as CPU index
    seed_labels_gpu = cp.zeros((h, w), dtype=cp.int32)
    seed_labels_gpu[vm[:, 0], vm[:, 1]] = cp.arange(1, len(valid_markers) + 1,
                                                      dtype=cp.int32)

    # Map every pixel to the label of its nearest seed
    labels_gpu = cp.where(
        mask_gpu,
        seed_labels_gpu[indices_gpu[0], indices_gpu[1]],
        cp.int32(0),
    )

    return _to_numpy(labels_gpu).astype(np.int32)
