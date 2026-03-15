"""
Grain Analysis Module (Deprecated)

This module is maintained for backward compatibility.
For new code, use the following modules:
    - grain_analyzer.segmentation: segmentation functions
    - grain_analyzer.statistics: statistics calculation functions

Migration Guide:
    # Old code:
    from grain_analyzer.grain_analysis import segment_by_marker_growth
    from grain_analyzer.grain_analysis import calculate_grain_statistics

    # New code:
    from grain_analyzer.segmentation import segment_advanced
    from grain_analyzer.statistics import calculate_grain_statistics
"""

import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage


# Legacy class - deprecated
class AFMGrainAnalyzer:
    """
    AFM grain analysis class (Deprecated)

    For new code, use the function-based API:
        - segment_advanced(), segment_stardist()
        - calculate_grain_statistics()
    """
    
    def __init__(self):
        warnings.warn(
            "AFMGrainAnalyzer is deprecated. "
            "Use segment_advanced() and calculate_grain_statistics() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.grain_labels = None
        self.grain_props = None
    
    def segment_by_marker_growth(
        self,
        height: np.ndarray,
        marker_coords_px: np.ndarray,
        *,
        meta: Optional[Dict] = None,
        mask: Optional[np.ndarray] = None,
        max_radius_nm: Optional[float] = None,
        anisotropic_nm_metric: bool = True,
    ) -> np.ndarray:
        """Segment by growing circles/ellipses from markers (Legacy)"""
        return segment_by_marker_growth(
            height, marker_coords_px,
            meta=meta, mask=mask,
            max_radius_nm=max_radius_nm,
            anisotropic_nm_metric=anisotropic_nm_metric
        )
    
    def calculate_grain_statistics(
        self,
        grain_labels: np.ndarray,
        grain_props: Dict,
        meta: Optional[Dict] = None,
        min_pixel_area: int = 4
    ) -> Dict:
        """Calculate grain statistics (Legacy)"""
        from .statistics import calculate_grain_statistics as calc_stats
        # Calculate directly from labels without using grain_props
        return calc_stats(grain_labels, None, meta, min_pixel_area=min_pixel_area)


# Legacy function - maintained for backward compatibility
def segment_by_marker_growth(
    height: np.ndarray,
    marker_coords_px: np.ndarray,
    *,
    meta: Optional[Dict] = None,
    mask: Optional[np.ndarray] = None,
    max_radius_nm: Optional[float] = None,
    anisotropic_nm_metric: bool = True,
) -> np.ndarray:
    """
    Voronoi segmentation centered on markers.

    For new code, use segment_advanced() instead.
    This function is maintained for internal use and backward compatibility.

    Parameters
    ----------
    height : np.ndarray
        Height image.
    marker_coords_px : np.ndarray
        Marker coordinates (N, 2) - (row, col).
    meta : dict, optional
        Metadata (with 'pixel_nm' key).
    mask : np.ndarray, optional
        Mask to restrict the segmentation region.
    max_radius_nm : float, optional
        Maximum growth radius (nm).
    anisotropic_nm_metric : bool
        Whether to use anisotropic nm metric.

    Returns
    -------
    labels : np.ndarray
        Label image (int32).
    """
    h, w = height.shape
    labels = np.zeros((h, w), dtype=np.int32)
    
    if marker_coords_px is None or np.size(marker_coords_px) == 0:
        return labels

    # Pixel-to-nm scale
    xp_nm, yp_nm = 1.0, 1.0
    if meta and 'pixel_nm' in meta:
        xp_nm, yp_nm = float(meta['pixel_nm'][0]), float(meta['pixel_nm'][1])
    sampling = (yp_nm, xp_nm) if anisotropic_nm_metric else None

    # Seed mask
    seeds_mask = np.zeros((h, w), dtype=bool)
    valid_coords: List[Tuple[int, int]] = []
    for r, c in np.asarray(marker_coords_px, dtype=int):
        if 0 <= r < h and 0 <= c < w:
            seeds_mask[r, c] = True
            valid_coords.append((r, c))
    
    if not np.any(seeds_mask):
        return labels

    # Nearest seed index (nm metric)
    dist_nm, (iy, ix) = ndimage.distance_transform_edt(
        ~seeds_mask, sampling=sampling, return_indices=True
    )

    # Seed label table
    seed_labels = np.zeros_like(labels)
    for i, (r, c) in enumerate(valid_coords, start=1):
        seed_labels[r, c] = i
    labels = seed_labels[iy, ix]

    # Apply mask constraint
    if mask is not None:
        labels = np.where(mask.astype(bool), labels, 0)

    # Apply radius constraint
    if max_radius_nm is not None and max_radius_nm > 0:
        labels = np.where(dist_nm <= float(max_radius_nm), labels, 0)

    return labels


# Legacy function - redirect to new module
def calculate_grain_statistics(
    grain_labels: np.ndarray,
    grain_props: Dict,
    meta: Optional[Dict] = None,
    min_pixel_area: int = 4
) -> Dict:
    """
    Calculate grain statistics (Legacy).

    For new code, use statistics.calculate_grain_statistics() instead.
    """
    from .statistics import calculate_grain_statistics as calc_stats
    return calc_stats(grain_labels, None, meta, min_pixel_area=min_pixel_area)
