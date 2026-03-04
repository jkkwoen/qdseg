"""
Grain Statistics Calculation

Method-agnostic statistics from a label image.

Examples
--------
>>> from qdseg.statistics import calculate_grain_statistics
>>> stats = calculate_grain_statistics(labels, height, meta)
>>> print(f"Number of grains: {stats['num_grains']}")
"""

import math
import numpy as np
from typing import Dict, Optional, List, Any
from skimage import measure
from scipy.ndimage import sum as ndi_sum, maximum_position


def calculate_grain_statistics(
    labels: np.ndarray,
    height: Optional[np.ndarray] = None,
    meta: Optional[Dict] = None,
    *,
    min_pixel_area: int = 4,
    _precomputed_grains: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Calculate statistics from grain labels.

    Parameters
    ----------
    labels : np.ndarray
        Label image (0=background, 1,2,3...=grains)
    height : np.ndarray, optional
        Height image for intensity-based statistics
    meta : dict, optional
        Metadata with 'pixel_nm' key for pixel size
    min_pixel_area : int
        Minimum grain area in pixels to include (default: 4)
    _precomputed_grains : list of dict, optional
        If provided, skip the internal ``get_individual_grains`` call and
        use these grain dicts directly.  This avoids redundant computation
        when the caller already has the per-grain data.

    Returns
    -------
    stats : dict
        Grain statistics including:
        - num_grains: int
        - mean_area_nm2, std_area_nm2, min_area_nm2, max_area_nm2: float
        - mean_area_px, std_area_px, min_area_px, max_area_px: float
        - mean_diameter_nm, std_diameter_nm: float
        - mean_diameter_px, std_diameter_px: float
        - mean_perimeter_nm, std_perimeter_nm: float
        - mean_perimeter_px, std_perimeter_px: float
        - mean_eccentricity, mean_solidity, mean_aspect_ratio: float
        - grain_density, area_fraction: float
        - mean_height_nm, std_height_nm, min_height_nm, max_height_nm: float (if height provided)
        - mean_height_peak_nm, std_height_peak_nm: float (if height provided)
        - mean_height_centroid_nm, std_height_centroid_nm: float (if height provided)
        - mean_volume_nm3, std_volume_nm3: float (if height provided)
        - areas_nm2, diameters_nm: np.ndarray (per-grain arrays)

    Examples
    --------
    >>> stats = calculate_grain_statistics(labels, height, meta)
    >>> print(f"Found {stats['num_grains']} grains")
    >>> print(f"Mean diameter: {stats['mean_diameter_nm']:.1f} nm")
    """
    if labels.max() == 0:
        return _empty_statistics()

    # If height is provided, use detailed grain analysis for more statistics
    if height is not None:
        if _precomputed_grains is not None:
            individual_grains = _precomputed_grains
        else:
            individual_grains = get_individual_grains(labels, height, meta, min_pixel_area=min_pixel_area)
        
        if len(individual_grains) == 0:
            return _empty_statistics()
        
        # Extract arrays from individual grains
        areas_px = np.array([g['area_px'] for g in individual_grains])
        areas_nm2 = np.array([g['area_nm2'] for g in individual_grains])
        diameters_nm = np.array([g['diameter_nm'] for g in individual_grains])
        diameters_px = np.array([g['diameter_px'] for g in individual_grains])
        perimeters_nm = np.array([g['perimeter_nm'] for g in individual_grains])
        perimeters_px = np.array([g['perimeter_px'] for g in individual_grains])
        eccentricities = np.array([g['eccentricity'] for g in individual_grains])
        solidities = np.array([g['solidity'] for g in individual_grains])
        aspect_ratios = np.array([g['aspect_ratio'] for g in individual_grains])
        major_axis_nm = np.array([g['major_axis_nm'] for g in individual_grains])
        minor_axis_nm = np.array([g['minor_axis_nm'] for g in individual_grains])
        orientations_deg = np.array([g['orientation_deg'] for g in individual_grains])
        
        # Height-based statistics
        height_means = np.array([g['height_mean_nm'] for g in individual_grains])
        peak_heights = np.array([g['height_peak_nm'] for g in individual_grains])
        centroid_heights = np.array([g['height_centroid_nm'] for g in individual_grains])
        volumes = np.array([g['volume_nm3'] for g in individual_grains])
        
        # Calculate coverage statistics
        total_area_px = labels.shape[0] * labels.shape[1]
        grain_area_px = np.sum(areas_px)
        
        return {
            # Counts
            'num_grains': len(individual_grains),
            
            # Area statistics
            'mean_area_px': float(np.mean(areas_px)),
            'mean_area_nm2': float(np.mean(areas_nm2)),
            'std_area_px': float(np.std(areas_px)),
            'std_area_nm2': float(np.std(areas_nm2)),
            'min_area_px': float(np.min(areas_px)),
            'max_area_px': float(np.max(areas_px)),
            'min_area_nm2': float(np.min(areas_nm2)),
            'max_area_nm2': float(np.max(areas_nm2)),
            
            # Diameter statistics (nm and px)
            'mean_diameter_nm': float(np.mean(diameters_nm)),
            'std_diameter_nm': float(np.std(diameters_nm)),
            'min_diameter_nm': float(np.min(diameters_nm)),
            'max_diameter_nm': float(np.max(diameters_nm)),
            'mean_diameter_px': float(np.mean(diameters_px)),
            'std_diameter_px': float(np.std(diameters_px)),
            
            # Perimeter statistics (nm and px)
            'mean_perimeter_nm': float(np.mean(perimeters_nm)),
            'std_perimeter_nm': float(np.std(perimeters_nm)),
            'mean_perimeter_px': float(np.mean(perimeters_px)),
            'std_perimeter_px': float(np.std(perimeters_px)),
            
            # Shape statistics
            'mean_eccentricity': float(np.mean(eccentricities)),
            'mean_solidity': float(np.mean(solidities)),
            'mean_aspect_ratio': float(np.mean(aspect_ratios)),
            
            # Height statistics
            'mean_height_nm': float(np.mean(height_means)),
            'std_height_nm': float(np.std(height_means)),
            'min_height_nm': float(np.min(height_means)),
            'max_height_nm': float(np.max(height_means)),
            
            # Peak height statistics
            'mean_height_peak_nm': float(np.mean(peak_heights)),
            'std_height_peak_nm': float(np.std(peak_heights)),

            # Centroid height statistics
            'mean_height_centroid_nm': float(np.mean(centroid_heights)),
            'std_height_centroid_nm': float(np.std(centroid_heights)),
            
            # Volume statistics
            'mean_volume_nm3': float(np.mean(volumes)),
            'std_volume_nm3': float(np.std(volumes)),
            
            # Coverage statistics
            'grain_density': float(len(individual_grains)) / total_area_px,
            'area_fraction': float(grain_area_px) / total_area_px,
            
            # Per-grain arrays
            'areas_px': areas_px,
            'areas_nm2': areas_nm2,
            'diameters_nm': diameters_nm,
            'diameters_px': diameters_px,
            'perimeters_nm': perimeters_nm,
            'perimeters_px': perimeters_px,
            'eccentricities': eccentricities,
            'solidities': solidities,
            'aspect_ratios': aspect_ratios,
            'major_axis_nm': major_axis_nm,
            'minor_axis_nm': minor_axis_nm,
            'orientations_rad': np.deg2rad(orientations_deg),
        }
    
    # Fallback: basic statistics without height data
    # Get region properties
    props = measure.regionprops_table(
        labels,
        intensity_image=height,
        properties=[
            'area', 'perimeter', 'eccentricity', 'solidity',
            'major_axis_length', 'minor_axis_length', 'orientation',
            'centroid'
        ]
    )
    
    # Filter small grains
    valid = props['area'] >= min_pixel_area
    if not np.any(valid):
        return _empty_statistics()
    
    # Extract valid properties
    areas_px = props['area'][valid]
    perimeters_px = props['perimeter'][valid]
    eccentricities = props['eccentricity'][valid]
    solidities = props['solidity'][valid]
    major_axis_px = props['major_axis_length'][valid]
    minor_axis_px = props['minor_axis_length'][valid]
    orientations = props['orientation'][valid]
    
    # Get pixel size in nm
    pixel_area_nm2 = 1.0
    pixel_length_nm = 1.0
    if meta and 'pixel_nm' in meta:
        px, py = meta['pixel_nm']
        pixel_area_nm2 = float(px) * float(py)
        pixel_length_nm = math.sqrt(pixel_area_nm2)
    
    # Convert to nm units
    areas_nm2 = areas_px * pixel_area_nm2
    perimeters_nm = perimeters_px * pixel_length_nm
    diameters_nm = 2 * np.sqrt(areas_nm2 / np.pi)
    diameters_px = 2 * np.sqrt(areas_px / np.pi)
    major_axis_nm = major_axis_px * pixel_length_nm
    minor_axis_nm = minor_axis_px * pixel_length_nm
    
    # Calculate aspect ratios (handle division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        aspect_ratios = major_axis_px / minor_axis_px
        aspect_ratios = np.nan_to_num(aspect_ratios, nan=1.0, posinf=1.0)
    
    # Calculate coverage statistics
    total_area_px = labels.shape[0] * labels.shape[1]
    grain_area_px = np.sum(areas_px)
    
    return {
        # Counts
        'num_grains': int(np.sum(valid)),
        
        # Area statistics
        'mean_area_px': float(np.mean(areas_px)),
        'mean_area_nm2': float(np.mean(areas_nm2)),
        'std_area_px': float(np.std(areas_px)),
        'std_area_nm2': float(np.std(areas_nm2)),
        'min_area_px': float(np.min(areas_px)),
        'max_area_px': float(np.max(areas_px)),
        'min_area_nm2': float(np.min(areas_nm2)),
        'max_area_nm2': float(np.max(areas_nm2)),
        
        # Diameter statistics
        'mean_diameter_nm': float(np.mean(diameters_nm)),
        'std_diameter_nm': float(np.std(diameters_nm)),
        'min_diameter_nm': float(np.min(diameters_nm)),
        'max_diameter_nm': float(np.max(diameters_nm)),
        'mean_diameter_px': float(np.mean(diameters_px)),
        'std_diameter_px': float(np.std(diameters_px)),
        
        # Perimeter statistics
        'mean_perimeter_nm': float(np.mean(perimeters_nm)),
        'std_perimeter_nm': float(np.std(perimeters_nm)),
        'mean_perimeter_px': float(np.mean(perimeters_px)),
        'std_perimeter_px': float(np.std(perimeters_px)),
        
        # Shape statistics
        'mean_eccentricity': float(np.mean(eccentricities)),
        'mean_solidity': float(np.mean(solidities)),
        'mean_aspect_ratio': float(np.mean(aspect_ratios)),
        
        # Coverage statistics
        'grain_density': float(np.sum(valid)) / total_area_px,
        'area_fraction': float(grain_area_px) / total_area_px,
        
        # Per-grain arrays
        'areas_px': areas_px,
        'areas_nm2': areas_nm2,
        'diameters_nm': diameters_nm,
        'diameters_px': diameters_px,
        'perimeters_nm': perimeters_nm,
        'perimeters_px': perimeters_px,
        'eccentricities': eccentricities,
        'solidities': solidities,
        'aspect_ratios': aspect_ratios,
        'major_axis_nm': major_axis_nm,
        'minor_axis_nm': minor_axis_nm,
        'orientations_rad': orientations,
    }


def get_individual_grains(
    labels: np.ndarray,
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    min_pixel_area: int = 4,
) -> List[Dict[str, Any]]:
    """
    Get detailed properties for each individual grain
    
    Parameters
    ----------
    labels : np.ndarray
        Label image (0=background)
    height : np.ndarray
        Height image (nm units)
    meta : dict, optional
        Metadata with 'pixel_nm' key
    min_pixel_area : int
        Minimum grain area to include (default: 4)
    
    Returns
    -------
    grains : List[Dict]
        List of dictionaries, one per grain, containing:
        - grain_id: int
        - area_nm2, area_px, diameter_nm, diameter_px, volume_nm3: float
        - centroid_x_nm, centroid_y_nm, centroid_x_px, centroid_y_px: float
        - height_centroid_nm: float (height at centroid position)
        - peak_x_nm, peak_y_nm, peak_x_px, peak_y_px, height_peak_nm: float
        - peak_to_centroid_dist_nm: float (distance between peak and centroid)
        - equivalent_radius_nm: float (radius from area: sqrt(area/π))
        - height_mean_nm, height_std_nm, height_max_nm: float
        - major_axis_nm, minor_axis_nm, aspect_ratio: float
        - perimeter_nm, perimeter_px: float
        - eccentricity, solidity, orientation_deg: float
    
    Examples
    --------
    >>> grains = get_individual_grains(labels, height, meta)
    >>> for g in grains:
    ...     print(f"Grain {g['grain_id']}: {g['diameter_nm']:.1f} nm")
    """
    if labels.max() == 0:
        return []

    # Get pixel size
    xp_nm, yp_nm = 1.0, 1.0
    if meta and 'pixel_nm' in meta:
        xp_nm, yp_nm = float(meta['pixel_nm'][0]), float(meta['pixel_nm'][1])
    px_area_nm2 = xp_nm * yp_nm
    px_length_nm = math.sqrt(px_area_nm2)

    H, W = height.shape

    # ── regionprops_table: single pass with intensity stats ──
    props = measure.regionprops_table(
        labels,
        intensity_image=height,
        properties=[
            'label', 'area', 'centroid', 'perimeter',
            'major_axis_length', 'minor_axis_length', 'orientation',
            'eccentricity', 'solidity', 'convex_area',
            'intensity_mean', 'intensity_max', 'intensity_min',
        ]
    )

    # ── Filter by min_pixel_area up front ──
    valid = props['area'] >= min_pixel_area
    if not np.any(valid):
        return []

    label_ids   = props['label'][valid]
    area_px     = props['area'][valid].astype(float)
    cent_y      = props['centroid-0'][valid]
    cent_x      = props['centroid-1'][valid]
    perimeter_px_arr = props['perimeter'][valid]
    major_px    = props['major_axis_length'][valid]
    minor_px    = props['minor_axis_length'][valid]
    orientation = props['orientation'][valid]
    ecc         = props['eccentricity'][valid]
    sol         = props['solidity'][valid]
    convex_area = props['convex_area'][valid].astype(float)
    i_mean      = props['intensity_mean'][valid]
    i_max       = props['intensity_max'][valid]
    i_min       = props['intensity_min'][valid]

    n = len(label_ids)

    # ── Derived geometry (vectorised) ──
    area_nm2            = area_px * px_area_nm2
    diameter_nm         = 2.0 * np.sqrt(area_nm2 / np.pi)
    diameter_px         = 2.0 * np.sqrt(area_px / np.pi)
    equivalent_radius   = np.sqrt(area_nm2 / np.pi)
    perimeter_nm_arr    = perimeter_px_arr * px_length_nm
    major_nm            = major_px * px_length_nm
    minor_nm            = minor_px * px_length_nm
    with np.errstate(divide='ignore', invalid='ignore'):
        aspect_ratio = np.where(minor_px > 0, major_px / minor_px, 1.0)
    orientation_deg     = orientation * (180.0 / np.pi)
    convex_area_nm2     = convex_area * px_area_nm2

    # Centroid in nm
    cx_nm = cent_x * xp_nm
    cy_nm = cent_y * yp_nm

    # ── Centroid height: vectorised lookup ──
    cy_int = np.clip(np.round(cent_y).astype(int), 0, H - 1)
    cx_int = np.clip(np.round(cent_x).astype(int), 0, W - 1)
    centroid_h = height[cy_int, cx_int]

    # ── Volume: single-pass ndi_sum ──
    all_volumes = np.atleast_1d(
        ndi_sum(height, labels, index=label_ids)
    ) * px_area_nm2

    # ── Height std: E[X²] − E[X]²  (ddof=0, matches np.std default) ──
    all_sum_sq = np.atleast_1d(
        ndi_sum(height ** 2, labels, index=label_ids)
    )
    variance = np.maximum(all_sum_sq / area_px - i_mean ** 2, 0.0)
    i_std = np.sqrt(variance)

    # ── Peak positions: single-pass maximum_position ──
    all_peak_pos = maximum_position(height, labels, index=label_ids)
    # maximum_position returns a single tuple when index is scalar
    if n == 1:
        all_peak_pos = [all_peak_pos]
    peak_y = np.array([p[0] for p in all_peak_pos], dtype=int)
    peak_x = np.array([p[1] for p in all_peak_pos], dtype=int)
    peak_h = height[peak_y, peak_x]

    peak_x_nm = peak_x.astype(float) * xp_nm
    peak_y_nm = peak_y.astype(float) * yp_nm

    # ── Peak-to-centroid distance (vectorised) ──
    p2c_dist = np.sqrt((peak_x_nm - cx_nm) ** 2 + (peak_y_nm - cy_nm) ** 2)

    # ── Build list of dicts from pre-computed arrays ──
    individual_grains = []
    for i in range(n):
        individual_grains.append({
            'grain_id':                int(label_ids[i]),
            'area_px':                 float(area_px[i]),
            'area_nm2':                float(area_nm2[i]),
            'diameter_nm':             float(diameter_nm[i]),
            'diameter_px':             float(diameter_px[i]),
            'equivalent_radius_nm':    float(equivalent_radius[i]),
            'volume_nm3':              float(all_volumes[i]),
            'centroid_x_nm':           float(cx_nm[i]),
            'centroid_y_nm':           float(cy_nm[i]),
            'centroid_x_px':           float(cent_x[i]),
            'centroid_y_px':           float(cent_y[i]),
            'height_centroid_nm':      float(centroid_h[i]),
            'peak_x_nm':              float(peak_x_nm[i]),
            'peak_y_nm':              float(peak_y_nm[i]),
            'peak_x_px':              float(peak_x[i]),
            'peak_y_px':              float(peak_y[i]),
            'height_peak_nm':          float(peak_h[i]),
            'peak_to_centroid_dist_nm': float(p2c_dist[i]),
            'height_mean_nm':          float(i_mean[i]),
            'height_std_nm':           float(i_std[i]),
            'height_min_nm':           float(i_min[i]),
            'height_max_nm':           float(i_max[i]),
            'major_axis_nm':           float(major_nm[i]),
            'minor_axis_nm':           float(minor_nm[i]),
            'aspect_ratio':            float(aspect_ratio[i]),
            'orientation_deg':         float(orientation_deg[i]),
            'perimeter_nm':            float(perimeter_nm_arr[i]),
            'perimeter_px':            float(perimeter_px_arr[i]),
            'eccentricity':            float(ecc[i]),
            'solidity':                float(sol[i]),
            'convex_area_nm2':         float(convex_area_nm2[i]),
        })

    return individual_grains


def _empty_statistics() -> Dict[str, Any]:
    """Return empty statistics dictionary"""
    return {
        'num_grains': 0,
        'mean_area_px': 0.0,
        'mean_area_nm2': 0.0,
        'std_area_px': 0.0,
        'std_area_nm2': 0.0,
        'min_area_px': 0.0,
        'max_area_px': 0.0,
        'min_area_nm2': 0.0,
        'max_area_nm2': 0.0,
        'mean_diameter_nm': 0.0,
        'std_diameter_nm': 0.0,
        'min_diameter_nm': 0.0,
        'max_diameter_nm': 0.0,
        'mean_diameter_px': 0.0,
        'std_diameter_px': 0.0,
        'mean_perimeter_nm': 0.0,
        'std_perimeter_nm': 0.0,
        'mean_perimeter_px': 0.0,
        'std_perimeter_px': 0.0,
        'mean_eccentricity': 0.0,
        'mean_solidity': 0.0,
        'mean_aspect_ratio': 0.0,
        'mean_height_nm': 0.0,
        'std_height_nm': 0.0,
        'min_height_nm': 0.0,
        'max_height_nm': 0.0,
        'mean_height_peak_nm': 0.0,
        'std_height_peak_nm': 0.0,
        'mean_height_centroid_nm': 0.0,
        'std_height_centroid_nm': 0.0,
        'mean_volume_nm3': 0.0,
        'std_volume_nm3': 0.0,
        'grain_density': 0.0,
        'area_fraction': 0.0,
        'areas_px': np.array([]),
        'areas_nm2': np.array([]),
        'diameters_nm': np.array([]),
        'diameters_px': np.array([]),
        'perimeters_nm': np.array([]),
        'perimeters_px': np.array([]),
        'eccentricities': np.array([]),
        'solidities': np.array([]),
        'aspect_ratios': np.array([]),
        'major_axis_nm': np.array([]),
        'minor_axis_nm': np.array([]),
        'orientations_rad': np.array([]),
    }

