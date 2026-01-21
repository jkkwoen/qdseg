"""
Grain Statistics Calculation

세그멘테이션 방법과 무관하게 label 이미지로부터 통계 계산

사용 예시:
    >>> from grain_analyzer.statistics import calculate_grain_statistics
    >>> stats = calculate_grain_statistics(labels, height, meta)
    >>> print(f"Number of grains: {stats['num_grains']}")
"""

import math
import numpy as np
from typing import Dict, Optional, List, Any
from skimage import measure


def calculate_grain_statistics(
    labels: np.ndarray,
    height: Optional[np.ndarray] = None,
    meta: Optional[Dict] = None,
    *,
    min_pixel_area: int = 4,
) -> Dict[str, Any]:
    """
    Calculate statistics from grain labels
    
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
    
    Returns
    -------
    stats : dict
        Grain statistics including:
        - num_grains: int
        - mean_area_nm2, std_area_nm2: float
        - mean_diameter_nm, std_diameter_nm: float
        - mean_perimeter_nm: float
        - mean_eccentricity, mean_solidity, mean_aspect_ratio: float
        - grain_density, area_fraction: float
        - areas_nm2, diameters_nm: np.ndarray (per-grain arrays)
    
    Examples
    --------
    >>> stats = calculate_grain_statistics(labels, height, meta)
    >>> print(f"Found {stats['num_grains']} grains")
    >>> print(f"Mean diameter: {stats['mean_diameter_nm']:.1f} nm")
    """
    if labels.max() == 0:
        return _empty_statistics()
    
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
        'min_area_nm2': float(np.min(areas_nm2)),
        'max_area_nm2': float(np.max(areas_nm2)),
        
        # Diameter statistics
        'mean_diameter_nm': float(np.mean(diameters_nm)),
        'std_diameter_nm': float(np.std(diameters_nm)),
        'min_diameter_nm': float(np.min(diameters_nm)),
        'max_diameter_nm': float(np.max(diameters_nm)),
        
        # Shape statistics
        'mean_perimeter_nm': float(np.mean(perimeters_nm)),
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
        'perimeters_nm': perimeters_nm,
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
        - area_nm2, diameter_nm, volume_nm3: float
        - centroid_x_nm, centroid_y_nm: float
        - peak_x_nm, peak_y_nm, peak_height_nm: float
        - height_mean_nm, height_std_nm, height_max_nm: float
        - major_axis_nm, minor_axis_nm, aspect_ratio: float
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
    
    # Get region properties
    props = measure.regionprops_table(
        labels,
        intensity_image=height,
        properties=[
            'label', 'area', 'centroid', 'perimeter',
            'major_axis_length', 'minor_axis_length', 'orientation',
            'eccentricity', 'solidity', 'convex_area'
        ]
    )
    
    individual_grains = []
    
    for idx in range(len(props['label'])):
        area_px = float(props['area'][idx])
        
        # Skip small grains
        if area_px < min_pixel_area:
            continue
        
        lab_id = int(props['label'][idx])
        mask = (labels == lab_id)
        
        if not np.any(mask):
            continue
        
        # Basic properties
        area_nm2 = area_px * px_area_nm2
        diameter_nm = 2.0 * math.sqrt(area_nm2 / math.pi)
        
        # Centroid
        cent_y = props['centroid-0'][idx]
        cent_x = props['centroid-1'][idx]
        cx_nm = float(cent_x) * xp_nm
        cy_nm = float(cent_y) * yp_nm
        
        # Peak location
        ys, xs = np.nonzero(mask)
        if ys.size > 0:
            vals = height[ys, xs]
            kmax = int(np.argmax(vals))
            pr, pc = int(ys[kmax]), int(xs[kmax])
        else:
            pr, pc = int(round(cent_y)), int(round(cent_x))
        
        peak_x_nm = float(pc) * xp_nm
        peak_y_nm = float(pr) * yp_nm
        peak_h_nm = float(height[pr, pc])
        
        # Volume
        vol_nm3 = float(np.sum(height[mask]) * px_area_nm2)
        
        # Shape properties
        major_axis_px = float(props['major_axis_length'][idx])
        minor_axis_px = float(props['minor_axis_length'][idx])
        major_axis_nm = major_axis_px * px_length_nm
        minor_axis_nm = minor_axis_px * px_length_nm
        aspect_ratio = major_axis_px / minor_axis_px if minor_axis_px > 0 else 1.0
        
        # Height statistics
        vals_mask = height[mask]
        
        grain_data = {
            'grain_id': lab_id,
            'area_px': area_px,
            'area_nm2': area_nm2,
            'diameter_nm': diameter_nm,
            'volume_nm3': vol_nm3,
            'centroid_x_nm': cx_nm,
            'centroid_y_nm': cy_nm,
            'peak_x_nm': peak_x_nm,
            'peak_y_nm': peak_y_nm,
            'peak_height_nm': peak_h_nm,
            'height_mean_nm': float(np.mean(vals_mask)),
            'height_std_nm': float(np.std(vals_mask)),
            'height_min_nm': float(np.min(vals_mask)),
            'height_max_nm': float(np.max(vals_mask)),
            'major_axis_nm': major_axis_nm,
            'minor_axis_nm': minor_axis_nm,
            'aspect_ratio': aspect_ratio,
            'orientation_deg': float(props['orientation'][idx]) * 180.0 / math.pi,
            'perimeter_nm': float(props['perimeter'][idx]) * px_length_nm,
            'eccentricity': float(props['eccentricity'][idx]),
            'solidity': float(props['solidity'][idx]),
            'convex_area_nm2': float(props['convex_area'][idx]) * px_area_nm2,
        }
        
        individual_grains.append(grain_data)
    
    return individual_grains


def _empty_statistics() -> Dict[str, Any]:
    """Return empty statistics dictionary"""
    return {
        'num_grains': 0,
        'mean_area_px': 0.0,
        'mean_area_nm2': 0.0,
        'std_area_px': 0.0,
        'std_area_nm2': 0.0,
        'min_area_nm2': 0.0,
        'max_area_nm2': 0.0,
        'mean_diameter_nm': 0.0,
        'std_diameter_nm': 0.0,
        'min_diameter_nm': 0.0,
        'max_diameter_nm': 0.0,
        'mean_perimeter_nm': 0.0,
        'mean_eccentricity': 0.0,
        'mean_solidity': 0.0,
        'mean_aspect_ratio': 0.0,
        'grain_density': 0.0,
        'area_fraction': 0.0,
        'areas_px': np.array([]),
        'areas_nm2': np.array([]),
        'diameters_nm': np.array([]),
        'perimeters_nm': np.array([]),
        'eccentricities': np.array([]),
        'solidities': np.array([]),
        'aspect_ratios': np.array([]),
        'major_axis_nm': np.array([]),
        'minor_axis_nm': np.array([]),
        'orientations_rad': np.array([]),
    }

