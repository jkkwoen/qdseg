"""
Main analysis function for grain analysis

High-level pipeline that combines segmentation and statistics.

사용 예시:
    >>> from grain_analyzer import analyze_grains
    >>> result = analyze_grains(height, meta, method="classical")
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from .afm_data_wrapper import AFMData
from .segmentation import (
    segment_rule_based,
    segment_watershed,
    segment_thresholding,
    segment_stardist,
    segment_cellpose,
    segment_cellulus,
)
from .statistics import calculate_grain_statistics, get_individual_grains
from skimage.segmentation import find_boundaries


def analyze_grains(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    method: str = "rule_based",
    **kwargs
) -> Dict[str, Any]:
    """
    Analyze grains in height image

    Parameters
    ----------
    height : np.ndarray
        Height image (2D, nm units)
    meta : dict, optional
        Metadata with 'pixel_nm' key
    method : str
        Segmentation method (default: 'rule_based')
        - 'rule_based': Otsu + Distance + DBSCAN + Voronoi (default, recommended)
        - 'watershed': Watershed segmentation
        - 'thresholding': Height threshold based
        - 'stardist': StarDist deep learning
        - 'cellpose': CellPose deep learning
        - 'cellulus': Cellulus deep learning
    **kwargs
        Additional arguments passed to segmentation function

    Returns
    -------
    result : dict
        Analysis result containing:
        - labels: np.ndarray - Label image
        - stats: dict - Overall statistics
        - grains: List[dict] - Individual grain data
        - method: str - Method used

    Examples
    --------
    >>> result = analyze_grains(height, meta, method="rule_based")
    >>> print(f"Found {result['stats']['num_grains']} grains")

    >>> # With watershed
    >>> result = analyze_grains(height, meta, method="watershed")

    >>> # With StarDist
    >>> result = analyze_grains(height, meta, method="stardist", prob_thresh=0.6)
    """
    # Method mapping for backward compatibility
    method_map = {
        'classical': 'rule_based',
        'advanced_watershed': 'rule_based',
        'simple_watershed': 'watershed',
    }
    method = method_map.get(method.lower(), method.lower())

    # Select segmentation method
    if method == "rule_based":
        labels = segment_rule_based(height, meta, **kwargs)
    elif method == "watershed":
        labels = segment_watershed(height, meta, **kwargs)
    elif method == "thresholding":
        labels = segment_thresholding(height, meta, **kwargs)
    elif method == "stardist":
        labels = segment_stardist(height, meta, **kwargs)
    elif method == "cellpose":
        labels = segment_cellpose(height, meta, **kwargs)
    elif method == "cellulus":
        labels = segment_cellulus(height, meta, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Supported methods: 'rule_based', 'watershed', 'thresholding', "
            f"'stardist', 'cellpose', 'cellulus'"
        )

    # Calculate statistics
    stats = calculate_grain_statistics(labels, height, meta)
    grains = get_individual_grains(labels, height, meta)

    return {
        "labels": labels,
        "stats": stats,
        "grains": grains,
        "method": method,
    }


def analyze_single_file_with_grain_data(
    xqd_file: Path,
    output_dir: Path,
    method: str = "rule_based",
    gaussian_sigma: float = 1.0,
    min_area_nm2: float = 78.5,
    min_peak_separation_nm: float = 10.0,
    **kwargs
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
    """
    Analyze a single XQD file and return grain data and statistics.

    Parameters
    ----------
    xqd_file : Path
        Path to the XQD file
    output_dir : Path
        Output directory for PDF file
    method : str
        Segmentation method (default: 'rule_based')
        - 'rule_based': Otsu + Distance + DBSCAN + Voronoi (default, recommended)
        - 'watershed': Watershed segmentation
        - 'thresholding': Height threshold based
        - 'stardist': StarDist deep learning
        - 'cellpose': CellPose deep learning
        - 'cellulus': Cellulus deep learning
    gaussian_sigma : float
        Gaussian smoothing sigma (for rule_based and watershed methods)
    min_area_nm2 : float
        Minimum grain area in nm²
    min_peak_separation_nm : float
        Minimum peak separation in nm (for rule_based method)
    **kwargs
        Additional arguments for segmentation (method-specific)

    Returns
    -------
    Tuple[bool, Optional[Dict], Optional[Dict], Optional[str]]
        (success, individual_grain_data, grain_stats, pdf_path)
    """
    from .utils import nm2_to_px_area
    
    print(f"📊 Processing: {xqd_file.name}")
    
    try:
        # Load data using AFMData
        data = AFMData(str(xqd_file))
        print(f"   ✓ Data loaded: {data.get_data().shape}")
        
        # Create output directory for this file
        file_output_dir = output_dir / xqd_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply corrections
        print("   🔬 Applying corrections...")
        data.first_correction().second_correction().third_correction()
        data.flat_correction("line_by_line").baseline_correction("min_to_zero")
        
        # Get data
        height_corrected = data.get_data()
        meta = data.get_meta()
        height_raw = data.get_raw_data()
        
        pixel_nm = meta.get("pixel_nm", (1.0, 1.0))
        x_size_nm, y_size_nm = meta.get("scan_size_nm", (height_raw.shape[1], height_raw.shape[0]))
        extent = [0, x_size_nm, 0, y_size_nm]
        
        # Calculate min_area_px (for classical method)
        min_area_px = nm2_to_px_area(min_area_nm2, pixel_nm)
        
        # Method mapping for backward compatibility
        method_map = {
            'classical': 'rule_based',
            'advanced_watershed': 'rule_based',
            'simple_watershed': 'watershed',
        }
        method = method_map.get(method.lower(), method.lower())

        # Perform segmentation
        print(f"   🔬 Performing {method} grain segmentation...")

        if method == "rule_based":
            labels = segment_rule_based(
                height_corrected,
                meta,
                gaussian_sigma=gaussian_sigma,
                min_area_px=min_area_px,
                min_peak_separation_nm=min_peak_separation_nm,
            )
        elif method == "watershed":
            labels = segment_watershed(
                height_corrected,
                meta,
                gaussian_sigma=gaussian_sigma,
                min_area_px=min_area_px,
            )
        elif method == "thresholding":
            labels = segment_thresholding(
                height_corrected,
                meta,
                min_area_px=min_area_px,
            )
        elif method == "stardist":
            labels = segment_stardist(height_corrected, meta, **kwargs)
        elif method == "cellpose":
            labels = segment_cellpose(height_corrected, meta, **kwargs)
        elif method == "cellulus":
            labels = segment_cellulus(height_corrected, meta, **kwargs)
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported methods: 'rule_based', 'watershed', 'thresholding', "
                f"'stardist', 'cellpose', 'cellulus'"
            )
        
        num_grains = int(labels.max())
        print(f"   ✓ Segmentation completed: {num_grains} grains detected")
        
        # Calculate statistics
        print("   📊 Calculating grain statistics...")
        grain_stats = calculate_grain_statistics(labels, height_corrected, meta)
        individual_grain_data = get_individual_grains(labels, height_corrected, meta)
        
        # Create PDF
        print("   📄 Creating PDF plot...")
        grain_mask = labels > 0
        boundaries = find_boundaries(labels, mode="outer") if num_grains > 0 else np.zeros_like(labels, dtype=bool)
        
        pdf_path = _create_grain_analysis_pdf(
            height_raw, height_corrected, grain_mask, labels, boundaries,
            xqd_file.stem, file_output_dir, extent, num_grains, method
        )
        
        print(f"   ✅ Analysis completed for {xqd_file.name}")
        return True, individual_grain_data, grain_stats, str(pdf_path)
        
    except Exception as e:
        print(f"   ❌ Error processing {xqd_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None, None


def _create_grain_analysis_pdf(
    height_raw: np.ndarray,
    height_corrected: np.ndarray,
    grain_mask: np.ndarray,
    grain_labels: np.ndarray,
    boundaries: np.ndarray,
    stem: str,
    output_dir: Path,
    extent: List[float],
    num_grains: int,
    method: str = "classical"
) -> Path:
    """Create PDF with original and grain_mask plots."""
    
    pdf_path = output_dir / f"{stem}_grain_analysis_{method}.pdf"
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Original height data
    im1 = axes[0].imshow(height_raw, cmap='gray', origin='lower', extent=extent, 
                         vmin=np.percentile(height_raw, 2), 
                         vmax=np.percentile(height_raw, 98))
    axes[0].set_xlabel('X [nm]')
    axes[0].set_ylabel('Y [nm]')
    axes[0].set_title('Original Height Data')
    plt.colorbar(im1, ax=axes[0], label='Height [nm]')
    
    # Right: Grain mask overlay
    im2 = axes[1].imshow(height_corrected, cmap='gray', origin='lower', extent=extent,
                         vmin=np.percentile(height_corrected, 2),
                         vmax=np.percentile(height_corrected, 98))
    
    # Overlay grain labels with colormap
    if grain_labels.max() > 0:
        im3 = axes[1].imshow(grain_labels, cmap='tab20', origin='lower', extent=extent, 
                            alpha=0.5, vmin=0, vmax=grain_labels.max())
        plt.colorbar(im3, ax=axes[1], label='Grain ID')
    
    # Draw boundaries
    if np.any(boundaries):
        y_coords, x_coords = np.where(boundaries)
        if extent:
            x_nm = x_coords * (extent[1] / height_corrected.shape[1])
            y_nm = y_coords * (extent[3] / height_corrected.shape[0])
            axes[1].scatter(x_nm, y_nm, c='red', s=0.1, alpha=0.6)
        else:
            axes[1].scatter(x_coords, y_coords, c='red', s=0.1, alpha=0.6)
    
    axes[1].set_xlabel('X [nm]')
    axes[1].set_ylabel('Y [nm]')
    axes[1].set_title(f'Grain Mask Overlay ({method}, N={num_grains})')
    
    fig.suptitle(f'Grain Analysis - {stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(pdf_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ PDF saved: {pdf_path}")
    
    return pdf_path
