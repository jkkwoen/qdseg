"""
Main analysis function for grain analysis

High-level pipeline that combines segmentation and statistics.

사용 예시:
    >>> from grain_analyzer import analyze_grains
    >>> result = analyze_grains(height, meta, method="classical")
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

from .afm_data_wrapper import AFMData
from .segmentation import (
    segment_advanced,
    segment_watershed,
    segment_thresholding,
    segment_stardist,
    segment_cellpose,
)
from .statistics import calculate_grain_statistics, get_individual_grains
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries


def _filter_small_labels(labels: np.ndarray, min_area_px: int = 10) -> np.ndarray:
    """Remove label regions smaller than min_area_px pixels.

    Parameters
    ----------
    labels : np.ndarray
        Integer label image (0 = background).
    min_area_px : int
        Minimum region area in pixels. Regions strictly smaller than this
        value are set to 0 (background).

    Returns
    -------
    np.ndarray
        Label image with small regions removed (same dtype as input).
    """
    result = labels.copy()
    for prop in regionprops(labels):
        if prop.area < min_area_px:
            result[labels == prop.label] = 0
    return result


def _save_analysis_data(
    stats: Dict[str, Any],
    grains: List[Dict[str, Any]],
    stats_path: Path,
    grains_path: Path,
) -> None:
    """Save overall stats as JSON and per-grain data as CSV."""
    class _NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2, cls=_NpEncoder)

    if grains:
        grains_path.parent.mkdir(parents=True, exist_ok=True)
        with open(grains_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=grains[0].keys())
            writer.writeheader()
            writer.writerows(grains)

    print(f"   ✓ Data saved: {stats_path.name}, {grains_path.name}")


def analyze_grains(
    height: np.ndarray,
    meta: Optional[Dict] = None,
    *,
    method: str = "advanced",
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
        Segmentation method (default: 'advanced')
        - 'advanced': Otsu + Distance + DBSCAN + Voronoi (default, recommended)
        - 'watershed': Watershed segmentation
        - 'thresholding': Height threshold based
        - 'stardist': StarDist deep learning
        - 'cellpose': CellPose deep learning
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
    >>> result = analyze_grains(height, meta, method="advanced")
    >>> print(f"Found {result['stats']['num_grains']} grains")

    >>> # With watershed
    >>> result = analyze_grains(height, meta, method="watershed")

    >>> # With StarDist
    >>> result = analyze_grains(height, meta, method="stardist", prob_thresh=0.6)
    """
    method = method.lower()

    # Select segmentation method
    if method == "advanced":
        labels = segment_advanced(height, meta, **kwargs)
    elif method == "watershed":
        labels = segment_watershed(height, meta, **kwargs)
    elif method == "thresholding":
        labels = segment_thresholding(height, meta, **kwargs)
    elif method == "stardist":
        labels = segment_stardist(height, meta, **kwargs)
    elif method == "cellpose":
        labels = segment_cellpose(height, meta, **kwargs)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Supported methods: 'advanced', 'watershed', 'thresholding', "
            f"'stardist', 'cellpose'"
        )

    # Calculate statistics (get_individual_grains once, reuse in calculate_grain_statistics)
    grains = get_individual_grains(labels, height, meta)
    stats = calculate_grain_statistics(labels, height, meta, _precomputed_grains=grains)

    return {
        "labels": labels,
        "stats": stats,
        "grains": grains,
        "method": method,
    }


def analyze_single_file_with_grain_data(
    xqd_file: Path,
    output_dir: Path,
    method: str = "advanced",
    gaussian_sigma: float = 1.0,
    min_area_px: int = 10,
    min_peak_separation_nm: float = 10.0,
    save_pdf: bool = False,
    stats_path: Optional[Path] = None,
    grains_path: Optional[Path] = None,
    **kwargs
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Path]]:
    """
    Analyze a single XQD file and return grain data and statistics.

    Always saves analysis results:
      - ``{stem}_stats.json``  — overall statistics
      - ``{stem}_grains.csv``  — per-grain measurements

    By default both files are written to ``output_dir/{stem}/``.
    Use ``stats_path`` / ``grains_path`` to override individual file locations.

    Parameters
    ----------
    xqd_file : Path
        Path to the XQD file
    output_dir : Path
        Base output directory.  A subdirectory named after the file stem is
        created automatically unless ``stats_path`` / ``grains_path`` are given.
    method : str
        Segmentation method (default: 'advanced')
        - 'advanced': Otsu + Distance + DBSCAN + Voronoi (default, recommended)
        - 'watershed': Watershed segmentation
        - 'thresholding': Height threshold based
        - 'stardist': StarDist deep learning
        - 'cellpose': CellPose deep learning
    gaussian_sigma : float
        Gaussian smoothing sigma (for advanced and watershed methods)
    min_area_px : int
        Minimum grain area in pixels (default: 10, same as segment_advanced default)
    min_peak_separation_nm : float
        Minimum peak separation in nm (for advanced method)
    save_pdf : bool
        Whether to also generate a PDF report (default: False).
    stats_path : Path, optional
        Explicit destination for the stats JSON file.
        Overrides the default ``output_dir/{stem}/{stem}_stats.json``.
    grains_path : Path, optional
        Explicit destination for the grains CSV file.
        Overrides the default ``output_dir/{stem}/{stem}_grains.csv``.
    **kwargs
        Additional arguments for segmentation (method-specific)

    Returns
    -------
    Tuple[bool, Optional[Dict], Optional[Dict], Optional[Path]]
        (success, individual_grain_data, grain_stats, data_dir)
        data_dir is the directory where results were saved (None on failure)
    """
    print(f"📊 Processing: {xqd_file.name}")

    try:
        # Load data using AFMData
        data = AFMData(str(xqd_file))
        print(f"   ✓ Data loaded: {data.get_data().shape}")

        # Apply corrections
        print("   🔬 Applying corrections...")
        data.first_correction().second_correction().third_correction()
        data.flat_correction("line_by_line").baseline_correction("min_to_zero")

        # Get data
        height_corrected = data.get_data()
        meta = data.get_meta()
        height_raw = data.get_raw_data()

        x_size_nm, y_size_nm = meta.get("scan_size_nm", (height_raw.shape[1], height_raw.shape[0]))
        extent = [0, x_size_nm, 0, y_size_nm]

        method = method.lower()

        # Perform segmentation
        print(f"   🔬 Performing {method} grain segmentation...")

        if method == "advanced":
            labels = segment_advanced(
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
        else:
            raise ValueError(
                f"Unknown method: {method}. "
                f"Supported methods: 'advanced', 'watershed', 'thresholding', "
                f"'stardist', 'cellpose'"
            )
        
        num_grains = int(labels.max())
        print(f"   ✓ Segmentation completed: {num_grains} grains detected")

        # Calculate statistics (get_individual_grains once, reuse in calculate_grain_statistics)
        print("   📊 Calculating grain statistics...")
        individual_grain_data = get_individual_grains(labels, height_corrected, meta)
        grain_stats = calculate_grain_statistics(labels, height_corrected, meta, _precomputed_grains=individual_grain_data)

        # Save data
        file_output_dir = output_dir / xqd_file.stem
        file_output_dir.mkdir(parents=True, exist_ok=True)
        resolved_stats_path = stats_path or file_output_dir / f"{xqd_file.stem}_stats.json"
        resolved_grains_path = grains_path or file_output_dir / f"{xqd_file.stem}_grains.csv"
        _save_analysis_data(grain_stats, individual_grain_data, resolved_stats_path, resolved_grains_path)

        # Create PDF (optional)
        if save_pdf:
            print("   📄 Creating PDF plot...")
            grain_mask = labels > 0
            boundaries = find_boundaries(labels, mode="outer") if num_grains > 0 else np.zeros_like(labels, dtype=bool)
            _create_grain_analysis_pdf(
                height_raw, height_corrected, grain_mask, labels, boundaries,
                xqd_file.stem, file_output_dir, extent, num_grains, method
            )

        print(f"   ✅ Analysis completed for {xqd_file.name}")
        return True, individual_grain_data, grain_stats, file_output_dir

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
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_pdf import FigureCanvasPdf

    pdf_path = output_dir / f"{stem}_grain_analysis_{method}.pdf"

    fig = Figure(figsize=(16, 8))
    axes = fig.subplots(1, 2)

    # Left: Original height data
    vmin_raw, vmax_raw = np.percentile(height_raw, [2, 98])
    im1 = axes[0].imshow(height_raw, cmap='gray', origin='lower', extent=extent,
                         vmin=vmin_raw, vmax=vmax_raw)
    axes[0].set_xlabel('X [nm]')
    axes[0].set_ylabel('Y [nm]')
    axes[0].set_title('Original Height Data')
    fig.colorbar(im1, ax=axes[0], label='Height [nm]')

    # Right: Grain mask overlay
    vmin_corr, vmax_corr = np.percentile(height_corrected, [2, 98])
    axes[1].imshow(height_corrected, cmap='gray', origin='lower', extent=extent,
                   vmin=vmin_corr, vmax=vmax_corr)

    # Overlay grain labels with colormap
    if grain_labels.max() > 0:
        im3 = axes[1].imshow(grain_labels, cmap='tab20', origin='lower', extent=extent,
                             alpha=0.5, vmin=0, vmax=grain_labels.max())
        fig.colorbar(im3, ax=axes[1], label='Grain ID')

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
    fig.tight_layout()

    FigureCanvasPdf(fig).print_figure(str(pdf_path), dpi=300, bbox_inches='tight')

    print(f"   ✓ PDF saved: {pdf_path}")

    return pdf_path
