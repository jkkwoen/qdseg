"""
Analysis utilities for grain data.

High-level helpers: small-label filter, PDF report, and save_results.
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

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


def save_results(
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
