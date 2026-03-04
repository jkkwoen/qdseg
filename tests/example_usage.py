"""
Grain Analyzer Usage Example

This file demonstrates the main features of the grain_analyzer package.
It compares four segmentation methods: Rule-based, StarDist, Cellpose, and Cellulus.

Segmentation methods:
- Rule-based: Otsu + Distance Transform + Voronoi (no labels required)
- StarDist: Star-convex polygon detection (pretrained model)
- Cellpose: Gradient flow-based segmentation (pretrained model)
- Cellulus: Object-centric embeddings (unsupervised, requires pre-training)

How to run:
    python -m tests.example_usage

    Or from project root:
    python tests/example_usage.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg (save to file without GUI)
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional


# ============================================================
# Rule-based + StarDist + Cellpose comparative analysis
# ============================================================

def analyze_all_files_all_methods():
    """
    Analyze all XQD files in the input_data folder
    using four methods: Rule-based, StarDist, Cellpose, and Cellulus
    """
    print("=" * 80)
    print("🔬 Grain Analysis: Rule-based vs StarDist vs Cellpose vs Cellulus Comparison")
    print("=" * 80)
    
    from qdseg import (
        AFMData, 
        segment_rule_based, 
        calculate_grain_statistics,
    )
    
    # Find all XQD files
    data_dir = Path(__file__).parent / "input_data" / "xqd"
    xqd_files = sorted(data_dir.glob("*.xqd"))
    
    if not xqd_files:
        print("ERROR: No XQD files found.")
        print(f"   Path checked: {data_dir}")
        return
    
    print(f"✓ {len(xqd_files)} XQD files found\n")
    
    # Store results
    all_results: List[Dict[str, Any]] = []
    
    # Pre-load models
    stardist_model = _load_stardist_model()
    cellpose_model = _load_cellpose_model()
    cellulus_model = _load_cellulus_model()
    
    # Analyze each file
    for i, xqd_file in enumerate(xqd_files, 1):
        print(f"\n[{i}/{len(xqd_files)}] 📂 {xqd_file.name}")
        print("-" * 70)
        
        try:
            # Load data and apply corrections
            data = AFMData(str(xqd_file))
            data.first_correction().second_correction().third_correction()
            data.align_rows(method='median')  # Scan line artefacts correction (before flat correction)
            
            # Generate mask with Classical Segmentation
            # height_for_mask = data.get_data()
            # labels_mask = segment_rule_based(
            #     height_for_mask, 
            #     meta={'pixel_nm': data.get_meta().get('pixel_nm', (1.0, 1.0))},
            #     gaussian_sigma=1.0,
            #     min_area_px=10
            # )
            # num_grains_mask = labels_mask.max()
            # print(f"   Mask created: {num_grains_mask} grains detected")

            # Flat correction (excluding mask)
            data.flat_correction("line_by_line")
            
            # Baseline correction (entire image)
            data.baseline_correction("min_to_zero")
            
            height = data.get_data()
            meta = data.get_meta()
            height_raw = data.get_raw_data()
            
            pixel_nm = meta.get("pixel_nm", (1.0, 1.0))
            scan_size = meta.get("scan_size_nm", (height.shape[1], height.shape[0]))
            
            print(f"   Image: {height.shape}, Scan: {scan_size[0]:.0f}x{scan_size[1]:.0f} nm")
            
            result = {
                'file': xqd_file.name,
                'stem': xqd_file.stem,
                'shape': height.shape,  # Image resolution (H, W)
            }
            
            # === 1. Rule-based segmentation ===
            print("   🔸 Rule-based segmentation...")
            labels_classical = segment_rule_based(
                height, meta,
                gaussian_sigma=1.0,
                min_area_px=10,
                min_peak_separation_nm=20.0,
            )
            stats_classical = calculate_grain_statistics(labels_classical, height, meta)
            
            result['rule_based'] = {
                'labels': labels_classical,
                'num_grains': stats_classical['num_grains'],
                'mean_diameter': stats_classical['mean_diameter_nm'],
                'coverage': stats_classical['area_fraction'] * 100,
            }
            print(f"      → {stats_classical['num_grains']} grains, "
                  f"Ø={stats_classical['mean_diameter_nm']:.1f}nm, "
                  f"cov={stats_classical['area_fraction']*100:.1f}%")
            
            # === 2. StarDist segmentation ===
            if stardist_model is not None:
                print("   🔹 StarDist segmentation...")
                try:
                    labels_stardist = _segment_with_stardist(stardist_model, height)
                    stats_stardist = calculate_grain_statistics(labels_stardist, height, meta)
                    
                    result['stardist'] = {
                        'labels': labels_stardist,
                        'num_grains': stats_stardist['num_grains'],
                        'mean_diameter': stats_stardist['mean_diameter_nm'],
                        'coverage': stats_stardist['area_fraction'] * 100,
                    }
                    print(f"      → {stats_stardist['num_grains']} grains, "
                          f"Ø={stats_stardist['mean_diameter_nm']:.1f}nm, "
                          f"cov={stats_stardist['area_fraction']*100:.1f}%")
                except Exception as e:
                    print(f"      FAILED: {e}")
                    result['stardist'] = None
            else:
                result['stardist'] = None
            
            # === 3. Cellpose segmentation ===
            if cellpose_model is not None:
                print("   🔶 Cellpose segmentation...")
                try:
                    labels_cellpose = _segment_with_cellpose(cellpose_model, height)
                    stats_cellpose = calculate_grain_statistics(labels_cellpose, height, meta)
                    
                    result['cellpose'] = {
                        'labels': labels_cellpose,
                        'num_grains': stats_cellpose['num_grains'],
                        'mean_diameter': stats_cellpose['mean_diameter_nm'],
                        'coverage': stats_cellpose['area_fraction'] * 100,
                    }
                    print(f"      → {stats_cellpose['num_grains']} grains, "
                          f"Ø={stats_cellpose['mean_diameter_nm']:.1f}nm, "
                          f"cov={stats_cellpose['area_fraction']*100:.1f}%")
                except Exception as e:
                    print(f"      FAILED: {e}")
                    result['cellpose'] = None
            else:
                result['cellpose'] = None
            
            # === 4. Cellulus segmentation ===
            if cellulus_model is not None:
                print("   🔷 Cellulus segmentation...")
                try:
                    labels_cellulus = _segment_with_cellulus(cellulus_model, height)
                    stats_cellulus = calculate_grain_statistics(labels_cellulus, height, meta)
                    
                    result['cellulus'] = {
                        'labels': labels_cellulus,
                        'num_grains': stats_cellulus['num_grains'],
                        'mean_diameter': stats_cellulus['mean_diameter_nm'],
                        'coverage': stats_cellulus['area_fraction'] * 100,
                    }
                    print(f"      → {stats_cellulus['num_grains']} grains, "
                          f"Ø={stats_cellulus['mean_diameter_nm']:.1f}nm, "
                          f"cov={stats_cellulus['area_fraction']*100:.1f}%")
                except Exception as e:
                    print(f"      FAILED: {e}")
                    result['cellulus'] = None
            else:
                result['cellulus'] = None
            
            # Save visualization
            output_dir = Path(__file__).parent / "output_data" / "local"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            _save_comparison_figure_4methods(
                height_raw, height, result,
                xqd_file.stem, output_dir,
                scan_size, pixel_nm
            )
            
            all_results.append(result)
            
        except Exception as e:
            print(f"   ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Summary of results
    _print_summary_4methods(all_results)
    
    return all_results


def _load_stardist_model():
    """Load StarDist model (auto-detect GPU)"""
    print("Loading StarDist model...")
    try:
        from qdseg.utils import setup_gpu_environment, check_tensorflow_gpu
        
        # Optimize GPU environment
        setup_gpu_environment()
        
        # Check TensorFlow GPU status
        gpu_available, msg = check_tensorflow_gpu(verbose=True)
        
        from stardist.models import StarDist2D
        model = StarDist2D.from_pretrained("2D_versatile_fluo")
        print("   ✓ StarDist model loaded")
        return model
    except Exception as e:
        print(f"   Warning: StarDist loading failed: {e}")
        return None


def _load_cellpose_model():
    """Load Cellpose model (auto-detect GPU: CUDA/MPS/CPU)"""
    print("Loading Cellpose model...")
    try:
        from qdseg.utils import setup_gpu_environment, get_torch_device
        from cellpose import models
        
        # Optimize GPU environment
        setup_gpu_environment()
        
        # Auto-detect optimal device
        device = get_torch_device(verbose=True)
        use_gpu = device.type != 'cpu'
        
        # Cellpose 4.x uses CellposeModel instead of Cellpose
        model = models.CellposeModel(
            model_type='cyto3',
            gpu=use_gpu,
            device=device,
        )
        print("   ✓ Cellpose model loaded")
        return model
    except Exception as e:
        print(f"   Warning: Cellpose loading failed: {e}")
        return None


def _load_cellulus_model():
    """
    Load Cellulus model (based on official Cellulus library)

    Uses the official Cellulus model trained on XQD data.

    How to train:
        python grain_analyzer/train_model.py --data-dir tests/input_data/xqd
    """
    print("Checking Cellulus model...")
    
    # 1. XQD trained model (priority)
    xqd_model_dir = Path(__file__).parent / "model_data" / "cellulus_official_xqd" / "models"
    xqd_model_path = xqd_model_dir / "009999.pth"
    
    if xqd_model_path.exists():
        print(f"   ✓ XQD trained model found: {xqd_model_path.name}")
        return {
            'checkpoint_path': str(xqd_model_path), 
            'model_dir': str(xqd_model_dir.parent),
            'use_official': True
        }
    
    # 2. Existing trained model
    model_dir = Path(__file__).parent / "model_data" / "cellulus"
    checkpoint_path = model_dir / "best_loss.pth"
    
    if checkpoint_path.exists():
        print(f"   ✓ Trained model found: {checkpoint_path.name}")
        return {'checkpoint_path': str(checkpoint_path), 'model_dir': str(model_dir), 'use_official': True}
    
    # No model found
    print("   Warning: Cellulus: no trained model found")
    print("      Train model: python grain_analyzer/train_model.py")
    print("      Reference: https://github.com/funkelab/cellulus")
    return None


def _segment_with_cellulus(model_config: dict, height: np.ndarray) -> np.ndarray:
    """Segmentation using Cellulus model (official Cellulus)"""
    from qdseg import segment_cellulus
    
    labels = segment_cellulus(
        height,
        checkpoint_path=model_config.get('checkpoint_path'),
        use_official=model_config.get('use_official', True),
        normalize_input=True,
        gpu=True,
    )
    return labels


def _segment_with_stardist(model, height: np.ndarray) -> np.ndarray:
    """Segmentation using StarDist model"""
    from csbdeep.utils import normalize
    img_norm = normalize(height, 1, 99.8)
    labels, _ = model.predict_instances(img_norm, prob_thresh=0.5, nms_thresh=0.4)
    return labels.astype(np.int32)


def _segment_with_cellpose(model, height: np.ndarray) -> np.ndarray:
    """Segmentation using Cellpose model"""
    # Normalize to 0-255 range
    pmin, pmax = np.percentile(height, [1, 99.8])
    if pmax - pmin < 1e-10:
        img_norm = np.zeros_like(height)
    else:
        img_norm = np.clip((height - pmin) / (pmax - pmin), 0, 1) * 255
    
    # Cellpose 4.x returns (masks, flows, styles) - 3 values
    result = model.eval(
        img_norm.astype(np.float32),
        diameter=None,  # auto-estimate
        flow_threshold=0.4,
        cellprob_threshold=0.0,
    )
    # result[0] is masks
    masks = result[0]
    return masks.astype(np.int32)


def _save_comparison_figure_4methods(
    height_raw: np.ndarray,
    height: np.ndarray,
    result: Dict,
    stem: str,
    output_dir: Path,
    scan_size: tuple,
    pixel_nm: tuple
):
    """Save comparison visualization for 4 methods"""
    from skimage.segmentation import find_boundaries
    
    extent = [0, scan_size[0], 0, scan_size[1]]
    
    # Boundary color
    BOUNDARY_COLOR = '#3071F6'
    
    # 2x3 layout (Reference, Rule-based, StarDist, Cellpose, Cellulus, Summary)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Reference, Rule-based, StarDist
    # 1. Reference (Corrected height) - grayscale, no colorbar
    axes[0, 0].imshow(height, cmap='gray', origin='lower', extent=extent,
                      vmin=np.percentile(height, 2), vmax=np.percentile(height, 98))
    axes[0, 0].set_title('Reference')
    axes[0, 0].set_xlabel('X [nm]')
    axes[0, 0].set_ylabel('Y [nm]')
    
    # 2. Rule-based
    classical = result.get('rule_based', {})
    _plot_segmentation(
        axes[0, 1], height, 
        classical.get('labels'),
        extent, pixel_nm,
        f"Rule-based (N={classical.get('num_grains', 0)}, Ø={classical.get('mean_diameter', 0):.1f}nm)",
        boundary_color=BOUNDARY_COLOR
    )
    
    # 3. StarDist
    stardist = result.get('stardist')
    if stardist:
        _plot_segmentation(
            axes[0, 2], height,
            stardist.get('labels'),
            extent, pixel_nm,
            f"StarDist (N={stardist.get('num_grains', 0)}, Ø={stardist.get('mean_diameter', 0):.1f}nm)",
            boundary_color=BOUNDARY_COLOR
        )
    else:
        _plot_not_available(axes[0, 2], height, extent, 'StarDist')
    
    # Row 2: Cellpose, Cellulus, Summary
    # 4. Cellpose
    cellpose = result.get('cellpose')
    if cellpose:
        _plot_segmentation(
            axes[1, 0], height,
            cellpose.get('labels'),
            extent, pixel_nm,
            f"Cellpose (N={cellpose.get('num_grains', 0)}, Ø={cellpose.get('mean_diameter', 0):.1f}nm)",
            boundary_color=BOUNDARY_COLOR
        )
    else:
        _plot_not_available(axes[1, 0], height, extent, 'Cellpose')
    
    # 5. Cellulus
    cellulus = result.get('cellulus')
    if cellulus:
        _plot_segmentation(
            axes[1, 1], height,
            cellulus.get('labels'),
            extent, pixel_nm,
            f"Cellulus (N={cellulus.get('num_grains', 0)}, Ø={cellulus.get('mean_diameter', 0):.1f}nm)",
            boundary_color=BOUNDARY_COLOR
        )
    else:
        _plot_not_available(axes[1, 1], height, extent, 'Cellulus')
    
    # 6. Summary table
    axes[1, 2].axis('off')
    
    # Create summary table
    methods = ['Rule-based', 'StarDist', 'Cellpose', 'Cellulus']
    data_keys = ['rule_based', 'stardist', 'cellpose', 'cellulus']
    
    table_data = []
    for method, key in zip(methods, data_keys):
        d = result.get(key)
        if d:
            table_data.append([
                method,
                f"{d['num_grains']}",
                f"{d['mean_diameter']:.1f}",
                f"{d['coverage']:.1f}%"
            ])
        else:
            table_data.append([method, 'N/A', 'N/A', 'N/A'])
    
    table = axes[1, 2].table(
        cellText=table_data,
        colLabels=['Method', 'Grains', 'Ø (nm)', 'Coverage'],
        loc='center',
        cellLoc='center',
        colColours=['#f0f0f0'] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    axes[1, 2].set_title('Comparison Summary', fontsize=14, fontweight='bold', pad=20)
    
    fig.suptitle(f'Grain Analysis Comparison: {stem}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save both PNG and PDF
    output_path_png = output_dir / f"{stem}_comparison.png"
    output_path_pdf = output_dir / f"{stem}_comparison.pdf"
    plt.savefig(output_path_png, dpi=150)
    plt.savefig(output_path_pdf, dpi=150)
    plt.close()
    
    print(f"   ✓ Saved: {output_path_png.name}, {output_path_pdf.name}")


def _plot_not_available(ax, height, extent, title):
    """N/A plot"""
    ax.imshow(height, cmap='gray', origin='lower', extent=extent,
              vmin=np.percentile(height, 2), vmax=np.percentile(height, 98))
    ax.text(0.5, 0.5, f'{title}\nNot Available', ha='center', va='center',
            transform=ax.transAxes, fontsize=14, color='white',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
    ax.set_title(title)
    ax.set_xlabel('X [nm]')
    ax.set_ylabel('Y [nm]')


def _plot_segmentation(ax, height, labels, extent, pixel_nm, title, boundary_color='#3071F6'):
    """Plot segmentation result (boundaries + darkened background)"""
    from skimage.segmentation import find_boundaries
    
    # Grayscale background image
    ax.imshow(height, cmap='gray', origin='lower', extent=extent,
              vmin=np.percentile(height, 2), vmax=np.percentile(height, 98))
    
    if labels is not None and labels.max() > 0:
        # Background (outside boundaries) #E96D68 color 25% opacity
        background_mask = labels == 0
        bg_rgba = np.zeros((*height.shape, 4))
        bg_rgba[background_mask, 0] = 0xE9 / 255  # R
        bg_rgba[background_mask, 1] = 0x6D / 255  # G
        bg_rgba[background_mask, 2] = 0x68 / 255  # B
        bg_rgba[background_mask, 3] = 0.25  # 25% opacity
        ax.imshow(bg_rgba, origin='lower', extent=extent)
        
        # Boundaries (50% opacity)
        boundaries = find_boundaries(labels, mode='outer')
        by, bx = np.where(boundaries)
        ax.scatter(bx * pixel_nm[0], by * pixel_nm[1], c=boundary_color, s=0.5, alpha=0.5)
    
    ax.set_title(title)
    ax.set_xlabel('X [nm]')
    ax.set_ylabel('Y [nm]')


def _print_summary_4methods(results: List[Dict]):
    """Print summary of results for 4 methods"""
    print("\n" + "=" * 110)
    print("Analysis Results Summary (Rule-based vs StarDist vs Cellpose vs Cellulus)")
    print("=" * 110)
    
    # Header
    methods = ['Rule-based', 'StarDist', 'Cellpose', 'Cellulus']
    keys = ['rule_based', 'stardist', 'cellpose', 'cellulus']
    
    header1 = f"{'Filename':<26} {'Resolution':>7}"
    header2 = f"{'':26} {'':>7}"
    for m in methods:
        header1 += f" {m:>14}"
        header2 += f" {'N':>4} {'Ø':>4} {'%':>4}"
    print(f"\n{header1}")
    print(header2)
    print("-" * 110)
    
    totals = {k: 0 for k in keys}
    diameters = {k: [] for k in keys}
    coverages = {k: [] for k in keys}
    
    for r in results:
        # Display resolution
        shape = r.get('shape', (0, 0))
        res_str = f"{shape[0]}x{shape[1]}" if shape[0] > 0 else "N/A"
        row = f"{r['file']:<26} {res_str:>7}"
        
        for key in keys:
            d = r.get(key)
            if d:
                row += f" {d['num_grains']:>4} {d['mean_diameter']:>4.0f} {d['coverage']:>4.0f}"
                totals[key] += d['num_grains']
                if d['mean_diameter'] > 0:
                    diameters[key].append(d['mean_diameter'])
                coverages[key].append(d['coverage'])
            else:
                row += f" {'N/A':>4} {'N/A':>4} {'N/A':>4}"
        
        print(row)
    
    # Total/Average
    print("-" * 110)
    row = f"{'Total/Average':<26} {'':>7}"
    for key in keys:
        if diameters[key]:
            avg_d = np.mean(diameters[key])
            avg_c = np.mean(coverages[key])
            row += f" {totals[key]:>4} {avg_d:>4.0f} {avg_c:>4.0f}"
        else:
            row += f" {'N/A':>4} {'N/A':>4} {'N/A':>4}"
    print(row)
    
    print("\nNote:")
    print("   - StarDist/Cellpose pretrained models are optimized for fluorescence/cell images.")
    print("   - Cellulus is unsupervised and requires a pre-trained model.")
    print("   - Custom training may be needed for AFM grain images.")


# ============================================================
# Main function
# ============================================================

def main():
    """Main execution"""
    print("\n" + "🔬 Grain Analyzer - Rule-based vs StarDist vs Cellpose vs Cellulus 🔬".center(100))
    print("=" * 100)
    
    # Compare four methods on all files
    results = analyze_all_files_all_methods()
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
