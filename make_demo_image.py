"""
Generate docs/demo.png — 6-panel segmentation comparison for the README.

All segmentation is computed fresh via qdseg (no pre-rendered PDFs).

Panels:
  [0] Original (corrected height map)
  [1] Rule-based model (Thresholding)
  [2] Rule-based model (Watershed)
  [3] Rule-based model (Advanced)
  [4] Machine learning model (StarDist)
  [5] Machine learning model (Cellpose)
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from skimage.segmentation import find_boundaries

from qdseg import (
    AFMData,
    segment_thresholding,
    segment_watershed,
    segment_rule_based,
    calculate_grain_statistics,
)

# ── 0. Load & correct XQD ────────────────────────────────────────────────────

XQD = Path("/Users/jkkwoen/data_raw/afm_nanonavi/2017/20170209/b2823_x-2_y-2_1um.xqd")

print("Loading XQD …")
data = AFMData(str(XQD))
data.first_correction().second_correction().third_correction()
data.align_rows(method='median')
data.flat_correction("line_by_line")
data.baseline_correction("min_to_zero")

height = data.get_data()
meta   = data.get_meta()
px_nm  = meta.get("pixel_nm", (1.0, 1.0))[0]
scan   = height.shape[1] * px_nm
extent = [0, scan, 0, scan]

# ── 1. Segmentation ───────────────────────────────────────────────────────────

print("Segmenting: thresholding …")
labels_thresh = segment_thresholding(height, meta)

print("Segmenting: watershed …")
labels_ws = segment_watershed(height, meta)

print("Segmenting: rule_based (advanced) …")
labels_rb = segment_rule_based(height, meta)

print("Segmenting: StarDist …")
from qdseg.utils import setup_gpu_environment
setup_gpu_environment()
from stardist.models import StarDist2D
from csbdeep.utils import normalize as sd_normalize
sd_model = StarDist2D.from_pretrained("2D_versatile_fluo")
img_norm = sd_normalize(height, 1, 99.8)
labels_sd, _ = sd_model.predict_instances(img_norm, prob_thresh=0.5, nms_thresh=0.4)
labels_sd = labels_sd.astype(np.int32)

print("Segmenting: Cellpose …")
from qdseg.utils import get_torch_device
from cellpose import models as cp_models
device = get_torch_device(verbose=False)
cp_model = cp_models.CellposeModel(
    model_type="cyto3", gpu=device.type != "cpu", device=device
)
pmin, pmax = np.percentile(height, [1, 99.8])
img_cp = np.clip((height - pmin) / (pmax - pmin + 1e-10), 0, 1) * 255
result = cp_model.eval(img_cp.astype(np.float32), diameter=None,
                       flow_threshold=0.4, cellprob_threshold=0.0)
labels_cp = result[0].astype(np.int32)

# ── 2. Solidity helper (green=convex, red=non-convex) ─────────────────────────

from skimage.measure import regionprops

def solidity_boundary_rgba(labels: np.ndarray,
                            convex_color=(0.18, 0.80, 0.18),
                            nonconvex_color=(0.90, 0.15, 0.15),
                            alpha: float = 0.9,
                            thickness: int = 1) -> np.ndarray:
    """Return RGBA overlay where grain boundaries are coloured by solidity."""
    from skimage.morphology import dilation, disk
    rgba = np.zeros((*labels.shape, 4), dtype=np.float32)
    props = regionprops(labels)
    solidity = {p.label: p.solidity for p in props}

    bounds = find_boundaries(labels, mode="outer")

    # Per-grain colour map on the boundary pixels
    color_r = np.zeros(labels.shape, dtype=np.float32)
    color_g = np.zeros(labels.shape, dtype=np.float32)
    color_b = np.zeros(labels.shape, dtype=np.float32)

    by, bx = np.where(bounds)
    for y, x in zip(by, bx):
        lbl = labels[y, x]
        if lbl == 0:
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y+dy, x+dx
                if 0 <= ny < labels.shape[0] and 0 <= nx < labels.shape[1]:
                    if labels[ny, nx] > 0:
                        lbl = labels[ny, nx]
                        break
        sol = solidity.get(lbl, 1.0)
        c = convex_color if sol >= 0.9 else nonconvex_color
        color_r[y, x] = c[0]
        color_g[y, x] = c[1]
        color_b[y, x] = c[2]

    # Dilate boundary mask and colour channels together
    selem = disk(thickness)
    thick_bounds = dilation(bounds, selem)
    color_r = dilation(color_r, selem)
    color_g = dilation(color_g, selem)
    color_b = dilation(color_b, selem)

    rgba[thick_bounds, 0] = color_r[thick_bounds]
    rgba[thick_bounds, 1] = color_g[thick_bounds]
    rgba[thick_bounds, 2] = color_b[thick_bounds]
    rgba[thick_bounds, 3] = alpha
    return rgba

# ── 3. Figure ─────────────────────────────────────────────────────────────────

PANELS = [
    ("Original",                          None),
    ("Rule-based model\n(Thresholding)",   labels_thresh),
    ("Rule-based model\n(Watershed)",      labels_ws),
    ("Rule-based model\n(Advanced)",       labels_rb),
    ("Machine learning model\n(StarDist)", labels_sd),
    ("Machine learning model\n(Cellpose)", labels_cp),
]

vmin, vmax = np.percentile(height, [1, 99])
norm = Normalize(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(2, 3, figsize=(15, 11.0), facecolor="white",
                         gridspec_kw={"hspace": 0.28, "wspace": 0.12})

for ax, (title, labels) in zip(axes.flat, PANELS):
    ax.imshow(height, cmap="gray", origin="lower", extent=extent,
              norm=norm, interpolation="bilinear")
    if labels is not None:
        overlay = solidity_boundary_rgba(labels)
        ax.imshow(overlay, origin="lower", extent=extent, interpolation="none")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5, linespacing=1.3)
    ax.set_xlabel("x (nm)", fontsize=8)
    ax.set_ylabel("y (nm)", fontsize=8)
    ax.tick_params(labelsize=7)

fig.text(
    0.5, 0.005,
    "  green = convex grain     red = non-convex grain     sample: b2823  |  1 × 1 µm  ",
    ha="center", va="bottom", fontsize=9, color="#555",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#ccc"),
)

out = Path("docs/demo.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {out}")
