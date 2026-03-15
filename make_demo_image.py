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
matplotlib.rcParams["font.family"] = "Helvetica Neue"
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
scan_um = height.shape[1] * px_nm / 1000.0
extent = [0, scan_um, 0, scan_um]

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

def paint_boundaries(rgb: np.ndarray, labels: np.ndarray,
                     convex_color=(46, 204, 46),
                     nonconvex_color=(230, 38, 38),
                     thickness: int = 1) -> np.ndarray:
    """Directly paint boundary pixels onto an RGB uint8 image. No blending."""
    from skimage.morphology import dilation, disk
    out = rgb.copy()
    props = regionprops(labels)
    solidity = {p.label: p.solidity for p in props}

    # Inner boundary: pixels are inside the grain → labels[y,x] gives own label
    bounds = find_boundaries(labels, mode="inner")

    # Assign color index on inner boundary pixels: 1=convex, 2=nonconvex
    # Inner boundary pixels always carry their own grain's label
    index = np.zeros(labels.shape, dtype=np.uint8)
    ys, xs = np.where(bounds)
    for y, x in zip(ys, xs):
        lbl = labels[y, x]
        sol = solidity.get(lbl, 1.0)
        index[y, x] = 1 if sol >= 0.95 else 2

    # Dilate index map — max dilation, single integer array, no channel mixing
    selem = disk(thickness)
    index = dilation(index, selem)

    # Paint: index>0 pixels get their assigned color
    out[index == 1] = convex_color
    out[index == 2] = nonconvex_color
    return out

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

# Convert height to uint8 RGB once
gray_norm = np.clip((height - vmin) / (vmax - vmin + 1e-10), 0, 1)
gray_uint8 = (gray_norm * 255).astype(np.uint8)
gray_rgb = np.stack([gray_uint8] * 3, axis=-1)  # H×W×3

fig, axes = plt.subplots(2, 3, figsize=(15, 11.0), facecolor="white",
                         gridspec_kw={"hspace": 0.28, "wspace": 0.12})

for ax, (title, labels) in zip(axes.flat, PANELS):
    if labels is not None:
        img = paint_boundaries(gray_rgb, labels)
    else:
        img = gray_rgb
    ax.imshow(img, origin="lower", extent=extent, interpolation="bilinear")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=5, linespacing=1.3)
    ax.set_xlabel("μm", fontsize=8)
    ax.set_ylabel("μm", fontsize=8)
    ax.tick_params(labelsize=7)

out = Path("docs/demo.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved → {out}")
