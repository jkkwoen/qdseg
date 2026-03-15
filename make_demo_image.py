"""
Generate docs/demo.png — 6-panel segmentation comparison for the README.

Panels  : Original | Rule-based ×3 | ML ×2
Layout  : 2 rows × 3 cols
Sources : b2823_x-2_y-2_1um.xqd  +  *_convex.pdf (5 files)
"""

import subprocess, tempfile, shutil
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from qdseg import AFMData

# ── Paths ─────────────────────────────────────────────────────────────────────

XQD = Path("/Users/jkkwoen/data_raw/afm_nanonavi/2017/20170209/b2823_x-2_y-2_1um.xqd")

PDF_BASE = Path(
    "/Users/jkkwoen/1_work/14_code/141_script/"
    "afm_pl_relation/output/afm_export"
)

PDF_PANELS = [
    ("Rule-based model\n(Thresholding)",   PDF_BASE / "b2823_x-2_y-2_1um_thresholding_convex.pdf"),
    ("Rule-based model\n(Watershed)",      PDF_BASE / "b2823_x-2_y-2_1um_watershed_convex.pdf"),
    ("Rule-based model\n(Advanced)",       PDF_BASE / "b2823_x-2_y-2_1um_classical_convex.pdf"),
    ("Machine learning model\n(StarDist)", PDF_BASE / "b2823_x-2_y-2_1um_stardist_convex.pdf"),
    ("Machine learning model\n(Cellpose)", PDF_BASE / "b2823_x-2_y-2_1um_cellpose_convex.pdf"),
]

# ── 1. Original AFM image (corrected, no segmentation) ───────────────────────

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

# ── 2. PDF → PNG (crop internal title) ───────────────────────────────────────

def pdf_to_img(pdf_path: Path, dpi: int = 180, crop_frac: float = 0.085) -> np.ndarray:
    tmp = Path(tempfile.mkdtemp())
    try:
        subprocess.run(
            ["pdftoppm", "-r", str(dpi), "-png", "-singlefile",
             str(pdf_path), str(tmp / "out")],
            check=True, capture_output=True,
        )
        img = mpimg.imread(str(tmp / "out.png"))
        h = img.shape[0]
        return img[int(h * crop_frac):, :]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

print("Converting PDFs …")
pdf_images = [(lbl, pdf_to_img(pdf)) for lbl, pdf in PDF_PANELS]

# ── 3. Figure: 2 rows × 3 cols ───────────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(15, 10.5), facecolor="white",
                         gridspec_kw={"hspace": 0.10, "wspace": 0.04})
fig.patch.set_facecolor("white")

# Panel 0 — Original
ax0 = axes.flat[0]
vmin, vmax = np.percentile(height, [1, 99])
ax0.imshow(height, cmap="gray", origin="lower", extent=extent,
           vmin=vmin, vmax=vmax, interpolation="bilinear")
ax0.set_title("Original", fontsize=12, fontweight="bold", pad=6)
ax0.set_xlabel("x (nm)", fontsize=8)
ax0.set_ylabel("y (nm)", fontsize=8)
ax0.tick_params(labelsize=7)

# Panels 1–5 — segmentation results
for ax, (label, img) in zip(list(axes.flat)[1:], pdf_images):
    ax.imshow(img)
    ax.set_title(label, fontsize=11, fontweight="bold", pad=5, linespacing=1.3)
    ax.axis("off")

# Legend (bottom of figure)
fig.text(
    0.5, 0.005,
    "  green = convex grain     red = non-convex grain     sample: b2823  |  1 × 1 µm  ",
    ha="center", va="bottom", fontsize=9, color="#555",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#ccc"),
)

out = Path("docs/demo.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {out}")
