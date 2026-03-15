"""
Generate docs/demo.png — a side-by-side figure of a real AFM image
and the rule_based segmentation result, suitable for the README.

Source: b2827_x-2_y-30_1um.xqd
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from pathlib import Path

from qdseg import AFMData, segment_rule_based, calculate_grain_statistics


# ── 1. Load and correct real XQD file ───────────────────────────────────────

XQD_PATH = Path.home() / "data_raw/afm_nanonavi/2017/20170215/b2827_x-2_y-30_1um.xqd"

data = AFMData(str(XQD_PATH))
data.first_correction().second_correction().third_correction()
data.align_rows(method='median')
data.flat_correction("line_by_line")
data.baseline_correction("min_to_zero")

height = data.get_data()
meta   = data.get_meta()

# ── 2. Segmentation ─────────────────────────────────────────────────────────

labels = segment_rule_based(
    height, meta,
    gaussian_sigma=1.2,
    min_area_px=8,
    min_peak_separation_nm=15.0,
)
stats = calculate_grain_statistics(labels, height, meta)

# ── 3. Figure ───────────────────────────────────────────────────────────────

pixel_nm  = meta.get("pixel_nm", (1.0, 1.0))
scan_size = meta.get("scan_size_nm", (height.shape[1] * pixel_nm[0], height.shape[0] * pixel_nm[1]))
PIXEL_NM  = pixel_nm[0]
SCAN_NM   = scan_size[0]
extent    = [0, scan_size[0], 0, scan_size[1]]

vmin = np.percentile(height, 1)
vmax = np.percentile(height, 99)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
fig.patch.set_facecolor("#1a1a2e")

for ax in axes:
    ax.set_facecolor("#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#555")
    ax.tick_params(colors="#aaa")
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    ax.title.set_color("#ddd")

# Left — raw height map
im = axes[0].imshow(
    height, cmap="afmhot", origin="lower", extent=extent,
    vmin=vmin, vmax=vmax, interpolation="bilinear",
)
axes[0].set_title("AFM height map", fontsize=11)
axes[0].set_xlabel("X  [nm]")
axes[0].set_ylabel("Y  [nm]")
cb = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
cb.set_label("Height  [nm]", color="#aaa")
cb.ax.yaxis.set_tick_params(color="#aaa")
plt.setp(cb.ax.yaxis.get_ticklabels(), color="#aaa")

# Right — segmentation overlay
axes[1].imshow(
    height, cmap="afmhot", origin="lower", extent=extent,
    vmin=vmin, vmax=vmax, interpolation="bilinear",
)

# Boundary overlay
if labels.max() > 0:
    bounds = find_boundaries(labels, mode="outer")
    by, bx = np.where(bounds)
    axes[1].scatter(
        bx * PIXEL_NM, by * PIXEL_NM,
        c="#3cc6f5", s=0.6, alpha=0.7, linewidths=0,
    )

n   = stats["num_grains"]
d   = stats["mean_diameter_nm"]
cov = stats["area_fraction"] * 100
axes[1].set_title(
    f"rule_based  |  N = {n}  |  Ø = {d:.1f} nm  |  cov = {cov:.1f} %",
    fontsize=10,
)
axes[1].set_xlabel("X  [nm]")
axes[1].set_ylabel("Y  [nm]")

fig.suptitle("QDSeg — quantum dot segmentation", color="#eee", fontsize=13, y=1.01)
plt.tight_layout()

out = Path("docs/demo.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved → {out}  ({n} grains detected)")
