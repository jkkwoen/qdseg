"""
Generate docs/demo.png — a side-by-side figure of a synthetic AFM image
and the rule_based segmentation result, suitable for the README.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
from pathlib import Path

from qdseg import segment_rule_based, calculate_grain_statistics


# ── 1. Synthetic AFM image ──────────────────────────────────────────────────

RNG = np.random.default_rng(42)

SIZE_PX   = 256          # image pixels
PIXEL_NM  = 2.0          # nm per pixel  →  512 × 512 nm scan
N_QD      = 60           # number of QDs
HEIGHT_NM = 8.0          # typical QD height (nm)
BG_NOISE  = 0.3          # background noise amplitude (nm)

height = np.zeros((SIZE_PX, SIZE_PX), dtype=np.float32)

# Random QD positions (avoid edges)
margin = 12
xs = RNG.integers(margin, SIZE_PX - margin, N_QD)
ys = RNG.integers(margin, SIZE_PX - margin, N_QD)

# Random QD sizes (sigma in pixels, ≈ 10–18 nm diameter)
sigmas = RNG.uniform(2.5, 4.5, N_QD)
peak_h = RNG.uniform(0.6, 1.0, N_QD) * HEIGHT_NM

yy, xx = np.mgrid[0:SIZE_PX, 0:SIZE_PX]

for x0, y0, sig, h in zip(xs, ys, sigmas, peak_h):
    height += h * np.exp(-((xx - x0)**2 + (yy - y0)**2) / (2 * sig**2))

# Add low-frequency background tilt + noise
height += np.linspace(0, 1.5, SIZE_PX)[np.newaxis, :] * np.ones((SIZE_PX, 1))
height += BG_NOISE * RNG.standard_normal((SIZE_PX, SIZE_PX)).astype(np.float32)
height = np.clip(height, 0, None)

meta = {
    "pixel_nm": (PIXEL_NM, PIXEL_NM),
    "scan_size_nm": (SIZE_PX * PIXEL_NM, SIZE_PX * PIXEL_NM),
}

# ── 2. Segmentation ─────────────────────────────────────────────────────────

labels = segment_rule_based(
    height, meta,
    gaussian_sigma=1.2,
    min_area_px=8,
    min_peak_separation_nm=15.0,
)
stats = calculate_grain_statistics(labels, height, meta)

# ── 3. Figure ───────────────────────────────────────────────────────────────

SCAN_NM = SIZE_PX * PIXEL_NM
extent  = [0, SCAN_NM, 0, SCAN_NM]

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
axes[0].set_title("AFM height map  (synthetic)", fontsize=11)
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
