"""
AFM Data Corrections
====================
Correction routines for AFM height data.

All polynomial surface fitting uses numpy.linalg.lstsq — no scikit-learn dependency.
"""

import numpy as np
from typing import List, Optional, Union
from scipy import stats


# ─── helpers ────────────────────────────────────────────────────────────────


def _poly_feature_matrix(xf: np.ndarray, yf: np.ndarray, degree: int) -> np.ndarray:
    """Build a 2-D polynomial feature matrix up to *degree*.

    For degree=1: columns are [1, x, y]
    For degree=2: columns are [1, x, y, x², xy, y²]
    For degree=3: columns are [1, x, y, x², xy, y², x³, x²y, xy², y³]
    """
    cols = [np.ones(xf.size, dtype=np.float64)]
    for d in range(1, degree + 1):
        for k in range(d + 1):
            cols.append((xf ** (d - k)) * (yf ** k))
    return np.column_stack(cols)


def _fit_poly_surface(height: np.ndarray, degree: int) -> np.ndarray:
    """Fit a 2-D polynomial surface of *degree* and return the fitted values."""
    rows, cols = height.shape
    y, x = np.mgrid[:rows, :cols]
    xf = x.ravel().astype(np.float64)
    yf = y.ravel().astype(np.float64)
    zf = height.ravel().astype(np.float64)
    A = _poly_feature_matrix(xf, yf, degree)
    coeffs, _, _, _ = np.linalg.lstsq(A, zf, rcond=None)
    return (A @ coeffs).reshape(height.shape)


class AFMCorrections:
    """AFM data correction class.

    Corrections provided:
    1. Polynomial tilt removal (1st / 2nd / 3rd order)
    2. Scan-line artefact removal (align_rows)
    3. Flat correction (line-by-line, global, or median-filter)
    4. Baseline correction (min / mean / median to zero)
    """

    def __init__(self):
        self.slope_method: str = 'polynomial'
        self.align_rows_method: str = 'median'
        self.flat_method: str = 'line_by_line'
        self.baseline_method: str = 'min_to_zero'
        self.filter_size: float = 0.1   # fraction of image size for median filter

    # ── setter helpers ───────────────────────────────────────────────────────

    def set_method(self, method: str):
        """Set slope-correction method: 'polynomial' or 'simple'."""
        if method not in ('polynomial', 'simple'):
            raise ValueError("method must be 'polynomial' or 'simple'")
        self.slope_method = method

    def set_flat_method(self, method: str):
        """Set flat-correction method: 'line_by_line', 'global', or 'median'."""
        if method not in ('line_by_line', 'global', 'median'):
            raise ValueError("flat_method must be 'line_by_line', 'global', or 'median'")
        self.flat_method = method

    def set_baseline_method(self, method: str):
        """Set baseline-correction method: 'min_to_zero', 'mean_to_zero', or 'median_to_zero'."""
        if method not in ('min_to_zero', 'mean_to_zero', 'median_to_zero'):
            raise ValueError(
                "baseline_method must be 'min_to_zero', 'mean_to_zero', or 'median_to_zero'"
            )
        self.baseline_method = method

    def set_align_rows_method(self, method: str):
        """Set scan-line artefact correction method.

        Options: 'median', 'mean', 'polynomial', 'median_difference', 'trimmed_mean'
        """
        valid = ('median', 'mean', 'polynomial', 'median_difference', 'trimmed_mean')
        if method not in valid:
            raise ValueError(f"align_rows_method must be one of {valid}")
        self.align_rows_method = method

    # ── input validation ─────────────────────────────────────────────────────

    def _check_input(self, height: np.ndarray):
        if not isinstance(height, np.ndarray):
            raise TypeError("height must be a numpy array")
        if height.ndim != 2:
            raise ValueError("height must be a 2-D array")
        if height.size == 0:
            raise ValueError("height array is empty")

    # ── public correction interface ──────────────────────────────────────────

    def correct_1st(self, height: np.ndarray) -> np.ndarray:
        """Remove a 1st-order (planar) trend."""
        self._check_input(height)
        if self.slope_method == 'polynomial':
            return height - _fit_poly_surface(height, 1)
        return self._correct_1st_simple(height)

    def correct_2nd(self, height: np.ndarray) -> np.ndarray:
        """Remove a 2nd-order (quadratic) trend."""
        self._check_input(height)
        if self.slope_method == 'polynomial':
            return height - _fit_poly_surface(height, 2)
        return self._correct_2nd_simple(height)

    def correct_3rd(self, height: np.ndarray) -> np.ndarray:
        """Remove a 3rd-order (cubic) trend."""
        self._check_input(height)
        if self.slope_method == 'polynomial':
            return height - _fit_poly_surface(height, 3)
        return self._correct_3rd_simple(height)

    def align_rows(
        self,
        height: np.ndarray,
        method: Optional[str] = None,
        poly_degree: int = 0,
        trim_fraction: float = 0.0,
    ) -> np.ndarray:
        """Remove scan-line artefacts (align rows).

        Parameters
        ----------
        height : np.ndarray
        method : str, optional
            Override ``self.align_rows_method``.
            Choices: 'median', 'mean', 'polynomial', 'median_difference',
            'trimmed_mean'.
        poly_degree : int
            Polynomial degree when method='polynomial' (default 0 = mean).
        trim_fraction : float
            Trim fraction when method='trimmed_mean' (0–0.5).
        """
        self._check_input(height)
        method = method or self.align_rows_method
        dispatch = {
            'median':           self._align_rows_median,
            'mean':             self._align_rows_mean,
            'polynomial':       lambda h: self._align_rows_polynomial(h, degree=poly_degree),
            'median_difference': self._align_rows_median_difference,
            'trimmed_mean':     lambda h: self._align_rows_trimmed_mean(h, trim_fraction=trim_fraction),
        }
        if method not in dispatch:
            raise ValueError(f"Unknown align_rows method: {method}")
        return dispatch[method](height)

    def correct_flat(
        self, height: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Apply flat correction.

        Parameters
        ----------
        height : np.ndarray
        mask : np.ndarray, optional
            If provided, pixels where ``mask == 0`` are treated as background
            and used to estimate the trend.
        """
        self._check_input(height)
        if self.flat_method == 'line_by_line':
            return self._correct_flat_line_by_line(height, mask)
        if self.flat_method == 'global':
            return self._correct_flat_global(height, mask)
        return self._correct_flat_median(height, mask)

    def correct_baseline(
        self, height: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Shift the baseline so that background pixels are near zero.

        Parameters
        ----------
        mask : np.ndarray, optional
            If provided, ``mask == 0`` pixels define the background.
        """
        self._check_input(height)
        background = height[mask == 0] if mask is not None and np.any(mask == 0) else height
        if self.baseline_method == 'min_to_zero':
            return height - background.min()
        if self.baseline_method == 'mean_to_zero':
            return height - background.mean()
        return height - np.median(background)   # median_to_zero

    # ── private: polynomial (numpy-based, no sklearn) ────────────────────────

    def _correct_1st_simple(self, height: np.ndarray) -> np.ndarray:
        """Separate-axis 1st-order simple correction."""
        return self._correct_separable_poly(height, degree=1)

    def _correct_2nd_simple(self, height: np.ndarray) -> np.ndarray:
        """Separable row-then-column 2nd-order correction."""
        return self._correct_separable_poly(height, degree=2)

    def _correct_3rd_simple(self, height: np.ndarray) -> np.ndarray:
        """Separable row-then-column 3rd-order correction."""
        return self._correct_separable_poly(height, degree=3)

    @staticmethod
    def _correct_separable_poly(height: np.ndarray, degree: int) -> np.ndarray:
        """Fit and subtract a separable degree-*n* polynomial (row then column).

        Each row is fitted independently with a degree-n polynomial; the
        residuals are then fitted column-by-column with the same degree.
        This is faster than a full 2-D fit and adequate for mild background
        curvature.
        """
        rows, cols = height.shape
        x = np.arange(cols, dtype=np.float64)
        y = np.arange(rows, dtype=np.float64)
        result = height.copy().astype(np.float64)

        for i in range(rows):
            result[i, :] -= np.polyval(np.polyfit(x, result[i, :], degree), x)

        for j in range(cols):
            result[:, j] -= np.polyval(np.polyfit(y, result[:, j], degree), y)

        return result.astype(height.dtype)

    # ── private: flat correction ─────────────────────────────────────────────

    def _correct_flat_line_by_line(
        self, height: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Subtract a per-row linear fit (using background pixels only if mask given)."""
        result = height.copy()
        rows, cols = height.shape
        x = np.arange(cols, dtype=np.float64)

        for i in range(rows):
            line = height[i, :]
            if mask is not None:
                bg = mask[i, :] == 0
                if np.sum(bg) < 2:
                    continue
                coeffs = np.polyfit(x[bg], line[bg], 1)
            else:
                coeffs = np.polyfit(x, line, 1)
            result[i, :] = line - np.polyval(coeffs, x)

        return result

    def _correct_flat_global(
        self, height: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fit and subtract a global plane (using background pixels only if mask given)."""
        rows, cols = height.shape
        y, x = np.mgrid[:rows, :cols]

        if mask is not None:
            bg = mask == 0
            if np.sum(bg) < 3:
                bg = np.ones_like(mask, dtype=bool)
            xs = x[bg].astype(np.float64)
            ys = y[bg].astype(np.float64)
            zs = height[bg].astype(np.float64)
        else:
            xs = x.ravel().astype(np.float64)
            ys = y.ravel().astype(np.float64)
            zs = height.ravel().astype(np.float64)

        A_fit = np.column_stack([np.ones(xs.size), xs, ys])
        coeffs, _, _, _ = np.linalg.lstsq(A_fit, zs, rcond=None)

        A_all = np.column_stack([
            np.ones(rows * cols),
            x.ravel().astype(np.float64),
            y.ravel().astype(np.float64),
        ])
        fitted = (A_all @ coeffs).reshape(height.shape)
        return height - fitted

    def _correct_flat_median(
        self, height: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Subtract a median-filtered background estimate."""
        from scipy import ndimage

        size = (
            max(1, int(min(height.shape) * self.filter_size))
            if isinstance(self.filter_size, float)
            else self.filter_size
        )
        flat_ref = ndimage.median_filter(height, size=size)
        return height - flat_ref

    # ── private: align-rows helpers ──────────────────────────────────────────

    def _align_rows_median(self, height: np.ndarray) -> np.ndarray:
        return height - np.median(height, axis=1, keepdims=True)

    def _align_rows_mean(self, height: np.ndarray) -> np.ndarray:
        return height - np.mean(height, axis=1, keepdims=True)

    def _align_rows_polynomial(self, height: np.ndarray, degree: int = 0) -> np.ndarray:
        result = height.copy()
        x = np.arange(height.shape[1], dtype=np.float64)
        for i in range(height.shape[0]):
            coeffs = np.polyfit(x, height[i, :], degree)
            result[i, :] -= np.polyval(coeffs, x)
        return result

    def _align_rows_median_difference(self, height: np.ndarray) -> np.ndarray:
        result = height.copy()
        for i in range(1, height.shape[0]):
            diff = height[i, :] - height[i - 1, :]
            result[i, :] = result[i - 1, :] + (diff - np.median(diff))
        return result

    def _align_rows_trimmed_mean(
        self, height: np.ndarray, trim_fraction: float = 0.1
    ) -> np.ndarray:
        if not 0 <= trim_fraction <= 0.5:
            raise ValueError("trim_fraction must be between 0 and 0.5")
        result = height.copy()
        for i in range(height.shape[0]):
            result[i, :] -= stats.trim_mean(height[i, :], trim_fraction)
        return result
