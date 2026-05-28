"""
AFMData wrapper class for XQD files.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .io import load_height_nm
from .corrections import AFMCorrections


class AFMData:
    """Load and correct AFM height data from an XQD file.

    Corrections can be chained:

    >>> data = AFMData("file.xqd")
    >>> data.first_correction().second_correction().third_correction()
    >>> data.align_rows(method='median')
    >>> data.flat_correction("line_by_line").baseline_correction("min_to_zero")
    >>> height = data.get_data()
    >>> meta   = data.get_meta()
    """

    def __init__(self, file_path: str):
        """
        Parameters
        ----------
        file_path : str
            Path to the XQD file.
        """
        self.file_path = file_path
        self.height_raw, self.meta = load_height_nm(file_path)
        self.current_data = self.height_raw.copy()
        self._corrector = AFMCorrections()
        self._labels: Optional[np.ndarray] = None

    # ── cropping ────────────────────────────────────────────────────────────

    def crop(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
        unit: str = "px",
    ) -> 'AFMData':
        """Crop the analysis area by pixel or nanometer coordinates.

        The range is half-open: ``x_min <= x < x_max`` and
        ``y_min <= y < y_max``.
        Pixel coordinates map to columns (x) and rows (y). Nanometer
        coordinates are converted using ``meta['pixel_nm']``. Omitted bounds
        default to the full current data range.

        Parameters
        ----------
        x_min, x_max, y_min, y_max : float, optional
            Crop bounds in pixels or nm, depending on ``unit``.
        unit : str
            ``"px"`` for pixel bounds or ``"nm"`` for nanometer bounds.

        Returns
        -------
        AFMData
            self, for method chaining.

        Examples
        --------
        >>> data.crop()  # full current range
        >>> data.crop(100, 300, 50, 250, unit="px")
        >>> data.crop(1000, 3000, 500, 2500, unit="nm")
        """
        x_start, x_stop, y_start, y_stop = self._resolve_crop_bounds(
            x_min, x_max, y_min, y_max, unit=unit
        )

        self.height_raw = self.height_raw[y_start:y_stop, x_start:x_stop].copy()
        self.current_data = self.current_data[y_start:y_stop, x_start:x_stop].copy()
        self.meta = self._cropped_meta(x_start, x_stop, y_start, y_stop)
        self._labels = None
        return self

    def crop_px(
        self,
        x_min: Optional[int] = None,
        x_max: Optional[int] = None,
        y_min: Optional[int] = None,
        y_max: Optional[int] = None,
    ) -> 'AFMData':
        """Crop by pixel coordinates."""
        return self.crop(x_min, x_max, y_min, y_max, unit="px")

    def crop_nm(
        self,
        x_min: Optional[float] = None,
        x_max: Optional[float] = None,
        y_min: Optional[float] = None,
        y_max: Optional[float] = None,
    ) -> 'AFMData':
        """Crop by nanometer coordinates."""
        return self.crop(x_min, x_max, y_min, y_max, unit="nm")

    def _resolve_crop_bounds(
        self,
        x_min: Optional[float],
        x_max: Optional[float],
        y_min: Optional[float],
        y_max: Optional[float],
        unit: str,
    ) -> Tuple[int, int, int, int]:
        unit = unit.lower()
        if unit not in {"px", "pixel", "pixels", "nm", "nanometer", "nanometers"}:
            raise ValueError("unit must be 'px' or 'nm'")

        if unit in {"nm", "nanometer", "nanometers"}:
            px_nm, py_nm = self.meta.get("pixel_nm", (None, None))
            if not px_nm or not py_nm:
                raise ValueError("meta['pixel_nm'] is required for nm cropping")
            scan_x_nm = self.current_data.shape[1] * float(px_nm)
            scan_y_nm = self.current_data.shape[0] * float(py_nm)
            x_min = 0.0 if x_min is None else x_min
            x_max = scan_x_nm if x_max is None else x_max
            y_min = 0.0 if y_min is None else y_min
            y_max = scan_y_nm if y_max is None else y_max
            if x_max <= x_min or y_max <= y_min:
                raise ValueError(
                    "crop bounds must satisfy x_max > x_min and y_max > y_min"
                )
            x_start = int(np.floor(x_min / float(px_nm)))
            x_stop = int(np.ceil(x_max / float(px_nm)))
            y_start = int(np.floor(y_min / float(py_nm)))
            y_stop = int(np.ceil(y_max / float(py_nm)))
        else:
            height, width = self.current_data.shape
            x_min = 0 if x_min is None else x_min
            x_max = width if x_max is None else x_max
            y_min = 0 if y_min is None else y_min
            y_max = height if y_max is None else y_max
            if x_max <= x_min or y_max <= y_min:
                raise ValueError(
                    "crop bounds must satisfy x_max > x_min and y_max > y_min"
                )
            if any(float(v) != int(v) for v in (x_min, x_max, y_min, y_max)):
                raise ValueError("pixel crop bounds must be integers")
            x_start, x_stop = int(x_min), int(x_max)
            y_start, y_stop = int(y_min), int(y_max)

        height, width = self.current_data.shape
        if x_start < 0 or y_start < 0 or x_stop > width or y_stop > height:
            raise ValueError(
                f"crop bounds exceed data shape: width={width}, height={height}"
            )

        if x_stop <= x_start or y_stop <= y_start:
            raise ValueError("crop bounds produce an empty array")

        return x_start, x_stop, y_start, y_stop

    def _cropped_meta(
        self,
        x_start: int,
        x_stop: int,
        y_start: int,
        y_stop: int,
    ) -> Dict:
        meta = dict(self.meta)
        width = x_stop - x_start
        height = y_stop - y_start
        px_nm, py_nm = meta.get("pixel_nm", (1.0, 1.0))
        px_nm = float(px_nm)
        py_nm = float(py_nm)
        origin_px = meta.get("crop_origin_px", (0, 0))
        origin_nm = meta.get("crop_origin_nm", (0.0, 0.0))
        abs_x_start = int(origin_px[0]) + x_start
        abs_y_start = int(origin_px[1]) + y_start
        abs_x_nm = float(origin_nm[0]) + x_start * px_nm
        abs_y_nm = float(origin_nm[1]) + y_start * py_nm

        meta.update({
            "xres": width,
            "yres": height,
            "scan_size_nm": (width * px_nm, height * py_nm),
            "scan_size_um": (width * px_nm / 1000.0, height * py_nm / 1000.0),
            "crop_origin_px": (abs_x_start, abs_y_start),
            "crop_origin_nm": (abs_x_nm, abs_y_nm),
            "crop_bounds_px": (
                abs_x_start,
                abs_x_start + width,
                abs_y_start,
                abs_y_start + height,
            ),
            "crop_bounds_nm": (
                abs_x_nm,
                abs_x_nm + width * px_nm,
                abs_y_nm,
                abs_y_nm + height * py_nm,
            ),
        })
        return meta

    # ── corrections ──────────────────────────────────────────────────────────

    def first_correction(self, method: str = 'polynomial') -> 'AFMData':
        """Remove a 1st-order (planar) tilt.

        Parameters
        ----------
        method : str
            'polynomial' (full 2-D fit, default) or 'simple' (separable fit).
        """
        self._corrector.set_method(method)
        self.current_data = self._corrector.correct_1st(self.current_data)
        return self

    def second_correction(self, method: str = 'polynomial') -> 'AFMData':
        """Remove a 2nd-order (quadratic) background.

        Parameters
        ----------
        method : str
            'polynomial' (full 2-D fit, default) or 'simple' (separable fit).
        """
        self._corrector.set_method(method)
        self.current_data = self._corrector.correct_2nd(self.current_data)
        return self

    def third_correction(self, method: str = 'polynomial') -> 'AFMData':
        """Remove a 3rd-order (cubic) background.

        Parameters
        ----------
        method : str
            'polynomial' (full 2-D fit, default) or 'simple' (separable fit).
        """
        self._corrector.set_method(method)
        self.current_data = self._corrector.correct_3rd(self.current_data)
        return self

    def align_rows(
        self,
        method: str = 'median',
        poly_degree: int = 0,
        trim_fraction: float = 0.0,
    ) -> 'AFMData':
        """Remove scan-line artefacts.  Apply *before* flat_correction.

        Parameters
        ----------
        method : str
            'median' (default), 'mean', 'polynomial',
            'median_difference', or 'trimmed_mean'.
        poly_degree : int
            Polynomial degree when method='polynomial' (default 0 = mean).
        trim_fraction : float
            Trim fraction when method='trimmed_mean' (0–0.5).
        """
        self._corrector.set_align_rows_method(method)
        self.current_data = self._corrector.align_rows(
            self.current_data,
            method=method,
            poly_degree=poly_degree,
            trim_fraction=trim_fraction,
        )
        return self

    def flat_correction(
        self,
        method: str = "line_by_line",
        mask: Optional[np.ndarray] = None,
    ) -> 'AFMData':
        """Subtract a slowly-varying background plane.

        Parameters
        ----------
        method : str
            'line_by_line' (default), 'global', or 'median'.
        mask : np.ndarray, optional
            If provided, pixels where ``mask == 0`` are treated as background.
        """
        self._corrector.set_flat_method(method)
        self.current_data = self._corrector.correct_flat(self.current_data, mask=mask)
        return self

    def baseline_correction(self, method: str = "min_to_zero") -> 'AFMData':
        """Shift the height baseline to zero.

        Parameters
        ----------
        method : str
            'min_to_zero' (default), 'mean_to_zero', or 'median_to_zero'.
        """
        self._corrector.set_baseline_method(method)
        self.current_data = self._corrector.correct_baseline(self.current_data)
        return self

    def reset(self) -> 'AFMData':
        """Discard all corrections and restore the raw loaded data.

        Returns
        -------
        AFMData
            self, for method chaining.
        """
        self.current_data = self.height_raw.copy()
        self._labels = None
        return self

    # ── segmentation & analysis ──────────────────────────────────────────────

    @property
    def labels(self) -> Optional[np.ndarray]:
        """Last segmentation result. None if segment() has not been called."""
        return self._labels

    def segment(self, method: str = "advanced", **kwargs) -> np.ndarray:
        """Segment grains and store the result internally.

        Parameters
        ----------
        method : str
            One of ``'advanced'`` (default), ``'watershed'``,
            ``'thresholding'``, ``'stardist'``, ``'cellpose'``.
        **kwargs
            Forwarded to the selected segmentation function.

        Returns
        -------
        np.ndarray
            Label image (int32, 0 = background, 1 … N = grain IDs).
            Also stored as ``self.labels`` for later access.
        """
        from .segmentation import segment as _segment
        self._labels = _segment(self.current_data, self.meta, method=method, **kwargs)
        return self._labels

    def stats(self) -> Dict:
        """Return overall grain statistics.

        Raises
        ------
        RuntimeError
            If ``segment()`` has not been called yet.
        """
        from .statistics import calculate_grain_statistics
        if self._labels is None:
            raise RuntimeError("Call segment() before stats().")
        return calculate_grain_statistics(self._labels, self.current_data, self.meta)

    def grains(self) -> List[Dict]:
        """Return per-grain measurements as a list of dicts.

        Raises
        ------
        RuntimeError
            If ``segment()`` has not been called yet.
        """
        from .statistics import get_individual_grains
        if self._labels is None:
            raise RuntimeError("Call segment() before grains().")
        return get_individual_grains(self._labels, self.current_data, self.meta)

    # ── accessors ────────────────────────────────────────────────────────────

    def get_data(self) -> np.ndarray:
        """Return the current (corrected) height array."""
        return self.current_data

    def get_raw_data(self) -> np.ndarray:
        """Return the original height array (before any corrections)."""
        return self.height_raw

    def get_meta(self) -> Dict:
        """Return file metadata (pixel size, scan size, z-scale, …)."""
        return self.meta
