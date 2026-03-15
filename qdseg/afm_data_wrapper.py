"""
AFMData wrapper class for XQD files.
"""

import numpy as np
from typing import Dict, List, Optional

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
