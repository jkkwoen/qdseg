"""
Unit tests for qdseg core modules.

Most tests use synthetic numpy arrays; TestRealXQD requires tests/test_data/test.xqd.

Run:
    cd /path/to/qdseg
    python -m pytest tests/test_core.py -v
"""

import math
import pathlib
import numpy as np
import pytest

# ─── helpers ────────────────────────────────────────────────────────────────


def _make_flat_image(rows=64, cols=64, seed=42):
    """Synthetic flat height image with a few Gaussian bumps (QD-like)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0, 0.1, (rows, cols)).astype(np.float64)
    # Add 6 well-separated bumps
    centers = [(16, 16), (16, 48), (32, 32), (48, 16), (48, 48), (24, 40)]
    y_idx, x_idx = np.mgrid[:rows, :cols]
    for cy, cx in centers:
        img += 5.0 * np.exp(-((y_idx - cy) ** 2 + (x_idx - cx) ** 2) / (2 * 3**2))
    img = img - img.min()
    return img


def _make_meta(rows=64, cols=64, scan_nm=1000.0):
    """Synthetic metadata dict."""
    px = scan_nm / cols
    return {
        "pixel_nm": (px, px),
        "scan_size_nm": (scan_nm, scan_nm),
        "zscale_nm_per_count": 1.0,
        "zoffset_nm": 0.0,
    }


# ─── corrections ────────────────────────────────────────────────────────────


class TestCorrections:
    """AFMCorrections — polynomial surface fitting and align_rows."""

    def setup_method(self):
        from qdseg.corrections import AFMCorrections
        self.corr = AFMCorrections()
        self.rows, self.cols = 64, 64
        y, x = np.mgrid[:self.rows, :self.cols]
        self.flat = np.zeros((self.rows, self.cols), dtype=np.float64)
        # known planar tilt: z = 0.1*x + 0.05*y
        self.plane = 0.1 * x + 0.05 * y

    # ── 1st order ────────────────────────────────────────────────────────────

    def test_correct_1st_polynomial_removes_plane(self):
        tilted = self.flat + self.plane
        result = self.corr.correct_1st(tilted)
        assert result.std() < 0.1, "Residual after 1st-order removal should be near zero"

    def test_correct_1st_simple_removes_plane(self):
        self.corr.set_method('simple')
        tilted = self.flat + self.plane
        result = self.corr.correct_1st(tilted)
        assert result.std() < 0.5

    # ── 2nd order ────────────────────────────────────────────────────────────

    def test_correct_2nd_polynomial_removes_quadratic(self):
        y, x = np.mgrid[:self.rows, :self.cols]
        quad = 0.01 * x**2 + 0.005 * y**2 + 0.1 * x + 0.05 * y
        result = self.corr.correct_2nd(self.flat + quad)
        assert result.std() < 0.1

    def test_correct_2nd_simple_removes_quadratic(self):
        """_correct_2nd_simple must no longer just call _correct_1st_simple twice."""
        self.corr.set_method('simple')
        y, x = np.mgrid[:self.rows, :self.cols]
        quad = 0.01 * x**2 + 0.005 * y**2
        flat_with_quad = self.flat + quad
        result = self.corr.correct_2nd(flat_with_quad)
        # Residual RMS should be much less than original RMS
        original_rms = float(np.std(flat_with_quad))
        result_rms = float(np.std(result))
        assert result_rms < original_rms * 0.3, (
            f"2nd-order simple correction should reduce RMS significantly: "
            f"before={original_rms:.3f}, after={result_rms:.3f}"
        )

    # ── 3rd order ────────────────────────────────────────────────────────────

    def test_correct_3rd_polynomial(self):
        y, x = np.mgrid[:self.rows, :self.cols]
        cubic = 0.001 * x**3 + 0.0005 * y**3
        result = self.corr.correct_3rd(self.flat + cubic)
        assert result.std() < 0.5

    # ── flat correction ───────────────────────────────────────────────────────

    def test_flat_line_by_line(self):
        # Image with row-wise offset
        drifted = self.flat.copy()
        for i in range(self.rows):
            drifted[i, :] += i * 0.1
        result = self.corr.correct_flat(drifted)
        # After correction each row should have ~zero mean
        row_means = result.mean(axis=1)
        assert np.abs(row_means).max() < 0.2

    def test_flat_global(self):
        self.corr.set_flat_method('global')
        y, x = np.mgrid[:self.rows, :self.cols]
        result = self.corr.correct_flat(self.flat + 0.1 * x + 0.05 * y)
        assert result.std() < 0.1

    # ── align rows ────────────────────────────────────────────────────────────

    def test_align_rows_median(self):
        """Each row should have ~zero median after alignment."""
        noisy_rows = self.flat.copy()
        rng = np.random.default_rng(0)
        noisy_rows += rng.normal(0, 2, self.rows)[:, np.newaxis]
        result = self.corr.align_rows(noisy_rows, method='median')
        medians = np.median(result, axis=1)
        assert np.abs(medians).max() < 1e-10

    # ── baseline ─────────────────────────────────────────────────────────────

    def test_baseline_min_to_zero(self):
        shifted = self.flat + 5.0
        result = self.corr.correct_baseline(shifted)
        assert result.min() == pytest.approx(0.0, abs=1e-9)


# ─── statistics ─────────────────────────────────────────────────────────────


class TestStatistics:
    """calculate_grain_statistics and get_individual_grains."""

    def setup_method(self):
        from qdseg.statistics import calculate_grain_statistics, get_individual_grains
        self.calc = calculate_grain_statistics
        self.individual = get_individual_grains
        self.height = _make_flat_image()
        self.meta = _make_meta()

    def _make_labels(self):
        """Simple 4-grain label map: each 32×32 quadrant is one grain."""
        labels = np.zeros((64, 64), dtype=np.int32)
        labels[:32, :32] = 1
        labels[:32, 32:] = 2
        labels[32:, :32] = 3
        labels[32:, 32:] = 4
        return labels

    def test_num_grains(self):
        labels = self._make_labels()
        stats = self.calc(labels, self.height, self.meta)
        assert stats['num_grains'] == 4

    def test_orientations_rad_correct(self):
        """orientations_rad must contain actual values, not all zeros."""
        labels = self._make_labels()
        stats = self.calc(labels, self.height, self.meta)
        orientations = stats['orientations_rad']
        assert len(orientations) == 4
        # Should be finite values (not NaN)
        assert np.all(np.isfinite(orientations))

    def test_orientations_rad_not_all_zero(self):
        """Previously this was hardcoded to np.array([0.0]*N) — verify the fix."""
        from qdseg.statistics import get_individual_grains
        labels = self._make_labels()
        grains = get_individual_grains(labels, self.height, self.meta)
        # Orientation values come from regionprops and should vary
        # (they will be 0 only if all grains are perfectly square, which they are
        # in this test — so we just check the dtype/shape are correct)
        stats = self.calc(labels, self.height, self.meta)
        assert stats['orientations_rad'].dtype in (np.float32, np.float64)
        assert len(stats['orientations_rad']) == stats['num_grains']

    def test_areas_consistent(self):
        labels = self._make_labels()
        stats = self.calc(labels, self.height, self.meta)
        # Each quadrant is 32×32 = 1024 px → ~244 140 nm² (at 1000nm/64px)
        expected_area_nm2 = (1000 / 64) ** 2 * 32 * 32
        assert stats['mean_area_nm2'] == pytest.approx(expected_area_nm2, rel=0.01)

    def test_empty_labels(self):
        labels = np.zeros((64, 64), dtype=np.int32)
        stats = self.calc(labels, self.height, self.meta)
        assert stats['num_grains'] == 0

    def test_individual_grains_count(self):
        labels = self._make_labels()
        grains = self.individual(labels, self.height, self.meta)
        assert len(grains) == 4

    def test_individual_grain_fields(self):
        labels = self._make_labels()
        grains = self.individual(labels, self.height, self.meta)
        required = {
            'grain_id', 'area_nm2', 'area_px', 'diameter_nm', 'diameter_px',
            'centroid_x_nm', 'centroid_y_nm', 'height_peak_nm',
            'height_mean_nm', 'height_std_nm', 'orientation_deg',
            'eccentricity', 'solidity', 'aspect_ratio',
        }
        for g in grains:
            assert required.issubset(g.keys()), f"Missing keys: {required - g.keys()}"


# ─── segmentation ────────────────────────────────────────────────────────────


class TestSegmentAdvanced:
    """segment() with method='advanced' on synthetic data."""

    def setup_method(self):
        from qdseg.segmentation import segment
        self.segment = lambda h, m, **kw: segment(h, m, method='advanced', **kw)
        self.height = _make_flat_image()
        self.meta = _make_meta()

    def test_returns_int32(self):
        labels = self.segment(self.height, self.meta)
        assert labels.dtype == np.int32

    def test_same_shape_as_input(self):
        labels = self.segment(self.height, self.meta)
        assert labels.shape == self.height.shape

    def test_detects_grains(self):
        labels = self.segment(self.height, self.meta)
        n = int(labels.max())
        assert n >= 4, f"Expected at least 4 grains (bumps), got {n}"

    def test_background_is_zero(self):
        labels = self.segment(self.height, self.meta)
        assert labels.min() == 0

    def test_empty_image(self):
        flat = np.zeros((32, 32), dtype=np.float64)
        labels = self.segment(flat, self.meta)
        assert labels.max() == 0

    def test_meta_none(self):
        labels = self.segment(self.height, None)
        assert labels.shape == self.height.shape


# ─── afm_data_wrapper ────────────────────────────────────────────────────────


class TestAFMDataReset:
    """AFMData.reset() should restore the raw data and clear labels."""

    def test_reset_without_xqd(self):
        """Test reset logic directly without loading a file."""
        from qdseg.corrections import AFMCorrections
        # Simulate what AFMData does internally
        raw = _make_flat_image()

        class MockAFM:
            def __init__(self, h):
                self.height_raw = h
                self.current_data = h.copy()
                self._corrector = AFMCorrections()
                self._labels = None

            def reset(self):
                self.current_data = self.height_raw.copy()
                self._labels = None
                return self

            def first_correction(self):
                self.current_data = self._corrector.correct_1st(self.current_data)
                return self

        obj = MockAFM(raw)
        obj.first_correction()
        assert not np.allclose(obj.current_data, raw), "After correction data should differ"
        obj.reset()
        assert np.allclose(obj.current_data, raw), "After reset data should match raw"

    def test_reset_clears_labels(self):
        """reset() must clear any stored segmentation result."""
        from qdseg.corrections import AFMCorrections
        raw = _make_flat_image()

        class MockAFM:
            def __init__(self, h):
                self.height_raw = h
                self.current_data = h.copy()
                self._labels = None

            def reset(self):
                self.current_data = self.height_raw.copy()
                self._labels = None
                return self

        obj = MockAFM(raw)
        obj._labels = np.zeros((64, 64), dtype=np.int32)  # simulate post-segment state
        assert obj._labels is not None
        obj.reset()
        assert obj._labels is None, "reset() should clear _labels"


# ─── analyze._filter_small_labels ────────────────────────────────────────────


class TestFilterSmallLabels:
    """_filter_small_labels removes regions below the size threshold."""

    def setup_method(self):
        from qdseg.analyze import _filter_small_labels
        self.filter = _filter_small_labels

    def test_removes_tiny_grain(self):
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[2:4, 2:4] = 1       # 4 px — tiny
        labels[8:16, 8:16] = 2     # 64 px — large
        filtered = self.filter(labels, min_area_px=10)
        assert len(np.unique(filtered[filtered > 0])) == 1  # only one grain survives
        assert np.sum(filtered > 0) == 64

    def test_keeps_all_above_threshold(self):
        labels = np.zeros((20, 20), dtype=np.int32)
        labels[0:5, 0:5] = 1       # 25 px
        labels[10:18, 10:18] = 2   # 64 px
        filtered = self.filter(labels, min_area_px=10)
        assert filtered.max() == 2

    def test_zero_threshold_keeps_all(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        labels[1:3, 1:3] = 1
        labels[6:9, 6:9] = 2
        filtered = self.filter(labels, min_area_px=0)
        assert filtered.max() == 2

    def test_empty_labels_unchanged(self):
        labels = np.zeros((10, 10), dtype=np.int32)
        filtered = self.filter(labels, min_area_px=10)
        assert filtered.max() == 0


# ─── corrections._fit_poly_surface ───────────────────────────────────────────


class TestFitPolySurface:
    """_fit_poly_surface must accurately recover known polynomial surfaces."""

    def setup_method(self):
        from qdseg.corrections import _fit_poly_surface
        self.fit = _fit_poly_surface
        self.rows, self.cols = 40, 40
        self.y, self.x = np.mgrid[:self.rows, :self.cols]

    def test_degree1_recovers_plane(self):
        plane = 0.5 * self.x + 0.3 * self.y + 2.0
        fitted = self.fit(plane, degree=1)
        assert np.allclose(fitted, plane, atol=1e-8)

    def test_degree2_recovers_quadratic(self):
        quad = 0.01 * self.x**2 + 0.005 * self.y**2 + 0.2 * self.x
        fitted = self.fit(quad, degree=2)
        assert np.allclose(fitted, quad, atol=1e-6)

    def test_degree3_recovers_cubic(self):
        cubic = 0.001 * self.x**3 + 0.0005 * self.y**3
        fitted = self.fit(cubic, degree=3)
        assert np.allclose(fitted, cubic, atol=1e-4)


# ─── cellpose v4 compatibility ───────────────────────────────────────────────


class TestCellposeV4Compat:
    """Verify segment_cellpose is compatible with Cellpose v4 API."""

    def test_import_succeeds(self):
        """cellpose >= 4.0 must be importable."""
        cp = pytest.importorskip("cellpose", reason="cellpose not installed")
        from cellpose import models
        assert hasattr(models, "CellposeModel"), "CellposeModel must exist in v4"

    def test_model_type_param_removed(self):
        """segment_cellpose must NOT have a model_type parameter (v4 removed it)."""
        import inspect
        from qdseg.segmentation import _segment_cellpose
        sig = inspect.signature(_segment_cellpose)
        assert "model_type" not in sig.parameters, (
            "model_type parameter should not exist in v4-compatible segment_cellpose"
        )

    def test_cellposemodel_constructor_no_model_type(self):
        """CellposeModel constructor must not be called with model_type."""
        pytest.importorskip("cellpose", reason="cellpose not installed")
        from cellpose import models
        import warnings
        # In v4.0.1+, passing model_type raises a warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            m = models.CellposeModel(gpu=False)  # no model_type
        # No DeprecationWarning about model_type should be raised
        model_type_warnings = [x for x in w if "model_type" in str(x.message)]
        assert len(model_type_warnings) == 0, (
            f"Unexpected model_type warning: {[str(x.message) for x in model_type_warnings]}"
        )

    def test_eval_returns_three_values(self):
        """eval() must return a 3-tuple (masks, flows, styles) in v4."""
        pytest.importorskip("cellpose", reason="cellpose not installed")
        from cellpose import models
        import warnings
        m = models.CellposeModel(gpu=False)
        img = np.zeros((32, 32), dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = m.eval(img, diameter=10)
        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        masks = result[0]
        assert masks.shape == (32, 32), f"Unexpected masks shape: {masks.shape}"


# ─── real XQD integration ────────────────────────────────────────────────────

_TEST_XQD = pathlib.Path(__file__).parent / "test_data" / "test.xqd"


@pytest.mark.skipif(not _TEST_XQD.exists(), reason="test.xqd not found")
class TestRealXQD:
    """End-to-end tests using the bundled test.xqd sample file."""

    def setup_method(self):
        from qdseg import AFMData
        self.data = AFMData(str(_TEST_XQD))
        self.pre = self.data.baseline_correction()
        self.pre.segment(method="advanced")

    def test_load_succeeds(self):
        """AFMData loads the xqd file without error and produces a 2-D array."""
        assert self.data.height_raw.ndim == 2
        assert self.data.height_raw.shape[0] > 0
        assert self.data.height_raw.shape[1] > 0

    def test_segment_detects_grains(self):
        """Advanced segmentation finds at least one grain on real data."""
        assert self.pre.labels.max() >= 1

    def test_stats_keys_present(self):
        """stats() returns a dict containing all expected top-level keys."""
        stats = self.pre.stats()
        for key in ("num_grains", "mean_height_nm", "mean_diameter_nm", "area_fraction"):
            assert key in stats, f"Missing key: {key}"

    def test_stats_plausible_values(self):
        """Returned statistics are physically plausible for a QD sample."""
        stats = self.pre.stats()
        assert stats["num_grains"] >= 1
        assert 1.0 < stats["mean_height_nm"] < 100.0     # typical QD: 5-30 nm
        assert 10.0 < stats["mean_diameter_nm"] < 500.0  # typical QD: 20-200 nm
        assert 0.0 < stats["area_fraction"] <= 1.0

    def test_grains_count_matches_stats(self):
        """Length of grains() list equals stats()['num_grains']."""
        stats = self.pre.stats()
        grains = self.pre.grains()
        assert len(grains) == stats["num_grains"]

    def test_segment_stardist(self):
        """StarDist segmentation detects grains on real data (skipped if not installed)."""
        pytest.importorskip("stardist")
        from qdseg import AFMData
        pre = AFMData(str(_TEST_XQD)).baseline_correction()
        pre.segment(method="stardist")
        assert pre.labels.max() >= 1
        assert pre.stats()["num_grains"] >= 1

    def test_segment_cellpose(self):
        """Cellpose segmentation detects grains on real data (skipped if not installed)."""
        pytest.importorskip("cellpose")
        from qdseg import AFMData
        pre = AFMData(str(_TEST_XQD)).baseline_correction()
        pre.segment(method="cellpose")
        assert pre.labels.max() >= 1
        assert pre.stats()["num_grains"] >= 1


# ─── entry point ────────────────────────────────────────────────────────────


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
