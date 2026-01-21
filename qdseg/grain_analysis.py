"""
Grain Analysis Module (Deprecated)

이 모듈은 하위 호환성을 위해 유지됩니다.
새 코드에서는 다음 모듈을 사용하세요:
    - grain_analyzer.segmentation: 세그멘테이션 함수
    - grain_analyzer.statistics: 통계 계산 함수

Migration Guide:
    # 이전 코드:
    from grain_analyzer.grain_analysis import segment_by_marker_growth
    from grain_analyzer.grain_analysis import calculate_grain_statistics
    
    # 새 코드:
    from grain_analyzer.segmentation import segment_rule_based
    from grain_analyzer.statistics import calculate_grain_statistics
"""

import warnings
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import ndimage


# Legacy class - deprecated
class AFMGrainAnalyzer:
    """
    AFM grain analysis class (Deprecated)
    
    새 코드에서는 함수 기반 API를 사용하세요:
        - segment_rule_based(), segment_stardist()
        - calculate_grain_statistics()
    """
    
    def __init__(self):
        warnings.warn(
            "AFMGrainAnalyzer는 deprecated입니다. "
            "segment_rule_based() 및 calculate_grain_statistics() 함수를 사용하세요.",
            DeprecationWarning,
            stacklevel=2
        )
        self.grain_labels = None
        self.grain_props = None
    
    def segment_by_marker_growth(
        self,
        height: np.ndarray,
        marker_coords_px: np.ndarray,
        *,
        meta: Optional[Dict] = None,
        mask: Optional[np.ndarray] = None,
        max_radius_nm: Optional[float] = None,
        anisotropic_nm_metric: bool = True,
    ) -> np.ndarray:
        """마커를 중심으로 원/타원을 키워가며 분할 (Legacy)"""
        return segment_by_marker_growth(
            height, marker_coords_px,
            meta=meta, mask=mask,
            max_radius_nm=max_radius_nm,
            anisotropic_nm_metric=anisotropic_nm_metric
        )
    
    def calculate_grain_statistics(
        self,
        grain_labels: np.ndarray,
        grain_props: Dict,
        meta: Optional[Dict] = None,
        min_pixel_area: int = 4
    ) -> Dict:
        """입자 통계 계산 (Legacy)"""
        from .statistics import calculate_grain_statistics as calc_stats
        # grain_props를 사용하지 않고 labels에서 직접 계산
        return calc_stats(grain_labels, None, meta, min_pixel_area=min_pixel_area)


# Legacy function - 하위 호환성 유지
def segment_by_marker_growth(
    height: np.ndarray,
    marker_coords_px: np.ndarray,
    *,
    meta: Optional[Dict] = None,
    mask: Optional[np.ndarray] = None,
    max_radius_nm: Optional[float] = None,
    anisotropic_nm_metric: bool = True,
) -> np.ndarray:
    """
    마커를 중심으로 Voronoi 분할
    
    새 코드에서는 segment_rule_based()을 사용하세요.
    이 함수는 내부 사용 및 하위 호환성을 위해 유지됩니다.
    
    Parameters
    ----------
    height : np.ndarray
        Height 이미지
    marker_coords_px : np.ndarray
        마커 좌표 (N, 2) - (row, col)
    meta : dict, optional
        메타데이터 ('pixel_nm' 키)
    mask : np.ndarray, optional
        분할 영역 제한 마스크
    max_radius_nm : float, optional
        최대 성장 반경 (nm)
    anisotropic_nm_metric : bool
        비등방성 nm 메트릭 사용 여부
    
    Returns
    -------
    labels : np.ndarray
        라벨 이미지 (int32)
    """
    h, w = height.shape
    labels = np.zeros((h, w), dtype=np.int32)
    
    if marker_coords_px is None or np.size(marker_coords_px) == 0:
        return labels

    # 픽셀→nm 스케일
    xp_nm, yp_nm = 1.0, 1.0
    if meta and 'pixel_nm' in meta:
        xp_nm, yp_nm = float(meta['pixel_nm'][0]), float(meta['pixel_nm'][1])
    sampling = (yp_nm, xp_nm) if anisotropic_nm_metric else None

    # 시드 마스크
    seeds_mask = np.zeros((h, w), dtype=bool)
    valid_coords: List[Tuple[int, int]] = []
    for r, c in np.asarray(marker_coords_px, dtype=int):
        if 0 <= r < h and 0 <= c < w:
            seeds_mask[r, c] = True
            valid_coords.append((r, c))
    
    if not np.any(seeds_mask):
        return labels

    # 최근접 시드 인덱스 (nm metric)
    dist_nm, (iy, ix) = ndimage.distance_transform_edt(
        ~seeds_mask, sampling=sampling, return_indices=True
    )

    # 시드 라벨 테이블
    seed_labels = np.zeros_like(labels)
    for i, (r, c) in enumerate(valid_coords, start=1):
        seed_labels[r, c] = i
    labels = seed_labels[iy, ix]

    # 마스크 제한
    if mask is not None:
        labels = np.where(mask.astype(bool), labels, 0)

    # 반경 제한
    if max_radius_nm is not None and max_radius_nm > 0:
        labels = np.where(dist_nm <= float(max_radius_nm), labels, 0)

    return labels


# Legacy function - redirect to new module
def calculate_grain_statistics(
    grain_labels: np.ndarray,
    grain_props: Dict,
    meta: Optional[Dict] = None,
    min_pixel_area: int = 4
) -> Dict:
    """
    입자 통계 계산 (Legacy)
    
    새 코드에서는 statistics.calculate_grain_statistics()를 사용하세요.
    """
    from .statistics import calculate_grain_statistics as calc_stats
    return calc_stats(grain_labels, None, meta, min_pixel_area=min_pixel_area)
