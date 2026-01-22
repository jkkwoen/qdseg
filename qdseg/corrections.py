"""
AFM Data Corrections
===================
AFM 데이터 보정 기능을 위한 클래스
"""

import numpy as np
from typing import List, Optional, Union
from scipy import stats


class AFMCorrections:
    """AFM 데이터 보정 클래스
    
    이 클래스는 AFM 데이터의 다음 보정을 수행합니다:
    1. 기울기 보정 (1차, 2차, 3차)
    2. Scan Line Artefacts 보정 (Align Rows)
    3. 평면 보정
    4. Baseline 보정
    """
    
    def __init__(self):
        """보정 클래스 초기화"""
        # 기본 설정
        self.slope_method = 'polynomial'      # 기울기 보정 방법
        self.align_rows_method = 'median'     # Scan Line Artefacts 보정 방법
        self.flat_method = 'line_by_line'     # 평면 보정 방법
        self.baseline_method = 'min_to_zero'  # baseline 보정 방법
        
        # 필터 크기 (평면 보정용)
        self.filter_size = 0.1  # 전체 크기의 10%
    
    def set_method(self, method: str):
        """보정 방법 설정
        
        Parameters
        ----------
        method : str
            'polynomial' 또는 'simple'
        """
        if method in ['polynomial', 'simple']:
            self.slope_method = method
        else:
            raise ValueError("method는 'polynomial' 또는 'simple'이어야 합니다")
    
    def set_flat_method(self, method: str):
        """평면 보정 방법 설정
        
        Parameters
        ----------
        method : str
            'line_by_line', 'global', 'median'
        """
        if method in ['line_by_line', 'global', 'median']:
            self.flat_method = method
        else:
            raise ValueError("flat_method는 'line_by_line', 'global', 'median' 중 하나여야 합니다")
    
    def set_baseline_method(self, method: str):
        """Baseline 보정 방법 설정
        
        Parameters
        ----------
        method : str
            'min_to_zero', 'mean_to_zero', 'median_to_zero'
        """
        if method in ['min_to_zero', 'mean_to_zero', 'median_to_zero']:
            self.baseline_method = method
        else:
            raise ValueError("baseline_method는 'min_to_zero', 'mean_to_zero', 'median_to_zero' 중 하나여야 합니다")
    
    def set_align_rows_method(self, method: str):
        """Scan Line Artefacts (Align Rows) 보정 방법 설정
        
        Parameters
        ----------
        method : str
            'median', 'mean', 'polynomial', 'median_difference', 'trimmed_mean'
            - median: 각 라인의 median 값을 빼서 정렬
            - mean: 각 라인의 mean 값을 빼서 정렬 (polynomial degree=0과 동일)
            - polynomial: 각 라인에서 다항식 피팅 후 제거 (degree 지정 가능)
            - median_difference: 수직 이웃 픽셀 간 높이 차이의 median을 0으로
            - trimmed_mean: 최저/최고값 일부를 제거한 평균 사용
        """
        valid_methods = ['median', 'mean', 'polynomial', 'median_difference', 'trimmed_mean']
        if method in valid_methods:
            self.align_rows_method = method
        else:
            raise ValueError(f"align_rows_method는 {valid_methods} 중 하나여야 합니다")
    
    def _check_input(self, height: np.ndarray):
        """입력 데이터 검증"""
        if not isinstance(height, np.ndarray):
            raise TypeError("height는 numpy array여야 합니다")
        if height.ndim != 2:
            raise ValueError("height는 2차원 배열이어야 합니다")
        if height.size == 0:
            raise ValueError("height 배열이 비어있습니다")
    
    def correct_1st(self, height: np.ndarray) -> np.ndarray:
        """1차 기울기 보정"""
        self._check_input(height)
        
        if self.slope_method == 'polynomial':
            return self._correct_1st_polynomial(height)
        else:
            return self._correct_1st_simple(height)
    
    def correct_2nd(self, height: np.ndarray) -> np.ndarray:
        """2차 기울기 보정"""
        self._check_input(height)
        
        if self.slope_method == 'polynomial':
            return self._correct_2nd_polynomial(height)
        else:
            return self._correct_2nd_simple(height)
    
    def correct_3rd(self, height: np.ndarray) -> np.ndarray:
        """3차 기울기 보정"""
        self._check_input(height)
        
        if self.slope_method == 'polynomial':
            return self._correct_3rd_polynomial(height)
        else:
            return self._correct_3rd_simple(height)
    
    def align_rows(self, height: np.ndarray, method: Optional[str] = None, 
                   poly_degree: int = 0, trim_fraction: float = 0.0) -> np.ndarray:
        """Scan Line Artefacts 보정 (Align Rows)
        
        스캔 라인 간 불일치를 보정합니다. flat_correction 전에 적용하는 것이 좋습니다.
        
        Parameters
        ----------
        height : np.ndarray
            입력 높이 데이터
        method : str, optional
            보정 방법. None이면 self.align_rows_method 사용
            - 'median': 각 라인의 median 값을 빼서 정렬
            - 'mean': 각 라인의 mean 값을 빼서 정렬
            - 'polynomial': 각 라인에서 다항식 피팅 후 제거
            - 'median_difference': 수직 이웃 간 높이 차이의 median을 0으로
            - 'trimmed_mean': 최저/최고값 일부를 제거한 평균 사용
        poly_degree : int
            polynomial 방법 사용 시 다항식 차수 (기본값: 0 = mean)
        trim_fraction : float
            trimmed_mean 방법 사용 시 제거할 비율 (0~0.5, 기본값: 0.0)
        
        Returns
        -------
        np.ndarray
            보정된 높이 데이터
        """
        self._check_input(height)
        
        if method is None:
            method = self.align_rows_method
        
        if method == 'median':
            return self._align_rows_median(height)
        elif method == 'mean':
            return self._align_rows_mean(height)
        elif method == 'polynomial':
            return self._align_rows_polynomial(height, degree=poly_degree)
        elif method == 'median_difference':
            return self._align_rows_median_difference(height)
        elif method == 'trimmed_mean':
            return self._align_rows_trimmed_mean(height, trim_fraction=trim_fraction)
        else:
            raise ValueError(f"Unknown align_rows method: {method}")
    
    def correct_flat(self, height: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """평면 보정
        
        Parameters
        ----------
        height : np.ndarray
            입력 높이 데이터
        mask : np.ndarray, optional
            마스크 배열 (0: 배경, >0: grain 영역)
            마스크가 제공되면 배경(mask==0) 영역만 사용하여 보정 계산
        
        Returns
        -------
        np.ndarray
            보정된 높이 데이터
        """
        self._check_input(height)
        
        if self.flat_method == 'line_by_line':
            return self._correct_flat_line_by_line(height, mask)
        elif self.flat_method == 'global':
            return self._correct_flat_global(height, mask)
        else:  # median
            return self._correct_flat_median(height, mask)
    
    def correct_baseline(self, height: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Baseline 보정 (0점 맞추기)
        
        Parameters
        ----------
        height : np.ndarray
            입력 높이 데이터
        mask : np.ndarray, optional
            마스크 배열 (0: 배경, >0: grain 영역)
            마스크가 제공되면 배경(mask==0) 영역만 사용하여 baseline 계산
        
        Returns
        -------
        np.ndarray
            보정된 높이 데이터
        """
        self._check_input(height)
        
        # 마스크가 있으면 배경 영역만 사용
        if mask is not None:
            background = height[mask == 0]
            if len(background) == 0:
                # 배경이 없으면 전체 사용
                background = height
        else:
            background = height
        
        if self.baseline_method == 'min_to_zero':
            return height - background.min()
        elif self.baseline_method == 'mean_to_zero':
            return height - background.mean()
        else:  # median_to_zero
            return height - np.median(background)
    
    def _correct_1st_polynomial(self, height: np.ndarray) -> np.ndarray:
        """다항식을 이용한 1차 보정"""
        rows, cols = height.shape
        y, x = np.mgrid[:rows, :cols]
        
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        coords = np.column_stack([x.ravel(), y.ravel()])
        poly = PolynomialFeatures(degree=1)
        X_poly = poly.fit_transform(coords)
        
        model = LinearRegression()
        model.fit(X_poly, height.ravel())
        
        fitted_plane = model.predict(X_poly).reshape(height.shape)
        return height - fitted_plane
    
    def _correct_1st_simple(self, height: np.ndarray) -> np.ndarray:
        """간단한 1차 보정"""
        rows, cols = height.shape
        
        x_coeffs = np.polyfit(np.arange(cols), height.mean(axis=0), 1)
        y_coeffs = np.polyfit(np.arange(rows), height.mean(axis=1), 1)
        
        x_trend = np.polyval(x_coeffs, np.arange(cols))
        y_trend = np.polyval(y_coeffs, np.arange(rows))
        
        x_correction = np.tile(x_trend, (rows, 1))
        y_correction = np.tile(y_trend.reshape(-1, 1), (1, cols))
        
        return height - x_correction - y_correction
    
    def _correct_2nd_polynomial(self, height: np.ndarray) -> np.ndarray:
        """다항식을 이용한 2차 보정"""
        rows, cols = height.shape
        y, x = np.mgrid[:rows, :cols]
        
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        coords = np.column_stack([x.ravel(), y.ravel()])
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(coords)
        
        model = LinearRegression()
        model.fit(X_poly, height.ravel())
        
        fitted_surface = model.predict(X_poly).reshape(height.shape)
        return height - fitted_surface
    
    def _correct_2nd_simple(self, height: np.ndarray) -> np.ndarray:
        """간단한 2차 보정"""
        result = self._correct_1st_simple(height)
        return self._correct_1st_simple(result)
    
    def _correct_3rd_polynomial(self, height: np.ndarray) -> np.ndarray:
        """다항식을 이용한 3차 보정"""
        rows, cols = height.shape
        y, x = np.mgrid[:rows, :cols]
        
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        
        coords = np.column_stack([x.ravel(), y.ravel()])
        poly = PolynomialFeatures(degree=3)
        X_poly = poly.fit_transform(coords)
        
        model = LinearRegression()
        model.fit(X_poly, height.ravel())
        
        fitted_surface = model.predict(X_poly).reshape(height.shape)
        return height - fitted_surface
    
    def _correct_3rd_simple(self, height: np.ndarray) -> np.ndarray:
        """간단한 3차 보정"""
        result = height.copy()
        for _ in range(3):
            result = self._correct_1st_simple(result)
        return result
    
    def _correct_flat_line_by_line(self, height: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """라인별 평면 보정
        
        mask가 제공되면 각 라인에서 배경(mask==0) 픽셀만 사용하여 트렌드 계산
        """
        result = height.copy()
        rows, cols = height.shape
        
        for i in range(rows):
            line = height[i, :]
            x = np.arange(cols)
            
            # 마스크가 있으면 배경 픽셀만 사용
            if mask is not None:
                line_mask = mask[i, :]
                background_idx = line_mask == 0
                
                if np.sum(background_idx) < 2:
                    # 배경 픽셀이 2개 미만이면 보정 스킵
                    continue
                
                x_bg = x[background_idx]
                line_bg = line[background_idx]
            else:
                x_bg = x
                line_bg = line
            
            # 1차 다항식 피팅
            coeffs = np.polyfit(x_bg, line_bg, 1)
            trend = np.polyval(coeffs, x)
            result[i, :] = line - trend
        
        return result
    
    def _correct_flat_global(self, height: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """전역 평면 보정
        
        mask가 제공되면 배경(mask==0) 픽셀만 사용하여 평면 피팅
        """
        rows, cols = height.shape
        y, x = np.mgrid[:rows, :cols]
        
        from sklearn.linear_model import LinearRegression
        
        # 마스크가 있으면 배경 픽셀만 사용
        if mask is not None:
            background_idx = mask == 0
            if np.sum(background_idx) < 3:
                # 배경 픽셀이 너무 적으면 전체 사용
                background_idx = np.ones_like(mask, dtype=bool)
            
            coords = np.column_stack([x[background_idx], y[background_idx]])
            z = height[background_idx]
        else:
            coords = np.column_stack([x.ravel(), y.ravel()])
            z = height.ravel()
        
        model = LinearRegression()
        model.fit(coords, z)
        
        # 전체 이미지에 대해 예측
        all_coords = np.column_stack([x.ravel(), y.ravel()])
        fitted_plane = model.predict(all_coords).reshape(height.shape)
        
        return height - fitted_plane
    
    def _correct_flat_median(self, height: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """중간값 필터를 이용한 평면 보정
        
        mask는 median filter에는 적용하지 않음 (구조적 필터링이므로)
        """
        from scipy import ndimage
        
        if isinstance(self.filter_size, float):
            size = max(1, int(min(height.shape) * self.filter_size))
        else:
            size = self.filter_size
        
        flat_ref = ndimage.median_filter(height, size=size)
        return height - flat_ref
    
    # ============================================================
    # Scan Line Artefacts (Align Rows) 보정 메서드들
    # ============================================================
    
    def _align_rows_median(self, height: np.ndarray) -> np.ndarray:
        """각 라인의 median을 빼서 라인 정렬
        
        기본적인 보정 방법으로, 각 스캔 라인의 대표 높이(median)를 찾아
        빼줌으로써 라인들을 같은 높이로 이동시킵니다.
        """
        result = height.copy()
        rows = height.shape[0]
        
        for i in range(rows):
            line_median = np.median(height[i, :])
            result[i, :] = height[i, :] - line_median
        
        return result
    
    def _align_rows_mean(self, height: np.ndarray) -> np.ndarray:
        """각 라인의 mean을 빼서 라인 정렬
        
        polynomial degree=0과 동일한 방법입니다.
        """
        result = height.copy()
        rows = height.shape[0]
        
        for i in range(rows):
            line_mean = np.mean(height[i, :])
            result[i, :] = height[i, :] - line_mean
        
        return result
    
    def _align_rows_polynomial(self, height: np.ndarray, degree: int = 0) -> np.ndarray:
        """각 라인에서 다항식을 피팅하고 제거
        
        Parameters
        ----------
        degree : int
            다항식 차수
            - 0: mean 제거 (Median과 유사)
            - 1: 선형 기울기 제거
            - 2: 곡률 제거
        """
        result = height.copy()
        rows, cols = height.shape
        x = np.arange(cols)
        
        for i in range(rows):
            line = height[i, :]
            coeffs = np.polyfit(x, line, degree)
            trend = np.polyval(coeffs, x)
            result[i, :] = line - trend
        
        return result
    
    def _align_rows_median_difference(self, height: np.ndarray) -> np.ndarray:
        """수직 이웃 픽셀 간 높이 차이의 median을 0으로 만듦
        
        라인들을 이동시켜서 수직 이웃 픽셀 간 높이 차이의 median이
        0이 되도록 합니다. 큰 특징을 더 잘 보존하면서도 완전히
        잘못된 라인에 더 민감합니다.
        """
        result = height.copy()
        rows = height.shape[0]
        
        # 첫 번째 라인은 그대로 유지
        for i in range(1, rows):
            # 현재 라인과 이전 라인 간의 차이
            diff = height[i, :] - height[i-1, :]
            median_diff = np.median(diff)
            
            # median difference가 0이 되도록 현재 라인 조정
            result[i, :] = result[i-1, :] + (diff - median_diff)
        
        return result
    
    def _align_rows_trimmed_mean(self, height: np.ndarray, trim_fraction: float = 0.1) -> np.ndarray:
        """Trimmed mean을 사용한 라인 정렬
        
        각 라인에서 최저값과 최고값의 일부를 제거한 후 평균을 계산합니다.
        
        Parameters
        ----------
        trim_fraction : float
            제거할 비율 (0~0.5)
            - 0.0: mean과 동일
            - 0.5: median과 동일
            - 0.1~0.2: 이상치의 영향을 줄이면서 robust한 평균
        """
        if not 0 <= trim_fraction <= 0.5:
            raise ValueError("trim_fraction은 0과 0.5 사이여야 합니다")
        
        result = height.copy()
        rows = height.shape[0]
        
        for i in range(rows):
            line = height[i, :]
            # scipy.stats.trim_mean 사용
            line_trimmed_mean = stats.trim_mean(line, trim_fraction)
            result[i, :] = line - line_trimmed_mean
        
        return result

