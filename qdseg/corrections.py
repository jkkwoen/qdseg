"""
AFM Data Corrections
===================
AFM 데이터 보정 기능을 위한 클래스
"""

import numpy as np
from typing import List, Optional, Union


class AFMCorrections:
    """AFM 데이터 보정 클래스
    
    이 클래스는 AFM 데이터의 다음 보정을 수행합니다:
    1. 기울기 보정 (1차, 2차, 3차)
    2. 평면 보정
    3. Baseline 보정
    """
    
    def __init__(self):
        """보정 클래스 초기화"""
        # 기본 설정
        self.slope_method = 'polynomial'      # 기울기 보정 방법
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
    
    def correct_flat(self, height: np.ndarray) -> np.ndarray:
        """평면 보정"""
        self._check_input(height)
        
        if self.flat_method == 'line_by_line':
            return self._correct_flat_line_by_line(height)
        elif self.flat_method == 'global':
            return self._correct_flat_global(height)
        else:  # median
            return self._correct_flat_median(height)
    
    def correct_baseline(self, height: np.ndarray) -> np.ndarray:
        """Baseline 보정 (0점 맞추기)"""
        self._check_input(height)
        
        if self.baseline_method == 'min_to_zero':
            return height - height.min()
        elif self.baseline_method == 'mean_to_zero':
            return height - height.mean()
        else:  # median_to_zero
            return height - np.median(height)
    
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
    
    def _correct_flat_line_by_line(self, height: np.ndarray) -> np.ndarray:
        """라인별 평면 보정"""
        result = height.copy()
        rows, cols = height.shape
        
        for i in range(rows):
            line = height[i, :]
            x = np.arange(cols)
            coeffs = np.polyfit(x, line, 1)
            trend = np.polyval(coeffs, x)
            result[i, :] = line - trend
        
        return result
    
    def _correct_flat_global(self, height: np.ndarray) -> np.ndarray:
        """전역 평면 보정"""
        rows, cols = height.shape
        y, x = np.mgrid[:rows, :cols]
        
        from sklearn.linear_model import LinearRegression
        
        coords = np.column_stack([x.ravel(), y.ravel()])
        model = LinearRegression()
        model.fit(coords, height.ravel())
        
        fitted_plane = model.predict(coords).reshape(height.shape)
        return height - fitted_plane
    
    def _correct_flat_median(self, height: np.ndarray) -> np.ndarray:
        """중간값 필터를 이용한 평면 보정"""
        from scipy import ndimage
        
        if isinstance(self.filter_size, float):
            size = max(1, int(min(height.shape) * self.filter_size))
        else:
            size = self.filter_size
        
        flat_ref = ndimage.median_filter(height, size=size)
        return height - flat_ref

