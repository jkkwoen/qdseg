"""
AFMData 래퍼 클래스
독립 패키지에서 사용하기 위한 간단한 AFMData 구현
"""

import numpy as np
from typing import Dict, Optional
from pathlib import Path

from .io import load_height_nm
from .corrections import AFMCorrections


class AFMData:
    """AFM 데이터 처리 클래스 (독립 패키지용 간단한 구현)"""
    
    def __init__(self, file_path: str):
        """AFMData 초기화
        
        Parameters
        ----------
        file_path : str
            XQD 파일 경로
        """
        self.file_path = file_path
        self.height_raw, self.meta = load_height_nm(file_path)
        self.current_data = self.height_raw.copy()
        self._corrector = AFMCorrections()
    
    def first_correction(self, method: str = 'polynomial') -> 'AFMData':
        """1차 기울기 보정"""
        self._corrector.set_method(method)
        self.current_data = self._corrector.correct_1st(self.current_data)
        return self
    
    def second_correction(self, method: str = 'polynomial') -> 'AFMData':
        """2차 기울기 보정"""
        self._corrector.set_method(method)
        self.current_data = self._corrector.correct_2nd(self.current_data)
        return self
    
    def third_correction(self, method: str = 'polynomial') -> 'AFMData':
        """3차 기울기 보정"""
        self._corrector.set_method(method)
        self.current_data = self._corrector.correct_3rd(self.current_data)
        return self
    
    def align_rows(self, method: str = 'median', poly_degree: int = 0, 
                   trim_fraction: float = 0.0) -> 'AFMData':
        """Scan Line Artefacts 보정 (flat_correction 전에 적용 권장)
        
        스캔 라인 간 불일치를 보정합니다. 빠른 스캔 축(일반적으로 x축)에서
        촬영한 프로필들이 서로 약간씩 이동되거나 기울기가 다를 수 있습니다.
        
        Parameters
        ----------
        method : str
            보정 방법 (기본값: 'median')
            - 'median': 각 라인의 median 값을 빼서 정렬
            - 'mean': 각 라인의 mean 값을 빼서 정렬
            - 'polynomial': 각 라인에서 다항식 피팅 후 제거
            - 'median_difference': 수직 이웃 간 높이 차이의 median을 0으로
            - 'trimmed_mean': 최저/최고값 일부를 제거한 평균 사용
        poly_degree : int
            polynomial 방법 사용 시 다항식 차수 (기본값: 0)
        trim_fraction : float
            trimmed_mean 방법 사용 시 제거할 비율 (0~0.5, 기본값: 0.0)
        
        Returns
        -------
        AFMData
            self (메서드 체이닝 지원)
        """
        self._corrector.set_align_rows_method(method)
        self.current_data = self._corrector.align_rows(
            self.current_data, 
            method=method,
            poly_degree=poly_degree,
            trim_fraction=trim_fraction
        )
        return self
    
    def flat_correction(self, method: str = "line_by_line", mask: Optional[np.ndarray] = None) -> 'AFMData':
        """평면 보정
        
        Parameters
        ----------
        method : str
            평면 보정 방법 (기본값: "line_by_line")
        mask : Optional[np.ndarray]
            세그멘테이션 마스크 (기본값: None)
            마스크가 제공되면 배경 픽셀(mask==0)만 사용하여 보정 계산
        
        Returns
        -------
        AFMData
            self (메서드 체이닝 지원)
        """
        self._corrector.set_flat_method(method)
        self.current_data = self._corrector.correct_flat(self.current_data, mask=mask)
        return self
    
    def baseline_correction(self, method: str = "min_to_zero") -> 'AFMData':
        """Baseline 보정"""
        self._corrector.set_baseline_method(method)
        self.current_data = self._corrector.correct_baseline(self.current_data)
        return self
    
    def get_data(self) -> np.ndarray:
        """현재 보정된 데이터 반환"""
        return self.current_data
    
    def get_raw_data(self) -> np.ndarray:
        """원본 데이터 반환"""
        return self.height_raw
    
    def get_meta(self) -> Dict:
        """메타데이터 반환"""
        return self.meta

