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
    
    def flat_correction(self, method: str = "line_by_line") -> 'AFMData':
        """평면 보정"""
        self._corrector.set_flat_method(method)
        self.current_data = self._corrector.correct_flat(self.current_data)
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

