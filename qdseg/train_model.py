#!/usr/bin/env python3
"""
QDSeg 모델 학습 파이프라인 (Wrapper)

이 파일은 하위 호환성을 위해 유지됩니다.
실제 구현은 qdseg.training 모듈에 있습니다.

사용법:
    # CLI로 실행
    python -m qdseg.train_model
    
    # Python에서 import
    from qdseg.training import CellulusTrainer, TrainingConfig
"""

# 모든 내용을 training.cellulus_trainer에서 가져옴
from .training.cellulus_trainer import *
from .training.cellulus_trainer import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
