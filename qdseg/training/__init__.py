"""
Grain Analyzer 학습 모듈

세그멘테이션 모델 학습을 위한 서브패키지입니다.

지원 모델:
- Cellulus: 비지도 학습 기반 인스턴스 세그멘테이션 (레이블 불필요)
- StarDist: Star-convex polygon detection (레이블 필요, 지도 학습)
- Cellpose: Gradient flow 기반 세그멘테이션 (레이블 필요, 지도 학습)

사용법:
    # Cellulus (비지도 학습 - 레이블 불필요)
    from grain_analyzer.training import CellulusTrainer, TrainingConfig
    
    config = TrainingConfig(
        num_epochs=10000,
        batch_size=16,
        data_dir='./data',
        output_dir='./models',
    )
    trainer = CellulusTrainer(config)
    images = load_afm_data(config.data_dir)
    zarr_path = trainer.prepare_data(images, for_official=True)
    checkpoint_path = trainer.train_official(zarr_path)
    
    # StarDist (지도 학습 - 레이블 필요)
    from grain_analyzer.training import StarDistTrainer, StarDistConfig
    
    config = StarDistConfig(
        data_dir='./labeled_data',  # images/, masks/ 폴더 포함
        output_dir='./models/stardist',
        epochs=100,
    )
    trainer = StarDistTrainer(config)
    model_path = trainer.train()
    
    # Cellpose (지도 학습 - 레이블 필요)
    from grain_analyzer.training import CellposeTrainer, CellposeConfig
    
    config = CellposeConfig(
        data_dir='./labeled_data',  # images/, masks/ 폴더 포함
        output_dir='./models/cellpose',
        n_epochs=100,
    )
    trainer = CellposeTrainer(config)
    model_path = trainer.train()
"""

# Cellulus (비지도 학습)
from .cellulus_trainer import (
    TrainingConfig,
    CellulusTrainer,
    load_afm_data,
    get_device,
    get_hardware_info,
    print_hardware_info,
    setup_environment,
)

# StarDist (지도 학습)
from .stardist_trainer import (
    StarDistConfig,
    StarDistTrainer,
    create_labeled_data_from_rule_based as create_stardist_labels,
)

# Cellpose (지도 학습)
from .cellpose_trainer import (
    CellposeConfig,
    CellposeTrainer,
    create_labeled_data_from_rule_based as create_cellpose_labels,
)

__all__ = [
    # Cellulus
    'TrainingConfig',
    'CellulusTrainer',
    'load_afm_data',
    'get_device',
    'get_hardware_info',
    'print_hardware_info',
    'setup_environment',
    # StarDist
    'StarDistConfig',
    'StarDistTrainer',
    'create_stardist_labels',
    # Cellpose
    'CellposeConfig',
    'CellposeTrainer',
    'create_cellpose_labels',
]

