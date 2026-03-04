"""
Grain Analyzer Training Module

Sub-package for training segmentation models.

Supported models:
- Cellulus: Unsupervised instance segmentation (no labels required)
- StarDist: Star-convex polygon detection (labels required, supervised learning)
- Cellpose: Gradient flow-based segmentation (labels required, supervised learning)

Usage:
    # Cellulus (unsupervised - no labels required)
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

    # StarDist (supervised - labels required)
    from grain_analyzer.training import StarDistTrainer, StarDistConfig

    config = StarDistConfig(
        data_dir='./labeled_data',  # contains images/, masks/ folders
        output_dir='./models/stardist',
        epochs=100,
    )
    trainer = StarDistTrainer(config)
    model_path = trainer.train()

    # Cellpose (supervised - labels required)
    from grain_analyzer.training import CellposeTrainer, CellposeConfig

    config = CellposeConfig(
        data_dir='./labeled_data',  # contains images/, masks/ folders
        output_dir='./models/cellpose',
        n_epochs=100,
    )
    trainer = CellposeTrainer(config)
    model_path = trainer.train()
"""

# Cellulus (unsupervised)
from .cellulus_trainer import (
    TrainingConfig,
    CellulusTrainer,
    load_afm_data,
    get_device,
    get_hardware_info,
    print_hardware_info,
    setup_environment,
)

# StarDist (supervised)
from .stardist_trainer import (
    StarDistConfig,
    StarDistTrainer,
    create_labeled_data_from_rule_based as create_stardist_labels,
)

# Cellpose (supervised)
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

