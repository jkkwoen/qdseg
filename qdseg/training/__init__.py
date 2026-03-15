"""
Grain Analyzer Training Module

Sub-package for training segmentation models.

Supported models:
- StarDist: Star-convex polygon detection (labels required, supervised learning)
- Cellpose: Gradient flow-based segmentation (labels required, supervised learning)

Usage:
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
    # StarDist
    'StarDistConfig',
    'StarDistTrainer',
    'create_stardist_labels',
    # Cellpose
    'CellposeConfig',
    'CellposeTrainer',
    'create_cellpose_labels',
]

