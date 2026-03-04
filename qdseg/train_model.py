#!/usr/bin/env python3
"""
QDSeg Model Training Pipeline (Wrapper)

This file is maintained for backward compatibility.
The actual implementation is in the qdseg.training module.

Usage:
    # Run from CLI
    python -m qdseg.train_model

    # Import from Python
    from qdseg.training import CellulusTrainer, TrainingConfig
"""

# Import everything from training.cellulus_trainer
from .training.cellulus_trainer import *
from .training.cellulus_trainer import main

if __name__ == "__main__":
    import sys
    sys.exit(main())
