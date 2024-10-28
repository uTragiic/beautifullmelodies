"""
Model training module for NYSE trading system.
Handles model training, validation, and persistence across market clusters.
"""

from .trainer import ModelTrainer, TrainingConfig
from .scheduler import TrainingScheduler, ScheduleConfig
from .validation import ModelValidator, ValidationMetrics

__all__ = [
    'ModelTrainer',
    'TrainingConfig',
    'TrainingScheduler',
    'ScheduleConfig',
    'ModelValidator',
    'ValidationMetrics'
]

# Version information
__version__ = '1.0.0'