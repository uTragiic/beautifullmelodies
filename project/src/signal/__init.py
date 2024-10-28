"""
Signal generation module providing trading signal generation capabilities.

This module contains functionality for generating trading signals based on 
market data analysis and machine learning models.
"""

# Standard Library Imports
from dataclasses import dataclass
from typing import Optional
import logging

# Local Imports
from .generator import SignalGenerator

logger = logging.getLogger(__name__)

@dataclass
class SignalParameters:
    """Parameters for signal generation configuration."""
    min_confidence_score: float = 0.7
    signal_threshold: float = 0.75
    lookback_period: int = 100
    use_ml_model: bool = True
    volatility_adjustment: bool = True
    market_regime_aware: bool = True
    min_data_points: int = 200
    max_position_size: float = 1.0
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if not 0 <= self.min_confidence_score <= 1:
            raise ValueError("min_confidence_score must be between 0 and 1")
        if not 0 <= self.signal_threshold <= 1:
            raise ValueError("signal_threshold must be between 0 and 1")
        if self.lookback_period < 20:
            raise ValueError("lookback_period must be at least 20")
        if self.min_data_points < self.lookback_period:
            raise ValueError("min_data_points must be greater than lookback_period")
        if not 0 < self.max_position_size <= 1:
            raise ValueError("max_position_size must be between 0 and 1")

def create_signal_generator(
    db_path: str,
    model_path: str,
    parameters: Optional[SignalParameters] = None
) -> SignalGenerator:
    """
    Create a signal generator instance with the specified configuration.

    Args:
        db_path: Path to market data database
        model_path: Path to trained model
        parameters: Signal generation parameters

    Returns:
        Configured SignalGenerator instance
    """
    try:
        # Use default parameters if none provided
        if parameters is None:
            parameters = SignalParameters()

        # Create generator instance
        generator = SignalGenerator(
            db_path=db_path,
            model_path=model_path,
            min_confidence=parameters.min_confidence_score,
            signal_threshold=parameters.signal_threshold,
            lookback_period=parameters.lookback_period,
            use_ml_model=parameters.use_ml_model,
            volatility_adjustment=parameters.volatility_adjustment,
            market_regime_aware=parameters.market_regime_aware,
            min_data_points=parameters.min_data_points,
            max_position_size=parameters.max_position_size
        )

        logger.info("Signal generator created successfully")
        return generator

    except Exception as e:
        logger.error(f"Error creating signal generator: {e}")
        raise

__all__ = [
    'SignalGenerator',
    'SignalParameters',
    'create_signal_generator'
]