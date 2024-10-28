# Import main classes and types from core modules
from .database_handler import DatabaseHandler
from .performance_metrics import PerformanceMetrics
from .types import (
    MarketRegime,
    TradingSignal,
    MarketCondition,
    TradeStatus,
    RiskLevel,
    TradeInfo,
    PriceData,
    IndicatorConfig,
    RiskConfig,
    BacktestConfig
)

# Define what should be available when using "from core import *"
__all__ = [
    'DatabaseHandler',
    'PerformanceMetrics',
    'MarketRegime',
    'TradingSignal',
    'MarketCondition',
    'TradeStatus',
    'RiskLevel',
    'TradeInfo',
    'PriceData',
    'IndicatorConfig',
    'RiskConfig',
    'BacktestConfig'
]

# Version information
__version__ = '1.0.0'

# Module level docstring
"""
Core package for the trading system.

This package provides the fundamental components and data structures
used throughout the trading system, including:
- Database handling
- Performance metrics calculation
- Common types and enums
- Configuration settings
"""