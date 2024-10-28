"""
Backtesting module for trading strategy evaluation.

This module provides comprehensive backtesting capabilities including:
- Walk-forward optimization
- Monte Carlo simulation
- Performance analytics
- Market condition analysis
- Interactive visualization
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

# Export main classes
from .enhancedbacktest import EnhancedBacktest, BacktestConfig

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class BacktestParameters:
    """Configuration parameters for backtesting sessions"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage: float = 0.0001       # 0.01%
    risk_free_rate: float = 0.02   # 2% annual
    
    # Optimization parameters
    min_samples: int = 252         # Minimum samples for training
    test_size: float = 0.3         # 30% for testing
    n_splits: int = 5              # Number of splits for walk-forward
    
    # Monte Carlo parameters
    n_simulations: int = 1000      # Number of Monte Carlo simulations
    confidence_level: float = 0.95  # Confidence level for metrics
    
    # Performance thresholds
    min_acceptable_sharpe: float = 1.0
    max_acceptable_drawdown: float = 0.20  # 20%
    target_annual_return: float = 0.15     # 15%

# Default configuration
DEFAULT_BACKTEST_PARAMS = BacktestParameters()

def create_backtest_session(
    data: Any,
    strategy: Any,
    parameters: Optional[BacktestParameters] = None,
    output_path: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Factory function to create and configure a backtest session.

    Args:
        data: Market data for backtesting
        strategy: Trading strategy to evaluate
        parameters: Optional custom backtest parameters
        output_path: Optional path for saving results

    Returns:
        Dictionary containing configured backtest components

    Raises:
        ValueError: If input parameters are invalid
        TypeError: If data or strategy are of wrong type
    """
    try:
        # Use default parameters if none provided
        parameters = parameters or DEFAULT_BACKTEST_PARAMS

        # Create BacktestConfig from parameters
        config = BacktestConfig(
            initial_capital=parameters.initial_capital,
            commission_rate=parameters.commission_rate,
            slippage=parameters.slippage,
            risk_free_rate=parameters.risk_free_rate,
            min_samples=parameters.min_samples,
            test_size=parameters.test_size,
            n_splits=parameters.n_splits
        )

        # Initialize backtest system
        backtest = EnhancedBacktest(data=data, config=config)

        # Create configuration dictionary
        backtest_config = {
            'backtest': backtest,
            'strategy': strategy,
            'parameters': parameters,
            'output_path': output_path
        }

        return backtest_config

    except Exception as e:
        logger.error(f"Error creating backtest session: {e}")
        raise

def run_backtest_analysis(
    backtest_config: Dict[str, Any],
    walk_forward: bool = True,
    monte_carlo: bool = True
) -> Dict[str, Any]:
    """
    Run comprehensive backtest analysis.

    Args:
        backtest_config: Configuration dictionary from create_backtest_session
        walk_forward: Whether to use walk-forward optimization
        monte_carlo: Whether to run Monte Carlo simulation

    Returns:
        Dictionary containing analysis results

    Raises:
        ValueError: If configuration is invalid
    """
    try:
        backtest = backtest_config['backtest']
        strategy = backtest_config['strategy']
        parameters = backtest_config['parameters']
        output_path = backtest_config['output_path']

        # Run backtest
        results = backtest.run_backtest(
            strategy=strategy,
            walk_forward=walk_forward,
            monte_carlo=monte_carlo,
            n_simulations=parameters.n_simulations
        )

        # Generate report
        report = backtest.generate_report(output_path)

        return {
            'results': results,
            'report': report,
            'performance_metrics': backtest.performance_metrics,
            'monte_carlo_results': backtest.monte_carlo_results if monte_carlo else None
        }

    except Exception as e:
        logger.error(f"Error running backtest analysis: {e}")
        raise

# Version information
__version__ = '1.0.0'

# Public API
__all__ = [
    'EnhancedBacktest',
    'BacktestConfig',
    'BacktestParameters',
    'DEFAULT_BACKTEST_PARAMS',
    'create_backtest_session',
    'run_backtest_analysis'
]