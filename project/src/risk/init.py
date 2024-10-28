"""
Risk management module for trading system.

This module provides components for managing trading risk, including:
- Position sizing
- Stop loss and take profit management 
- Risk monitoring and adjustment
- Portfolio heat calculation
"""

from typing import Dict, Any
from dataclasses import dataclass

# Export main classes
from .manager import RiskManagement
from .positionsizing import PositionSizing, PositionConfig
from .tpsl import TakeProfitStopLoss, TPSLConfig

# Define module level types
@dataclass
class RiskParameters:
    """Parameters for risk management configuration"""
    max_risk_per_trade: float
    max_position_size: float
    min_risk_reward_ratio: float
    max_portfolio_heat: float
    position_config: PositionConfig
    tpsl_config: TPSLConfig

class RiskLevel:
    """Enumeration of risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

# Default configurations
DEFAULT_RISK_PARAMS = RiskParameters(
    max_risk_per_trade=0.02,  # 2% risk per trade
    max_position_size=0.1,    # 10% of account
    min_risk_reward_ratio=2.0,
    max_portfolio_heat=0.5,   # 50% max heat
    position_config=PositionConfig(
        min_position_size=0.0,
        max_position_size=float('inf'),
        min_risk_multiplier=0.5,
        max_risk_multiplier=2.0,
        volatility_scale_factor=0.5,
        confidence_scale_factor=0.5
    ),
    tpsl_config=TPSLConfig(
        min_risk_reward_ratio=2.0,
        max_risk_reward_ratio=5.0,
        min_stop_distance=0.001,
        max_stop_distance=0.1,
        min_target_distance=0.002,
        max_target_distance=0.2,
        trailing_activation_threshold=0.01
    )
)

def create_risk_manager(
    account_balance: float,
    backtest_results: Any,
    market_conditions_file: str,
    db_path: str,
    risk_params: RiskParameters = DEFAULT_RISK_PARAMS,
    market_index: str = 'SPY'
) -> Dict[str, Any]:
    """
    Factory function to create risk management components.

    Args:
        account_balance: Current account balance
        backtest_results: Backtest performance data
        market_conditions_file: Path to market conditions data
        db_path: Path to market database
        risk_params: Risk management parameters
        market_index: Market index symbol

    Returns:
        Dictionary containing initialized risk management components

    Raises:
        ValueError: If input parameters are invalid
        FileNotFoundError: If required files are not found
    """
    # Create risk management components
    risk_manager = RiskManagement(
        backtest_results=backtest_results,
        market_conditions_file=market_conditions_file,
        db_path=db_path,
        market_index=market_index
    )

    position_sizer = PositionSizing(
        account_balance=account_balance,
        max_risk_per_trade=risk_params.max_risk_per_trade,
        min_risk_reward_ratio=risk_params.min_risk_reward_ratio,
        position_config=risk_params.position_config
    )

    tpsl_manager = TakeProfitStopLoss(
        config=risk_params.tpsl_config
    )

    return {
        'risk_manager': risk_manager,
        'position_sizer': position_sizer,
        'tpsl_manager': tpsl_manager
    }

# Version information
__version__ = '1.0.0'

# Public API
__all__ = [
    'RiskManagement',
    'PositionSizing',
    'TakeProfitStopLoss',
    'RiskParameters',
    'PositionConfig',
    'TPSLConfig',
    'RiskLevel',
    'create_risk_manager',
    'DEFAULT_RISK_PARAMS'
]