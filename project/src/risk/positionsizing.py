# Standard Library Imports
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

# Third-Party Imports
import numpy as np

# Local Imports
from ..utils.validation import validate_positive_float, validate_confidence_score
import config

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class PositionConfig:
    """Configuration parameters for position sizing"""
    min_position_size: float = 0.0
    max_position_size: float = float('inf')
    min_risk_multiplier: float = 0.5
    max_risk_multiplier: float = 2.0
    volatility_scale_factor: float = 0.5
    confidence_scale_factor: float = 0.5

class PositionSizing:
    """
    Handles position sizing calculations with dynamic adjustments based on
    multiple factors including account balance, risk parameters, market conditions,
    and confidence scores.
    """

    def __init__(self, 
                 account_balance: float, 
                 max_risk_per_trade: float,
                 min_risk_reward_ratio: float = 2.0,
                 volatility_adjustment_factor: float = 0.5,
                 position_config: Optional[PositionConfig] = None):
        """
        Initialize the position sizing system.

        Args:
            account_balance: Current account balance
            max_risk_per_trade: Maximum risk per trade as decimal (e.g., 0.02 for 2%)
            min_risk_reward_ratio: Minimum required risk/reward ratio
            volatility_adjustment_factor: Factor for volatility-based adjustments
            position_config: Optional configuration parameters

        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            self.account_balance = validate_positive_float(account_balance)
            self.max_risk_per_trade = validate_positive_float(max_risk_per_trade)
            self.min_risk_reward_ratio = validate_positive_float(min_risk_reward_ratio)
            self.volatility_adjustment_factor = validate_positive_float(volatility_adjustment_factor)
            self.position_config = position_config or PositionConfig()
            
            logger.info(f"Position sizing initialized with balance: {account_balance}")
            
        except Exception as e:
            logger.error(f"Error initializing position sizing: {e}")
            raise

    def calculate_position_size(self,
                              entry_price: float,
                              stop_loss: float,
                              take_profit: float,
                              confidence_score: float,
                              current_volatility: float,
                              average_volatility: float,
                              market_data: Dict[str, Any]) -> float:
        """
        Calculate optimal position size based on multiple factors.

        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence_score: Trade confidence score (0-1)
            current_volatility: Current market volatility
            average_volatility: Average historical volatility
            market_data: Additional market data for calculations

        Returns:
            Calculated position size

        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Validate inputs
            entry_price = validate_positive_float(entry_price)
            stop_loss = validate_positive_float(stop_loss)
            take_profit = validate_positive_float(take_profit)
            confidence_score = validate_confidence_score(confidence_score)
            
            # Check risk-reward ratio
            risk_per_share = abs(entry_price - stop_loss)
            reward_per_share = abs(take_profit - entry_price)
            risk_reward_ratio = reward_per_share / risk_per_share

            if risk_reward_ratio < self.min_risk_reward_ratio:
                logger.warning(f"Risk-reward ratio {risk_reward_ratio} below minimum {self.min_risk_reward_ratio}")
                return 0

            # Calculate base position size
            risk_amount = self.account_balance * self.max_risk_per_trade
            base_position_size = risk_amount / risk_per_share

            # Apply confidence adjustment
            confidence_adjusted_size = self._adjust_for_confidence(
                base_position_size, 
                confidence_score
            )

            # Apply volatility adjustment
            volatility_adjusted_size = self._adjust_for_volatility(
                confidence_adjusted_size,
                current_volatility,
                average_volatility
            )

            # Apply market condition adjustments
            final_position_size = self._apply_market_adjustments(
                volatility_adjusted_size,
                market_data
            )

            # Ensure position size doesn't exceed account constraints
            final_position_size = self._apply_account_constraints(
                final_position_size,
                entry_price
            )

            logger.info(f"Calculated position size: {final_position_size}")
            return round(final_position_size, 2)

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            raise

    def _adjust_for_confidence(self, 
                             base_size: float, 
                             confidence_score: float) -> float:
        """
        Adjust position size based on confidence score.

        Args:
            base_size: Base position size
            confidence_score: Confidence score (0-1)

        Returns:
            Adjusted position size
        """
        scale_factor = self.position_config.confidence_scale_factor
        adjustment = 0.5 + (confidence_score * scale_factor)
        return base_size * adjustment

    def _adjust_for_volatility(self,
                             position_size: float,
                             current_volatility: float,
                             average_volatility: float) -> float:
        """
        Adjust position size based on market volatility.

        Args:
            position_size: Current position size
            current_volatility: Current market volatility
            average_volatility: Average historical volatility

        Returns:
            Volatility-adjusted position size
        """
        volatility_ratio = current_volatility / average_volatility
        adjustment = 1 + ((volatility_ratio - 1) * self.volatility_adjustment_factor)
        return position_size / adjustment

    def _apply_market_adjustments(self,
                                position_size: float,
                                market_data: Dict[str, Any]) -> float:
        """
        Apply market condition-based adjustments to position size.

        Args:
            position_size: Current position size
            market_data: Dictionary containing market condition data

        Returns:
            Adjusted position size
        """
        adjusted_size = position_size

        # Reduce position size in high volatility conditions
        if market_data.get('high_volatility', False):
            adjusted_size *= 0.8

        # Increase position size in strong trend conditions
        if market_data.get('strong_trend', False):
            adjusted_size *= 1.2

        return adjusted_size

    def _apply_account_constraints(self,
                                 position_size: float,
                                 entry_price: float) -> float:
        """
        Apply account-level constraints to position size.

        Args:
            position_size: Calculated position size
            entry_price: Entry price for the trade

        Returns:
            Constrained position size
        """
        # Ensure position size doesn't exceed account balance
        max_affordable_shares = self.account_balance / entry_price
        constrained_size = min(position_size, max_affordable_shares)

        # Apply position config constraints
        constrained_size = max(
            min(constrained_size, self.position_config.max_position_size),
            self.position_config.min_position_size
        )

        return constrained_size

    def update_account_balance(self, new_balance: float) -> None:
        """
        Update the account balance.

        Args:
            new_balance: New account balance

        Raises:
            ValueError: If new_balance is not positive
        """
        try:
            self.account_balance = validate_positive_float(new_balance)
            logger.info(f"Account balance updated to: {new_balance}")
        except Exception as e:
            logger.error(f"Error updating account balance: {e}")
            raise

    def calculate_dynamic_risk_per_trade(self, performance_metric: float) -> None:
        """
        Dynamically adjust maximum risk per trade based on performance.

        Args:
            performance_metric: Performance metric (e.g., Sharpe ratio)

        Raises:
            ValueError: If performance_metric is invalid
        """
        try:
            validate_positive_float(performance_metric)
            
            base_risk = config.BASE_RISK_PER_TRADE
            max_risk = config.MAX_RISK_PER_TRADE
            
            self.max_risk_per_trade = np.clip(
                base_risk * performance_metric,
                base_risk,
                max_risk
            )
            
            logger.info(f"Risk per trade updated to: {self.max_risk_per_trade}")
            
        except Exception as e:
            logger.error(f"Error updating risk per trade: {e}")
            raise

    def calculate_portfolio_heat(self,
                               open_positions: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate total portfolio risk exposure.

        Args:
            open_positions: Dictionary of open positions with sizes and prices

        Returns:
            Portfolio heat as percentage of account balance

        Raises:
            ValueError: If position data is invalid
        """
        try:
            total_risk = sum(
                self.calculate_position_value(
                    pos['size'],
                    pos['current_price']
                ) * self.max_risk_per_trade
                for pos in open_positions.values()
            )
            
            heat = (total_risk / self.account_balance) * 100
            logger.info(f"Current portfolio heat: {heat}%")
            return heat
            
        except Exception as e:
            logger.error(f"Error calculating portfolio heat: {e}")
            raise

    @staticmethod
    def calculate_position_value(position_size: float,
                               current_price: float) -> float:
        """
        Calculate the current value of a position.

        Args:
            position_size: Size of the position
            current_price: Current price of the asset

        Returns:
            Current position value

        Raises:
            ValueError: If inputs are invalid
        """
        try:
            position_size = validate_positive_float(position_size)
            current_price = validate_positive_float(current_price)
            return position_size * current_price
        except Exception as e:
            logger.error(f"Error calculating position value: {e}")
            raise