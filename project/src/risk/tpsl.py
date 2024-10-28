# Standard Library Imports
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# Third-Party Imports
import numpy as np

# Local Imports
from ..utils.validation import (
    validate_positive_float,
    validate_market_condition,
    validate_trade_direction
)
from ..core.types import MarketRegime
import config

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class TPSLConfig:
    """Configuration parameters for take profit and stop loss calculations"""
    min_risk_reward_ratio: float = 2.0
    max_risk_reward_ratio: float = 5.0
    min_stop_distance: float = 0.001  # 0.1%
    max_stop_distance: float = 0.1    # 10%
    min_target_distance: float = 0.002 # 0.2%
    max_target_distance: float = 0.2   # 20%
    trailing_activation_threshold: float = 0.01  # 1%

class TakeProfitStopLoss:
    """
    Manages dynamic calculation and adjustment of take profit and stop loss levels
    based on market conditions, volatility, and confidence scores.
    """

    def __init__(self, 
                 atr_multiplier: float = 2.0,
                 confidence_factor: float = 0.2,
                 volatility_factor: float = 0.1,
                 config: Optional[TPSLConfig] = None):
        """
        Initialize the TPSL manager.

        Args:
            atr_multiplier: Multiplier for ATR-based calculations
            confidence_factor: Weight of confidence score in adjustments
            volatility_factor: Weight of volatility in adjustments
            config: Optional configuration parameters

        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            self.atr_multiplier = validate_positive_float(atr_multiplier)
            self.confidence_factor = validate_positive_float(confidence_factor)
            self.volatility_factor = validate_positive_float(volatility_factor)
            self.config = config or TPSLConfig()
            
            logger.info("TPSL manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing TPSL manager: {e}")
            raise

    def calculate_tp_sl(self,
                       entry_price: float,
                       atr: float,
                       risk_parameters: Dict[str, float],
                       trade_direction: str,
                       confidence_score: float,
                       current_volatility: float,
                       average_volatility: float) -> Tuple[float, float]:
        """
        Calculate initial stop loss and take profit levels.

        Args:
            entry_price: Entry price for the trade
            atr: Average True Range value
            risk_parameters: Dictionary containing base stop loss and take profit multipliers
            trade_direction: 'long' or 'short'
            confidence_score: Confidence score for the trade (0-1)
            current_volatility: Current market volatility
            average_volatility: Average historical volatility

        Returns:
            Tuple of (stop_loss, take_profit) prices

        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Validate inputs
            entry_price = validate_positive_float(entry_price)
            atr = validate_positive_float(atr)
            trade_direction = validate_trade_direction(trade_direction)
            
            # Get base levels
            base_sl = risk_parameters['stop_loss']
            base_tp = risk_parameters['take_profit']

            # Calculate adjustments
            confidence_adjustment = self._calculate_confidence_adjustment(confidence_score)
            volatility_adjustment = self._calculate_volatility_adjustment(
                current_volatility,
                average_volatility
            )

            # Apply adjustments to ATR
            adjusted_atr = atr * confidence_adjustment * volatility_adjustment

            # Calculate levels based on trade direction
            if trade_direction == 'long':
                stop_loss = entry_price - (base_sl * adjusted_atr * self.atr_multiplier)
                take_profit = entry_price + (base_tp * adjusted_atr * self.atr_multiplier)
            else:  # short
                stop_loss = entry_price + (base_sl * adjusted_atr * self.atr_multiplier)
                take_profit = entry_price - (base_tp * adjusted_atr * self.atr_multiplier)

            # Validate risk-reward ratio
            if not self._validate_risk_reward_ratio(entry_price, stop_loss, take_profit):
                logger.warning("Invalid risk-reward ratio, adjusting levels")
                stop_loss, take_profit = self._adjust_for_risk_reward(
                    entry_price,
                    stop_loss,
                    take_profit,
                    trade_direction
                )

            # Apply distance constraints
            stop_loss, take_profit = self._apply_distance_constraints(
                entry_price,
                stop_loss,
                take_profit
            )

            logger.info(f"Calculated SL: {stop_loss}, TP: {take_profit}")
            return round(stop_loss, 2), round(take_profit, 2)

        except Exception as e:
            logger.error(f"Error calculating TP/SL levels: {e}")
            raise

    def adjust_tp_sl(self,
                    current_price: float,
                    entry_price: float,
                    stop_loss: float,
                    take_profit: float,
                    atr: float,
                    trade_direction: str,
                    market_condition: str) -> Tuple[float, float]:
        """
        Adjust stop loss and take profit levels based on price movement.

        Args:
            current_price: Current price of the asset
            entry_price: Original entry price
            stop_loss: Current stop loss level
            take_profit: Current take profit level
            atr: Current Average True Range
            trade_direction: 'long' or 'short'
            market_condition: Current market condition

        Returns:
            Tuple of (new_stop_loss, new_take_profit)

        Raises:
            ValueError: If input parameters are invalid
        """
        try:
            # Validate inputs
            validate_market_condition(market_condition)
            trade_direction = validate_trade_direction(trade_direction)

            # Calculate trailing stop
            new_stop_loss = self._calculate_trailing_stop(
                current_price,
                atr,
                trade_direction,
                market_condition,
                stop_loss
            )

            # Adjust take profit
            new_take_profit = self._adjust_take_profit(
                current_price,
                entry_price,
                take_profit,
                atr,
                trade_direction,
                market_condition
            )

            logger.info(f"Adjusted SL: {new_stop_loss}, TP: {new_take_profit}")
            return round(new_stop_loss, 2), round(new_take_profit, 2)

        except Exception as e:
            logger.error(f"Error adjusting TP/SL levels: {e}")
            raise

    def _calculate_trailing_stop(self,
                               current_price: float,
                               atr: float,
                               trade_direction: str,
                               market_condition: str,
                               current_stop: float) -> float:
        """
        Calculate trailing stop based on market conditions.
        """
        try:
            base_distance = self.atr_multiplier * atr
            
            # Adjust distance based on market condition
            if market_condition == 'trending':
                distance = base_distance * 1.5  # Wider trailing stop in trends
            elif market_condition == 'ranging':
                distance = base_distance * 0.8  # Tighter trailing stop in ranges
            else:
                distance = base_distance

            if trade_direction == 'long':
                new_stop = current_price - distance
                return max(new_stop, current_stop)
            else:  # short
                new_stop = current_price + distance
                return min(new_stop, current_stop)

        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            raise

    def _adjust_take_profit(self,
                          current_price: float,
                          entry_price: float,
                          take_profit: float,
                          atr: float,
                          trade_direction: str,
                          market_condition: str) -> float:
        """
        Dynamically adjust take profit based on price movement and market conditions.
        """
        try:
            price_movement = abs(current_price - entry_price)
            
            if market_condition == 'trending':
                # Let profits run in trending markets
                tp_distance = max(price_movement, self.atr_multiplier * atr * 2)
            elif market_condition == 'ranging':
                # Take profits quicker in ranging markets
                tp_distance = min(price_movement, self.atr_multiplier * atr)
            else:
                tp_distance = self.atr_multiplier * atr

            if trade_direction == 'long':
                new_take_profit = max(take_profit, entry_price + tp_distance)
            else:  # short
                new_take_profit = min(take_profit, entry_price - tp_distance)

            return new_take_profit

        except Exception as e:
            logger.error(f"Error adjusting take profit: {e}")
            raise

    def _calculate_confidence_adjustment(self, confidence_score: float) -> float:
        """Calculate adjustment factor based on confidence score."""
        return 1 + (confidence_score - 0.5) * self.confidence_factor

    def _calculate_volatility_adjustment(self,
                                      current_volatility: float,
                                      average_volatility: float) -> float:
        """Calculate adjustment factor based on volatility."""
        volatility_ratio = current_volatility / average_volatility
        return 1 + (volatility_ratio - 1) * self.volatility_factor

    def _validate_risk_reward_ratio(self,
                                  entry_price: float,
                                  stop_loss: float,
                                  take_profit: float) -> bool:
        """
        Validate if the risk-reward ratio is within acceptable range.
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk == 0:
            return False
            
        ratio = reward / risk
        return (self.config.min_risk_reward_ratio <= ratio <= 
                self.config.max_risk_reward_ratio)

    def _adjust_for_risk_reward(self,
                              entry_price: float,
                              stop_loss: float,
                              take_profit: float,
                              trade_direction: str) -> Tuple[float, float]:
        """
        Adjust levels to maintain acceptable risk-reward ratio.
        """
        try:
            risk = abs(entry_price - stop_loss)
            min_reward = risk * self.config.min_risk_reward_ratio
            max_reward = risk * self.config.max_risk_reward_ratio

            if trade_direction == 'long':
                take_profit = min(
                    entry_price + max_reward,
                    max(entry_price + min_reward, take_profit)
                )
            else:
                take_profit = max(
                    entry_price - max_reward,
                    min(entry_price - min_reward, take_profit)
                )

            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"Error adjusting risk-reward ratio: {e}")
            raise

    def _apply_distance_constraints(self,
                                 entry_price: float,
                                 stop_loss: float,
                                 take_profit: float) -> Tuple[float, float]:
        """
        Apply minimum and maximum distance constraints to TP/SL levels.
        """
        try:
            # Calculate current distances as percentages
            sl_distance = abs(stop_loss - entry_price) / entry_price
            tp_distance = abs(take_profit - entry_price) / entry_price

            # Apply constraints to stop loss
            if sl_distance < self.config.min_stop_distance:
                sl_distance = self.config.min_stop_distance
            elif sl_distance > self.config.max_stop_distance:
                sl_distance = self.config.max_stop_distance

            # Apply constraints to take profit
            if tp_distance < self.config.min_target_distance:
                tp_distance = self.config.min_target_distance
            elif tp_distance > self.config.max_target_distance:
                tp_distance = self.config.max_target_distance

            # Recalculate levels
            if stop_loss < entry_price:
                stop_loss = entry_price * (1 - sl_distance)
                take_profit = entry_price * (1 + tp_distance)
            else:
                stop_loss = entry_price * (1 + sl_distance)
                take_profit = entry_price * (1 - tp_distance)

            return stop_loss, take_profit

        except Exception as e:
            logger.error(f"Error applying distance constraints: {e}")
            raise