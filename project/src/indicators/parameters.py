import logging
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

from ..core.types import MarketRegime
from ..utils.validation import validate_dataframe, ValidationError
import config

logger = logging.getLogger(__name__)

@dataclass
class AdjustmentFactors:
    """Stores adjustment factors for parameter calculations"""
    alpha: float = 0.2  # Factor for bullish threshold reduction
    beta: float = 0.3   # Factor for bearish threshold increase
    gamma: float = 0.5  # Win rate sensitivity
    delta: float = 0.5  # Loss streak sensitivity
    epsilon: float = 0.4  # Volatility sensitivity
    zeta: float = 0.3   # Volume sensitivity

class ParameterAdjuster:
    """
    Enhanced parameter adjuster with dynamic threshold calculations and
    market regime-specific adjustments.
    """
    
    def __init__(self):
        """Initialize the parameter adjuster with default settings."""
        self.asset_characteristics = {}
        self._market_volatility_cache = None
        self._last_volatility_update = None
        self.adjustment_factors = AdjustmentFactors()
        self.adjustment_history: Dict[str, List[float]] = {}
        
        # Base thresholds
        self.base_thresholds = {
            'T_bull_base': 0.7,
            'T_bear_base': 0.3,
            'momentum_threshold': 0.1,
            'volatility_threshold': 0.2,
            'volume_threshold': 1.5
        }

    def adjust_parameters(self, data: pd.DataFrame, ticker: str, market_regime: str) -> Dict[str, float]:
        """
        Adjust parameters based on market conditions and asset characteristics.
        
        Args:
            data: Market data DataFrame
            ticker: Asset ticker symbol
            market_regime: Current market regime
            
        Returns:
            Dictionary of adjusted parameters
        """
        try:
            # Calculate current volatility
            volatility = self._calculate_current_volatility(data)
            
            # Get or calculate asset characteristics
            if ticker not in self.asset_characteristics:
                self.asset_characteristics[ticker] = self._calculate_asset_characteristics(data)
            
            asset_volatility = self.asset_characteristics[ticker]['volatility']
            market_volatility = self._get_market_volatility()
            
            # Calculate win rate and trend factors
            win_rate = self._calculate_win_rate(data)
            trend_strength = self._calculate_trend_strength(data)
            
            # Calculate base adjustments
            volatility_ratio = asset_volatility / market_volatility
            current_vol_ratio = volatility / self.asset_characteristics[ticker]['avg_volatility']
            
            # Calculate market regime adjustments
            regime_adjustments = self._calculate_regime_adjustments(market_regime, trend_strength)
            
            # Calculate thresholds based on regime
            thresholds = self._calculate_dynamic_thresholds(market_regime, win_rate, trend_strength)
            
            # Apply adjustments to parameters
            adjusted_params = {
                'rsi_window': self._adjust_rsi_window(volatility_ratio, regime_adjustments),
                'macd_fast': self._adjust_macd_fast(volatility_ratio, regime_adjustments),
                'macd_slow': self._adjust_macd_slow(volatility_ratio, regime_adjustments),
                'adx_window': self._adjust_adx_window(volatility_ratio, regime_adjustments),
                'atr_window': self._adjust_atr_window(volatility_ratio, regime_adjustments),
                'bull_threshold': thresholds['T_bull'],
                'bear_threshold': thresholds['T_bear']
            }
            
            # Apply parameter bounds
            bounded_params = self._apply_parameter_bounds(adjusted_params)
            
            # Update adjustment history
            self._update_adjustment_history(bounded_params)
            
            return bounded_params
            
        except Exception as e:
            logger.error(f"Error adjusting parameters: {e}")
            raise

    def _calculate_dynamic_thresholds(self, 
                                    market_regime: str,
                                    win_rate: float,
                                    trend_strength: float) -> Dict[str, float]:
        """
        Calculate dynamic thresholds based on market regime and performance.
        
        Args:
            market_regime: Current market regime
            win_rate: Current win rate
            trend_strength: Current trend strength
            
        Returns:
            Dictionary of calculated thresholds
        """
        # Get base thresholds
        T_bull_base = self.base_thresholds['T_bull_base']
        T_bear_base = self.base_thresholds['T_bear_base']
        
        # Calculate adjustments based on regime
        if 'Uptrend' in market_regime:
            # Implement formula from document: T_bull_up = T_bull_base × (1 - α)
            T_bull = T_bull_base * (1 - self.adjustment_factors.alpha)
            # Implement formula from document: T_bear_up = T_bear_base × (1 + β)
            T_bear = T_bear_base * (1 + self.adjustment_factors.beta)
            
        elif 'Downtrend' in market_regime:
            # Opposite adjustments for downtrend
            T_bull = T_bull_base * (1 + self.adjustment_factors.beta)
            T_bear = T_bear_base * (1 - self.adjustment_factors.alpha)
            
        else:  # Ranging or undefined
            T_bull = T_bull_base
            T_bear = T_bear_base
        
        # Adjust based on win rate
        if win_rate > 0.6:  # Winning streak
            win_adjustment = self.adjustment_factors.gamma * (win_rate - 0.6)
            T_bull *= (1 - win_adjustment)
            T_bear *= (1 - win_adjustment)
        elif win_rate < 0.5:  # Losing streak
            loss_adjustment = self.adjustment_factors.delta * (0.5 - win_rate)
            T_bull *= (1 + loss_adjustment)
            T_bear *= (1 + loss_adjustment)
            
        # Adjust based on trend strength
        trend_adjustment = trend_strength * self.adjustment_factors.epsilon
        T_bull *= (1 + trend_adjustment)
        T_bear *= (1 + trend_adjustment)
        
        return {
            'T_bull': T_bull,
            'T_bear': T_bear
        }

    def _calculate_regime_adjustments(self, 
                                    market_regime: str,
                                    trend_strength: float) -> Dict[str, float]:
        """
        Calculate adjustment factors based on market regime.
        
        Args:
            market_regime: Current market regime
            trend_strength: Current trend strength
            
        Returns:
            Dictionary of regime-specific adjustments
        """
        adjustments = {
            'window_multiplier': 1.0,
            'sensitivity_multiplier': 1.0,
            'threshold_multiplier': 1.0
        }
        
        if 'Uptrend' in market_regime:
            trend_factor = min(1.5, 1 + trend_strength)
            adjustments['window_multiplier'] = trend_factor
            adjustments['sensitivity_multiplier'] = 1 / trend_factor
            adjustments['threshold_multiplier'] = 0.9  # More sensitive to bull signals
            
        elif 'Downtrend' in market_regime:
            trend_factor = min(1.5, 1 + trend_strength)
            adjustments['window_multiplier'] = trend_factor
            adjustments['sensitivity_multiplier'] = trend_factor
            adjustments['threshold_multiplier'] = 1.1  # Less sensitive to bull signals
            
        elif 'HighVol' in market_regime:
            adjustments['window_multiplier'] = 1.2
            adjustments['sensitivity_multiplier'] = 0.8
            adjustments['threshold_multiplier'] = 1.2
            
        elif 'LowVol' in market_regime:
            adjustments['window_multiplier'] = 0.8
            adjustments['sensitivity_multiplier'] = 1.2
            adjustments['threshold_multiplier'] = 0.9
            
        return adjustments

    def _adjust_rsi_window(self, volatility_ratio: float, regime_adjustments: Dict[str, float]) -> int:
        """Adjust RSI window based on volatility and regime."""
        base_window = config.RSI_WINDOW
        adjustment = volatility_ratio * regime_adjustments['window_multiplier']
        return int(base_window * adjustment)

    def _adjust_macd_fast(self, volatility_ratio: float, regime_adjustments: Dict[str, float]) -> int:
        """Adjust MACD fast period based on volatility and regime."""
        base_fast = config.MACD_FAST
        adjustment = 1 / (volatility_ratio * regime_adjustments['sensitivity_multiplier'])
        return int(base_fast * adjustment)

    def _adjust_macd_slow(self, volatility_ratio: float, regime_adjustments: Dict[str, float]) -> int:
        """Adjust MACD slow period based on volatility and regime."""
        base_slow = config.MACD_SLOW
        adjustment = volatility_ratio * regime_adjustments['window_multiplier']
        return int(base_slow * adjustment)

    def _calculate_win_rate(self, data: pd.DataFrame, lookback: int = 20) -> float:
        """Calculate recent win rate."""
        returns = data['close'].pct_change().tail(lookback)
        return (returns > 0).mean()

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators."""
        # SMA trend
        sma_ratio = data['SMA_50'].iloc[-1] / data['SMA_200'].iloc[-1] - 1
        
        # ADX trend strength
        adx = data['ADX'].iloc[-1] / 100
        
        # MACD trend
        macd = data['MACD'].iloc[-1] / data['close'].iloc[-1]
        
        # Combine indicators
        trend_strength = (
            0.4 * abs(sma_ratio) +
            0.4 * adx +
            0.2 * abs(macd)
        )
        
        return trend_strength

    def _update_adjustment_history(self, params: Dict[str, float]) -> None:
        """Update parameter adjustment history."""
        for param, value in params.items():
            if param not in self.adjustment_history:
                self.adjustment_history[param] = []
            self.adjustment_history[param].append(value)
            
            # Keep history length manageable
            if len(self.adjustment_history[param]) > 100:
                self.adjustment_history[param].pop(0)

    def calculate_adaptive_thresholds(self, 
                                    market_regime: str,
                                    win_rate: float,
                                    volatility_ratio: float) -> Dict[str, float]:
        """
        Calculate adaptive thresholds for various indicators.
        
        Args:
            market_regime: Current market regime
            win_rate: Current win rate
            volatility_ratio: Current volatility ratio
            
        Returns:
            Dictionary of adaptive thresholds
        """
        # Base thresholds from configuration
        base_thresholds = self.base_thresholds.copy()
        
        # Apply win rate adjustments
        win_factor = self.adjustment_factors.gamma * (win_rate - 0.5)
        
        # Apply volatility adjustments
        vol_factor = self.adjustment_factors.epsilon * (volatility_ratio - 1)
        
        # Calculate regime-specific adjustments
        regime_factor = 1.0
        if 'Uptrend' in market_regime:
            regime_factor = 0.9  # More sensitive in uptrends
        elif 'Downtrend' in market_regime:
            regime_factor = 1.1  # Less sensitive in downtrends
        elif 'HighVol' in market_regime:
            regime_factor = 1.2  # Much less sensitive in high volatility
            
        # Apply all adjustments
        adjusted_thresholds = {}
        for name, base_value in base_thresholds.items():
            adjustment = 1 + win_factor + vol_factor
            adjusted_thresholds[name] = base_value * adjustment * regime_factor
            
        return adjusted_thresholds

    def get_adjustment_stats(self, parameter: str) -> Dict[str, float]:
        """
        Get statistics about parameter adjustments.
        
        Args:
            parameter: Name of parameter to analyze
            
        Returns:
            Dictionary of adjustment statistics
        """
        if parameter not in self.adjustment_history:
            return {}
            
        history = self.adjustment_history[parameter]
        return {
            'mean': np.mean(history),
            'std': np.std(history),
            'min': np.min(history),
            'max': np.max(history),
            'current': history[-1],
            'stability': np.std(history) / np.mean(history) if np.mean(history) != 0 else np.inf
        }

    def reset_adjustment_history(self) -> None:
        """Reset the adjustment history."""
        self.adjustment_history = {}