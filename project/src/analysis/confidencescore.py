# Standard Library Imports
import logging
from typing import Dict, List, Optional

# Third-Party Imports
import numpy as np
import pandas as pd

# Local Imports
from ..core.types import MarketCondition

# Set up logging
logger = logging.getLogger(__name__)

class ConfidenceScoreCalculator:
    """
    Calculates confidence scores for trading signals based on multiple factors.
    """
    
    def __init__(self, lookback_period: int = 20):
        """
        Initialize ConfidenceScoreCalculator.
        
        Args:
            lookback_period: Number of periods to look back for calculations
        """
        self.lookback_period = lookback_period
        self.weights = {
            'lwr': 0.4,  # Live win rate
            'bp': 0.3,   # Backtest performance
            'mcwp': 0.2, # Market condition win percentage
            'ssm': 0.1,  # Signal strength metric
            'vaf': 0.0   # Volatility adjustment factor
        }

    def calculate_confidence_score(self, 
                                 lwr: float, 
                                 bp: float, 
                                 mcwp: float, 
                                 ssm: float, 
                                 vaf: float,
                                 market_regime: Optional[str] = None) -> float:
        """
        Calculate comprehensive confidence score.
        
        Args:
            lwr: Live win rate
            bp: Backtest performance
            mcwp: Market condition win percentage
            ssm: Signal strength metric
            vaf: Volatility adjustment factor
            market_regime: Current market regime (optional)
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Adjust weights based on performance comparison
            adjusted_weights = self._calculate_weights(lwr, bp, market_regime)
            
            # Calculate weighted score
            score = (
                adjusted_weights['lwr'] * lwr +
                adjusted_weights['bp'] * bp +
                adjusted_weights['mcwp'] * mcwp +
                adjusted_weights['ssm'] * ssm +
                adjusted_weights['vaf'] * vaf
            )
            
            # Ensure score is between 0 and 1
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0

    def _calculate_weights(self, 
                         lwr: float, 
                         bp: float, 
                         market_regime: Optional[str] = None) -> Dict[str, float]:
        """
        Calculate adaptive weights based on performance and market regime.
        
        Args:
            lwr: Live win rate
            bp: Backtest performance
            market_regime: Current market regime
            
        Returns:
            Dictionary of adjusted weights
        """
        # Calculate performance factor
        performance_factor = lwr / bp if bp != 0 else 1
        
        # Adjust weights based on performance comparison
        w_lwr = self.weights['lwr'] * performance_factor
        w_bp = self.weights['bp'] / performance_factor
        w_mcwp = self.weights['mcwp']
        w_ssm = self.weights['ssm']
        w_vaf = self.weights['vaf']
        
        # Adjust for market regime if provided
        if market_regime:
            weights = self._adjust_weights_for_regime(
                {
                    'lwr': w_lwr,
                    'bp': w_bp,
                    'mcwp': w_mcwp,
                    'ssm': w_ssm,
                    'vaf': w_vaf
                },
                market_regime
            )
        else:
            weights = {
                'lwr': w_lwr,
                'bp': w_bp,
                'mcwp': w_mcwp,
                'ssm': w_ssm,
                'vaf': w_vaf
            }
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        return {k: v / total_weight for k, v in weights.items()}

    def _adjust_weights_for_regime(self, 
                                 weights: Dict[str, float], 
                                 market_regime: str) -> Dict[str, float]:
        """
        Adjust weights based on market regime.
        
        Args:
            weights: Current weight dictionary
            market_regime: Current market regime
            
        Returns:
            Dictionary of regime-adjusted weights
        """
        if market_regime == "HIGH_VOLATILITY":
            # In high volatility, increase weight of recent performance
            weights['lwr'] *= 1.2
            weights['bp'] *= 0.8
        elif market_regime == "TRENDING":
            # In trending markets, increase weight of backtest performance
            weights['lwr'] *= 0.9
            weights['bp'] *= 1.1
            weights['mcwp'] *= 1.2
        elif market_regime == "RANGING":
            # In ranging markets, increase weight of signal strength
            weights['ssm'] *= 1.3
            weights['vaf'] *= 1.2
            
        return weights

    def calculate_live_win_rate(self, 
                              data: pd.DataFrame, 
                              signals: pd.Series) -> float:
        """
        Calculate win rate from recent live trading data.
        
        Args:
            data: Market data DataFrame
            signals: Series of trading signals
            
        Returns:
            Win rate as a float between 0 and 1
        """
        if len(data) < self.lookback_period:
            return 0.5  # Default to neutral if insufficient data
            
        recent_data = data.tail(self.lookback_period)
        returns = recent_data['close'].pct_change()
        signal_returns = returns * signals.shift(1)
        
        winning_trades = (signal_returns > 0).sum()
        total_trades = (signals != 0).sum()
        
        return winning_trades / total_trades if total_trades > 0 else 0.5

    def calculate_signal_strength(self, 
                                indicators: Dict[str, tuple]) -> float:
        """
        Calculate signal strength metric from technical indicators.
        
        Args:
            indicators: Dictionary of indicator values and their statistics
            
        Returns:
            Signal strength score between 0 and 1
        """
        try:
            weighted_sum = 0
            total_weight = 0
            
            for indicator, (value, mean, std) in indicators.items():
                if std > 0:  # Avoid division by zero
                    z_score = (value - mean) / std
                    weight = self._get_indicator_weight(indicator)
                    weighted_sum += z_score * weight
                    total_weight += weight
                    
            if total_weight > 0:
                normalized_score = weighted_sum / total_weight
                # Convert to probability using sigmoid function
                return 1 / (1 + np.exp(-normalized_score))
            return 0.5
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.5

    def _get_indicator_weight(self, indicator: str) -> float:
        """
        Get the weight for a specific indicator.
        
        Args:
            indicator: Name of the indicator
            
        Returns:
            Weight value for the indicator
        """
        weights = {
            'RSI': 0.2,
            'MACD': 0.2,
            'ATR': 0.15,
            'Stochastic': 0.15,
            'BB_width': 0.3
        }
        return weights.get(indicator, 0.1)

    def calculate_volatility_adjustment(self, 
                                     current_volatility: float,
                                     average_volatility: float) -> float:
        """
        Calculate volatility adjustment factor.
        
        Args:
            current_volatility: Current market volatility
            average_volatility: Average historical volatility
            
        Returns:
            Volatility adjustment factor between 0 and 1
        """
        if average_volatility == 0:
            return 1.0
            
        vol_ratio = current_volatility / average_volatility
        return 1.0 / (1.0 + abs(vol_ratio - 1.0))

    def update_weights(self, performance_metrics: Dict[str, float]) -> None:
        """
        Update weights based on performance metrics.
        
        Args:
            performance_metrics: Dictionary of performance metrics
        """
        if 'sharpe_ratio' in performance_metrics:
            # Adjust weights based on Sharpe ratio
            performance_factor = max(0.5, min(1.5, performance_metrics['sharpe_ratio']))
            self.weights['lwr'] *= performance_factor
            self.weights['bp'] /= performance_factor
            
            # Normalize weights
            total_weight = sum(self.weights.values())
            self.weights = {k: v / total_weight for k, v in self.weights.items()}