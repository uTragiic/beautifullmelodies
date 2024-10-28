"""
Signal generator implementation with support for different signal generation methods.
"""

# Standard Library Imports
import logging
from typing import Optional, Dict, Any, Tuple

# Third-Party Imports
import numpy as np
import pandas as pd

# Local Imports
from ..core.database_handler import DatabaseHandler
from ..models.mlmodel import MachineLearningModel
from ..indicators.calculator import IndicatorCalculator
from ..analysis.marketcondition import MarketConditionAnalyzer
from ..analysis.confidencescore import ConfidenceScoreCalculator

logger = logging.getLogger(__name__)

class SignalGenerator:
    """
    Generates trading signals based on market data analysis and ML predictions.
    """

    def __init__(
        self,
        db_path: str,
        model_path: str,
        min_confidence: float = 0.7,
        signal_threshold: float = 0.75,
        lookback_period: int = 100,
        use_ml_model: bool = True,
        volatility_adjustment: bool = True,
        market_regime_aware: bool = True,
        min_data_points: int = 200,
        max_position_size: float = 1.0
    ):
        """
        Initialize SignalGenerator.

        Args:
            db_path: Path to market data database
            model_path: Path to trained model
            min_confidence: Minimum confidence score to generate signal
            signal_threshold: Threshold for signal generation
            lookback_period: Number of periods for lookback analysis
            use_ml_model: Whether to use ML model for predictions
            volatility_adjustment: Whether to adjust for volatility
            market_regime_aware: Whether to consider market regime
            min_data_points: Minimum required data points
            max_position_size: Maximum position size as fraction of capital
        """
        self.db_handler = DatabaseHandler(db_path)
        self.ml_model = MachineLearningModel(lookback_period=lookback_period)
        self.indicator_calculator = IndicatorCalculator()
        self.market_condition_analyzer = MarketConditionAnalyzer()
        self.confidence_calculator = ConfidenceScoreCalculator()
        
        # Store configuration
        self.min_confidence = min_confidence
        self.signal_threshold = signal_threshold
        self.lookback_period = lookback_period
        self.use_ml_model = use_ml_model
        self.volatility_adjustment = volatility_adjustment
        self.market_regime_aware = market_regime_aware
        self.min_data_points = min_data_points
        self.max_position_size = max_position_size

        # Load model if using ML
        if self.use_ml_model:
            try:
                self.ml_model.load_model(model_path)
                logger.info("ML model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading ML model: {e}")
                raise

    def generate_signal(self, market_data: pd.DataFrame) -> Tuple[int, float]:
        """
        Generate trading signal from market data.

        Args:
            market_data: DataFrame with market data

        Returns:
            Tuple of (signal, confidence_score)
            signal: -1 (sell), 0 (hold), or 1 (buy)
        """
        try:
            # Validate data
            if len(market_data) < self.min_data_points:
                logger.warning("Insufficient data points for signal generation")
                return 0, 0.0

            # Calculate indicators
            data = self.indicator_calculator.calculate_indicators(market_data)

            # Get market condition
            market_condition = self.market_condition_analyzer.determine_market_condition(data)

            # Generate base signal
            base_signal = 0
            confidence_score = 0.0

            if self.use_ml_model:
                # Get ML prediction
                prediction_probability = self.ml_model.predict(data)
                
                if prediction_probability > self.signal_threshold:
                    base_signal = 1
                elif prediction_probability < (1 - self.signal_threshold):
                    base_signal = -1

                # Calculate confidence score
                confidence_score = self.calculate_confidence_score(
                    data, 
                    market_condition,
                    prediction_probability
                )

                # Adjust for market regime if enabled
                if self.market_regime_aware:
                    base_signal = self._adjust_for_market_regime(
                        base_signal,
                        market_condition,
                        data
                    )

                # Adjust for volatility if enabled
                if self.volatility_adjustment:
                    base_signal = self._adjust_for_volatility(
                        base_signal,
                        data
                    )

            else:
                # Use traditional indicators for signal generation
                signal_indicators = self._calculate_indicator_signals(data)
                base_signal = self._combine_indicator_signals(signal_indicators)
                confidence_score = self._calculate_indicator_confidence(signal_indicators)

            # Only return signal if confidence exceeds minimum
            final_signal = base_signal if confidence_score >= self.min_confidence else 0

            # Scale signal by position size
            return final_signal, confidence_score

        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return 0, 0.0

    def _adjust_for_market_regime(
        self,
        signal: int,
        market_condition: str,
        data: pd.DataFrame
    ) -> int:
        """Adjust signal based on market regime."""
        try:
            if "Downtrend" in market_condition and signal > 0:
                return 0
            if "Uptrend" in market_condition and signal < 0:
                return 0
            if "High_Volatility" in market_condition:
                return int(signal * 0.5)  # Reduce signal size in high volatility
            return signal
        except Exception as e:
            logger.error(f"Error adjusting for market regime: {e}")
            return signal

    def _adjust_for_volatility(self, signal: int, data: pd.DataFrame) -> int:
        """Adjust signal based on current volatility."""
        try:
            current_volatility = data['ATR'].iloc[-1]
            avg_volatility = data['ATR'].mean()
            
            if current_volatility > (2 * avg_volatility):
                return 0  # No trade in extreme volatility
            elif current_volatility > avg_volatility:
                return int(signal * 0.5)  # Reduce position size in high volatility
            return signal
        except Exception as e:
            logger.error(f"Error adjusting for volatility: {e}")
            return signal

    def _calculate_indicator_signals(self, data: pd.DataFrame) -> Dict[str, int]:
        """Calculate signals from traditional indicators."""
        try:
            signals = {}
            
            # RSI signals
            rsi = data['RSI'].iloc[-1]
            signals['RSI'] = 1 if rsi < 30 else -1 if rsi > 70 else 0
            
            # MACD signals
            macd_diff = data['MACD_diff'].iloc[-1]
            signals['MACD'] = 1 if macd_diff > 0 else -1 if macd_diff < 0 else 0
            
            # Moving average signals
            sma_50 = data['SMA_50'].iloc[-1]
            sma_200 = data['SMA_200'].iloc[-1]
            signals['MA'] = 1 if sma_50 > sma_200 else -1 if sma_50 < sma_200 else 0
            
            return signals
        except Exception as e:
            logger.error(f"Error calculating indicator signals: {e}")
            return {}

    def _combine_indicator_signals(self, signals: Dict[str, int]) -> int:
        """Combine signals from multiple indicators."""
        try:
            if not signals:
                return 0
                
            # Calculate weighted average of signals
            weights = {'RSI': 0.3, 'MACD': 0.4, 'MA': 0.3}
            weighted_sum = sum(signals.get(ind, 0) * weights.get(ind, 0) 
                             for ind in signals)
            
            # Convert to discrete signal
            if weighted_sum > self.signal_threshold:
                return 1
            elif weighted_sum < -self.signal_threshold:
                return -1
            return 0
        except Exception as e:
            logger.error(f"Error combining indicator signals: {e}")
            return 0

    def _calculate_indicator_confidence(self, signals: Dict[str, int]) -> float:
        """Calculate confidence score from indicator signals."""
        try:
            if not signals:
                return 0.0
                
            # Calculate agreement between indicators
            signal_values = list(signals.values())
            agreement = len([s for s in signal_values if s == signal_values[0]])
            
            return agreement / len(signal_values)
        except Exception as e:
            logger.error(f"Error calculating indicator confidence: {e}")
            return 0.0

    def calculate_confidence_score(
        self,
        data: pd.DataFrame,
        market_condition: str,
        prediction_probability: float
    ) -> float:
        """Calculate overall confidence score for signal."""
        try:
            # Get individual confidence components
            ml_confidence = abs(prediction_probability - 0.5) * 2
            market_confidence = self.confidence_calculator.calculate_confidence_score(
                data, market_condition
            )
            
            # Combine confidence scores (weighted average)
            combined_confidence = (
                0.6 * ml_confidence +
                0.4 * market_confidence
            )
            
            return min(max(combined_confidence, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error calculating confidence score: {e}")
            return 0.0