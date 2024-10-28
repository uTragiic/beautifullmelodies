# Standard Library Imports
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
from collections import deque, OrderedDict

# Third-Party Imports
from arch import arch_model  # New import

# Local Imports
from ..utils.validation import validate_dataframe  # Assuming this function is available

logger = logging.getLogger(__name__)

class VolatilityRegime(Enum):
    """Enumeration of volatility regime states"""
    EXTREMELY_LOW = "extremely_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREMELY_HIGH = "extremely_high"
    INCREASING = "increasing"
    DECREASING = "decreasing"

@dataclass
class VolatilityMetrics:
    """Data class for volatility metrics"""
    # Document-required metrics
    current_volatility: float
    historical_volatility: float
    atr_price_ratio: float  # Added from document requirement
    atr_price_percentile: float  # Added from document requirement
    regime: VolatilityRegime

    # Enhanced metrics
    implied_volatility: Optional[float]
    forecast: float
    confidence: float
    trend: float
    relative_vol: float
    vol_of_vol: float
    skew: float
    kurtosis: float

class VolatilityAnalyzer:
    """
    Advanced volatility analysis with regime detection, forecasting,
    and dynamic threshold adjustment capabilities.
    """

    def __init__(self, 
                lookback_period: int = 252,
                estimation_window: int = 20,
                vol_threshold: float = 2.0,
                regime_threshold: float = 0.5,
                high_vol_percentile: float = 0.8,  # Added for document requirement (80th percentile)
                decay_factor: float = 0.94):
        """
        Initialize the volatility analyzer.

        Args:
            lookback_period: Historical data period for analysis
            estimation_window: Window for volatility estimation
            vol_threshold: Threshold for regime changes
            regime_threshold: Threshold for regime classification
            high_vol_percentile: Percentile threshold for high volatility (document requirement)
            decay_factor: Decay factor for exponential weighting
        """
        self.lookback_period = lookback_period
        self.estimation_window = estimation_window
        self.vol_threshold = vol_threshold
        self.regime_threshold = regime_threshold
        self.high_vol_percentile = high_vol_percentile
        self.decay_factor = decay_factor

        # Initialize history tracking with maximum length
        self.max_history = 1000  # Define maximum history length
        self.regime_history = deque(maxlen=self.max_history)
        self.forecast_history = OrderedDict()
        self.metrics_history = OrderedDict()

    def analyze_volatility(self, data: pd.DataFrame) -> VolatilityMetrics:
        """
        Perform comprehensive volatility analysis.

        Args:
            data: Market data DataFrame

        Returns:
            VolatilityMetrics containing analysis results
        """
        try:
            # Validate data
            required_columns = ['ATR', 'close']
            validate_dataframe(data, required_columns)

            # Handle missing data
            data = data.dropna(subset=required_columns)

            # Document-required calculations
            atr_price_ratio = self._calculate_atr_price_ratio(data)
            atr_price_percentile = self._calculate_atr_percentile(data)

            # Calculate basic volatility measures
            current_vol = self._calculate_current_volatility(data)
            hist_vol = self._calculate_historical_volatility(data)

            # Determine volatility regime
            regime = self._determine_volatility_regime(
                current_vol, 
                hist_vol,
                atr_price_ratio,
                atr_price_percentile
            )

            # Calculate additional metrics
            forecast = self._forecast_volatility(data)
            confidence = self._calculate_forecast_confidence(data)
            trend = self._calculate_volatility_trend(data)
            relative_vol = current_vol / hist_vol if hist_vol != 0 else 1.0
            vol_of_vol = self._calculate_volatility_of_volatility(data)

            # Calculate higher moments
            returns = data['close'].pct_change().dropna()
            skewness = stats.skew(returns)
            kurt = stats.kurtosis(returns)

            # Create metrics object
            metrics = VolatilityMetrics(
                current_volatility=current_vol,
                historical_volatility=hist_vol,
                atr_price_ratio=atr_price_ratio,
                atr_price_percentile=atr_price_percentile,
                regime=regime,
                implied_volatility=None,  # Would require options data
                forecast=forecast,
                confidence=confidence,
                trend=trend,
                relative_vol=relative_vol,
                vol_of_vol=vol_of_vol,
                skew=skewness,
                kurtosis=kurt
            )

            # Update history
            self._update_history(data.index[-1], metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error analyzing volatility: {e}")
            raise

    def _calculate_atr_price_ratio(self, data: pd.DataFrame) -> float:
        """
        Calculate ATR/Price ratio as specified in document.

        Args:
            data: Market data DataFrame

        Returns:
            Current ATR/Price ratio
        """
        try:
            current_atr = data['ATR'].iloc[-1]
            current_price = data['close'].iloc[-1]
            if pd.isna(current_atr) or pd.isna(current_price):
                return np.nan
            return current_atr / current_price
        except Exception as e:
            logger.error(f"Error calculating ATR/Price ratio: {e}")
            raise

    def _calculate_atr_percentile(self, data: pd.DataFrame) -> float:
        """
        Calculate percentile of current ATR/Price ratio.

        Args:
            data: Market data DataFrame

        Returns:
            Percentile of current ATR/Price ratio
        """
        try:
            atr_price_ratios = (data['ATR'] / data['close']).dropna()
            if len(atr_price_ratios) == 0:
                return np.nan
            current_ratio = atr_price_ratios.iloc[-1]
            return float(stats.percentileofscore(atr_price_ratios, current_ratio) / 100)
        except Exception as e:
            logger.error(f"Error calculating ATR percentile: {e}")
            raise

    def _calculate_current_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current volatility using exponential weighting."""
        returns = data['close'].pct_change().dropna()
        returns = returns.tail(self.estimation_window)
        if len(returns) < self.estimation_window:
            return 0.0
        weights = np.array([self.decay_factor ** i for i in reversed(range(self.estimation_window))])
        weights = weights / weights.sum()

        recent_returns = returns.iloc[-self.estimation_window:]
        weighted_squares = (recent_returns ** 2 * weights).sum()
        return np.sqrt(weighted_squares * 252)

    def _calculate_historical_volatility(self, data: pd.DataFrame) -> float:
        """Calculate historical volatility."""
        returns = data['close'].pct_change().dropna()
        returns = returns.tail(self.lookback_period)
        if len(returns) == 0:
            return 0.0
        return returns.std() * np.sqrt(252)

    def _determine_volatility_regime(self, 
                                   current_vol: float, 
                                   hist_vol: float,
                                   atr_price_ratio: float,
                                   atr_percentile: float) -> VolatilityRegime:
        """
        Determine the current volatility regime using multiple factors.
        Incorporates both traditional volatility measures and document-specified ATR ratio.
        """
        try:
            # Check ATR/Price ratio threshold first (document requirement)
            if atr_percentile >= self.high_vol_percentile:
                if current_vol > hist_vol * self.vol_threshold:
                    return VolatilityRegime.EXTREMELY_HIGH
                return VolatilityRegime.HIGH

            # Additional regime classification
            vol_ratio = current_vol / hist_vol if hist_vol != 0 else 1.0

            if vol_ratio > self.vol_threshold * 2:
                return VolatilityRegime.EXTREMELY_HIGH
            elif vol_ratio > self.vol_threshold:
                return VolatilityRegime.HIGH
            elif vol_ratio < 1 / (self.vol_threshold * 2):
                return VolatilityRegime.EXTREMELY_LOW
            elif vol_ratio < 1 / self.vol_threshold:
                return VolatilityRegime.LOW
            else:
                return VolatilityRegime.NORMAL

        except Exception as e:
            logger.error(f"Error determining volatility regime: {e}")
            raise

    def calculate_position_adjustment(self, 
                                   current_volatility: float,
                                   average_volatility: float,
                                   adjustment_factor: float = 0.1) -> float:
        """
        Calculate position size adjustment based on document formula.

        Args:
            current_volatility: Current volatility level
            average_volatility: Average historical volatility
            adjustment_factor: Volatility adjustment factor

        Returns:
            Position size adjustment multiplier
        """
        try:
            # Implement exact formula from document
            volatility_ratio = current_volatility / average_volatility
            volatility_adjustment = 1 + (volatility_ratio - 1) * adjustment_factor
            return volatility_adjustment
        except Exception as e:
            logger.error(f"Error calculating position adjustment: {e}")
            raise

    def _forecast_volatility(self, data: pd.DataFrame) -> float:
        """
        Forecast future volatility using GARCH(1,1) model.
        """
        try:
            returns = data['close'].pct_change().dropna() * 100  # Convert to percentage
            returns = returns.tail(self.lookback_period)
            if len(returns) < self.estimation_window:
                return self._calculate_current_volatility(data)

            model = arch_model(returns, mean='Zero', vol='GARCH', p=1, q=1)
            res = model.fit(disp='off')
            forecast = res.forecast(horizon=1)
            forecast_volatility = np.sqrt(forecast.variance.iloc[-1].values[0]) * np.sqrt(252)
            return forecast_volatility / 100  # Convert back from percentage

        except Exception as e:
            logger.error(f"Error forecasting volatility: {e}")
            return self._calculate_current_volatility(data)

    def _calculate_forecast_confidence(self, data: pd.DataFrame) -> float:
        """
        Calculate confidence level in volatility forecast.
        """
        if not self.forecast_history:
            return 0.5

        # Calculate forecast accuracy
        actual_vols = []
        forecasted_vols = []

        for timestamp, forecast in self.forecast_history.items():
            if timestamp in data.index:
                future_data = data.loc[timestamp:].iloc[:20]
                future_returns = future_data['close'].pct_change().dropna()
                if len(future_returns) >= 1:
                    actual_vol = future_returns.std() * np.sqrt(252)
                    actual_vols.append(actual_vol)
                    forecasted_vols.append(forecast)

        if not actual_vols:
            return 0.5

        # Calculate RMSE and convert to confidence score
        errors = np.array(actual_vols) - np.array(forecasted_vols)
        rmse = np.sqrt(np.mean(errors ** 2))
        return 1 / (1 + rmse)

    def _calculate_volatility_trend(self, data: pd.DataFrame) -> float:
        """
        Calculate trend in volatility using the Mann-Kendall trend test.
        """
        vol_series = data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        vol_series = vol_series.dropna()
        if len(vol_series) < 10:
            return 0.0

        # Mann-Kendall trend test
        tau, p_value = stats.kendalltau(np.arange(len(vol_series)), vol_series)
        return tau  # Return the Kendall Tau coefficient as trend indicator

    def _calculate_volatility_of_volatility(self, data: pd.DataFrame) -> float:
        """Calculate volatility of volatility."""
        vol_series = data['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        vol_series = vol_series.dropna()
        return vol_series.std()

    def _update_history(self, timestamp: pd.Timestamp, metrics: VolatilityMetrics) -> None:
        """Update historical records."""
        self.regime_history.append((timestamp, metrics.regime))
        self.forecast_history[timestamp] = metrics.forecast
        self.metrics_history[timestamp] = metrics

        # Maintain forecast_history length
        if len(self.forecast_history) > self.max_history:
            self.forecast_history.popitem(last=False)  # Pop the oldest item

        if len(self.metrics_history) > self.max_history:
            self.metrics_history.popitem(last=False)

    def clear_cache(self, older_than_days: Optional[int] = None) -> None:
        """
        Clear historical data cache to manage memory usage.

        Args:
            older_than_days: Optional, clear only data older than this many days
        """
        try:
            if older_than_days is None:
                self.regime_history.clear()
                self.forecast_history.clear()
                self.metrics_history.clear()
            else:
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=older_than_days)
                self.regime_history = deque(
                    [(t, r) for t, r in self.regime_history if t > cutoff],
                    maxlen=self.max_history
                )
                self.forecast_history = OrderedDict(
                    (k, v) for k, v in self.forecast_history.items() if k > cutoff
                )
                self.metrics_history = OrderedDict(
                    (k, v) for k, v in self.metrics_history.items() if k > cutoff
                )
            logger.info(f"Cache cleared: older_than_days={older_than_days}")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")


    def get_current_status(self) -> Dict[str, Any]:
        """
        Get quick summary of current volatility status.
        
        Returns:
            Dictionary with current volatility status and alerts
        """
        try:
            if not self.metrics_history:
                return {"status": "No data available"}
                
            latest_metrics = list(self.metrics_history.values())[-1]
            prev_metrics = list(self.metrics_history.values())[-2] if len(self.metrics_history) > 1 else latest_metrics
            
            return {
                "current_regime": latest_metrics.regime.value,
                "atr_price_ratio": latest_metrics.atr_price_ratio,
                "volatility_change": (latest_metrics.current_volatility / prev_metrics.current_volatility - 1) * 100,
                "alert_level": "HIGH" if latest_metrics.regime in [VolatilityRegime.HIGH, VolatilityRegime.EXTREMELY_HIGH] else "NORMAL",
                "forecast_confidence": latest_metrics.confidence,
                "requires_position_adjustment": abs(latest_metrics.current_volatility / latest_metrics.historical_volatility - 1) > 0.2
            }
            
        except Exception as e:
            logger.error(f"Error getting current status: {e}")
            return {"status": "Error", "error": str(e)}
    def get_forecast_accuracy(self) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics.
        
        Returns:
            Dictionary containing forecast accuracy metrics
        """
        if not self.forecast_history or not self.metrics_history:
            return {}
            
        forecasts = []
        actuals = []
        
        for timestamp, forecast in self.forecast_history.items():
            if timestamp in self.metrics_history:
                forecasts.append(forecast)
                actuals.append(self.metrics_history[timestamp].current_volatility)
                
        if not forecasts:
            return {}
            
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)
        
        errors = forecasts - actuals
        
        return {
            'mae': np.mean(np.abs(errors)),
            'rmse': np.sqrt(np.mean(errors**2)),
            'mape': np.mean(np.abs(errors/actuals)) * 100,
            'bias': np.mean(errors),
            'correlation': np.corrcoef(forecasts, actuals)[0,1]
        }

    def analyze_volatility_regimes(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive volatility regime analysis.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary containing regime analysis results
        """
        try:
            atr_price_ratios = data['ATR'] / data['close']
            returns = data['close'].pct_change().dropna()
            
            analysis = {
                # Document-required metrics
                'atr_price_threshold': np.percentile(atr_price_ratios, self.high_vol_percentile * 100),
                'current_atr_ratio': atr_price_ratios.iloc[-1],
                'is_high_volatility': atr_price_ratios.iloc[-1] > np.percentile(atr_price_ratios, self.high_vol_percentile * 100),
                
                # Enhanced metrics
                'regime_stability': self._calculate_regime_stability(),
                'regime_duration': self._calculate_regime_durations(),
                'transition_probabilities': self.get_regime_transitions().to_dict(),
                
                # Statistical properties
                'distribution': {
                    'skewness': stats.skew(returns),
                    'kurtosis': stats.kurtosis(returns),
                    'jarque_bera': stats.jarque_bera(returns)[0],
                    'is_normal': stats.normaltest(returns)[1] > 0.05
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in volatility regime analysis: {e}")
            raise

    def _calculate_regime_stability(self) -> float:
        """
        Calculate stability score for current volatility regime.
        
        Returns:
            Stability score between 0 and 1
        """
        if len(self.regime_history) < 2:
            return 1.0
            
        # Calculate number of regime changes
        changes = sum(1 for i in range(1, len(self.regime_history))
                     if self.regime_history[i][1] != self.regime_history[i-1][1])
                     
        # Return stability score (1 - change_ratio)
        return 1 - (changes / (len(self.regime_history) - 1))

    def _calculate_regime_durations(self) -> Dict[VolatilityRegime, Dict[str, float]]:
        """
        Calculate statistics about regime durations.
        
        Returns:
            Dictionary containing duration statistics for each regime
        """
        if len(self.regime_history) < 2:
            return {}
            
        regime_durations = {}
        current_regime = self.regime_history[0][1]
        current_start = self.regime_history[0][0]
        
        for timestamp, regime in self.regime_history[1:]:
            if regime != current_regime:
                duration = (timestamp - current_start).total_seconds() / 86400  # Convert to days
                
                if current_regime not in regime_durations:
                    regime_durations[current_regime] = []
                regime_durations[current_regime].append(duration)
                
                current_regime = regime
                current_start = timestamp
                
        # Calculate statistics for each regime
        stats = {}
        for regime, durations in regime_durations.items():
            if durations:
                stats[regime] = {
                    'avg_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'min_duration': np.min(durations),
                    'std_duration': np.std(durations),
                    'total_occurrences': len(durations)
                }
                
        return stats