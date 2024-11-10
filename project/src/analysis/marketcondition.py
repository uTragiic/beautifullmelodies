# Standard Library Imports
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
import pandas as pd
from enum import Enum
from datetime import datetime
# Set up logging
logger = logging.getLogger(__name__)

class MarketRegimeType(Enum):
    """Enumeration of market regime types"""
    BULLISH_TRENDING = "bullish_trending"
    BEARISH_TRENDING = "bearish_trending"
    LOW_VOLATILITY = "low_volatility"
    HIGH_VOLATILITY = "high_volatility"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    BREAKDOWN = "breakdown"

@dataclass
class MarketStats:
    """Data class for market condition statistics"""
    win_rate: float = 0.0
    avg_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    trade_count: int = 0
    avg_trade_duration: float = 0.0
    profit_factor: float = 0.0

class MarketConditionAnalyzer:
    """
    Enhanced analyzer for market conditions with detailed statistics tracking
    and regime detection capabilities.
    """
    
    def __init__(self, 
                 volatility_window: int = 20,
                 trend_window: int = 50,
                 breakout_threshold: float = 2.0,
                 regime_change_threshold: float = 0.2):
        """
        Initialize the market condition analyzer.
        
        Args:
            volatility_window: Window for volatility calculations
            trend_window: Window for trend detection
            breakout_threshold: Threshold for breakout detection (in standard deviations)
            regime_change_threshold: Threshold for regime change detection
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.breakout_threshold = breakout_threshold
        self.regime_change_threshold = regime_change_threshold
        
        self.condition_stats: Dict[str, MarketStats] = defaultdict(MarketStats)
        self.regime_history: List[Tuple[datetime, MarketRegimeType]] = []
        self.condition_transitions: Dict[Tuple[str, str], int] = defaultdict(int)
        self._current_regime: Optional[MarketRegimeType] = None

        # Define required columns for validation
        self._required_columns = {
            'close', 'SMA_50', 'SMA_200', 'BB_upper', 'BB_lower', 
            'ADX', 'RSI', 'MACD_diff', 'Stoch_K', 'ATR', 
            'volume', 'Volume_Ratio'
        }

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate that all required indicators and data properties are present and valid.
        
        Args:
            data: DataFrame with price and indicator data
            
        Raises:
            ValueError: If data validation fails
        """
        try:
            # Check for required columns
            missing_columns = self.required_indicators - set(data.columns)
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Check for required price columns
            price_columns = {'open', 'high', 'low', 'close', 'volume'}
            missing_price_cols = price_columns - set(data.columns)
            if missing_price_cols:
                raise ValueError(f"Missing required price columns: {missing_price_cols}")

            # Check for NaN values
            nan_check = data[list(self.required_indicators)].isna()
            if nan_check.any().any():
                problematic_cols = nan_check.any()[nan_check.any()].index.tolist()
                nan_counts = nan_check.sum()[nan_check.sum() > 0]
                raise ValueError(
                    f"NaN values found in indicators: "
                    f"{dict(zip(problematic_cols, nan_counts))}"
                )

            # Check for infinite values
            inf_check = np.isinf(data[list(self.required_indicators)])
            if inf_check.any().any():
                problematic_cols = inf_check.any()[inf_check.any()].index.tolist()
                raise ValueError(f"Infinite values found in indicators: {problematic_cols}")

            # Check data length
            min_required_length = 200  # Based on longest indicator lookback
            if len(data) < min_required_length:
                raise ValueError(
                    f"Insufficient data points. Required: {min_required_length}, "
                    f"Got: {len(data)}"
                )

            # Check data freshness
            max_staleness_days = 1  # Maximum allowed age of most recent data point
            latest_date = pd.to_datetime(data.index[-1])
            staleness = (pd.Timestamp.now() - latest_date).days
            if staleness > max_staleness_days:
                raise ValueError(
                    f"Data is too stale. Latest point is {staleness} days old. "
                    f"Maximum allowed: {max_staleness_days}"
                )

            # Verify indicator ranges
            validation_rules = {
                'RSI': (0, 100),
                'ADX': (0, 100),
                'Stoch_K': (0, 100),
                'Volume_Ratio': (0, np.inf)
            }

            for indicator, (min_val, max_val) in validation_rules.items():
                values = data[indicator]
                if (values < min_val).any() or (values > max_val).any():
                    invalid_count = ((values < min_val) | (values > max_val)).sum()
                    raise ValueError(
                        f"{indicator} has {invalid_count} values outside valid "
                        f"range [{min_val}, {max_val}]"
                    )

            self.logger.info("Data validation completed successfully")

        except Exception as e:
            self.logger.error(f"Data validation failed: {str(e)}")
            raise 
        
    def determine_market_condition(self, data: pd.DataFrame) -> str:
        """
        Determine the current market condition with enhanced analysis.
        
        Args:
            data: DataFrame with price and indicator data
            
        Returns:
            String describing the market condition
        """
        try:
            self._validate_data(data)
            last_row = data.iloc[-1]
            
            # Calculate trend components
            trend_direction = self._determine_trend_direction(data)
            trend_strength = self._determine_trend_strength(data)
            
            # Calculate volatility components
            volatility_condition = self._determine_volatility_condition(data)
            
            # Calculate momentum and volume components
            volume_condition = self._determine_volume_condition(data)
            momentum_condition = self._determine_momentum_condition(data)
            
            # Determine market regime
            regime = self._determine_regime(data)
            self._update_regime_history(regime)
            
            # Combine conditions
            market_condition = (
                f"{trend_direction}-{trend_strength}-"
                f"{volatility_condition}-{volume_condition}-"
                f"{momentum_condition}"
            )
            
            # Update statistics
            self._update_condition_stats(market_condition, data)
            
            return market_condition
            
        except Exception as e:
            logger.error(f"Error determining market condition: {e}")
            raise

    def get_condition_statistics(self, condition: str) -> MarketStats:
        """
        Get detailed statistics for a specific market condition.
        
        Args:
            condition: Market condition identifier
            
        Returns:
            MarketStats object with condition statistics
        """
        return self.condition_stats[condition]

    def get_regime_transitions(self) -> pd.DataFrame:
        """
        Get regime transition probability matrix.
        
        Returns:
            DataFrame containing regime transition probabilities
        """
        regimes = list(MarketRegimeType)
        transition_matrix = pd.DataFrame(0.0, index=regimes, columns=regimes)
        
        for (from_regime, to_regime), count in self.condition_transitions.items():
            total_from = sum(1 for r1, r2 in self.condition_transitions.keys() if r1 == from_regime)
            if total_from > 0:
                transition_matrix.loc[from_regime, to_regime] = count / total_from
                
        return transition_matrix

    def get_regime_duration_stats(self) -> Dict[MarketRegimeType, Dict[str, float]]:
        """
        Calculate statistics about regime durations.
        
        Returns:
            Dictionary containing duration statistics for each regime
        """
        regime_durations = defaultdict(list)
        
        for i in range(1, len(self.regime_history)):
            prev_time, prev_regime = self.regime_history[i-1]
            curr_time, curr_regime = self.regime_history[i]
            
            if prev_regime != curr_regime:
                duration = (curr_time - prev_time).total_seconds() / 86400  # Convert to days
                regime_durations[prev_regime].append(duration)
                
        stats = {}
        for regime, durations in regime_durations.items():
            if durations:
                stats[regime] = {
                    'avg_duration': np.mean(durations),
                    'max_duration': np.max(durations),
                    'min_duration': np.min(durations),
                    'std_duration': np.std(durations)
                }
                
        return stats

    def _determine_regime(self, data: pd.DataFrame) -> MarketRegimeType:
        """
        Determine the current market regime using multiple factors.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Current market regime type
        """
        # Calculate trend strength
        sma_50 = data['SMA_50'].iloc[-1]
        sma_200 = data['SMA_200'].iloc[-1]
        trend_strength = abs((sma_50 - sma_200) / sma_200)
        
        # Calculate volatility
        returns = data['close'].pct_change()
        current_vol = returns.rolling(window=self.volatility_window).std().iloc[-1]
        avg_vol = returns.rolling(window=self.volatility_window*2).std().mean()
        
        # Detect breakouts/breakdowns
        price = data['close'].iloc[-1]
        bollinger_upper = data['BB_upper'].iloc[-1]
        bollinger_lower = data['BB_lower'].iloc[-1]
        
        if price > bollinger_upper and trend_strength > self.regime_change_threshold:
            return MarketRegimeType.BREAKOUT
        elif price < bollinger_lower and trend_strength > self.regime_change_threshold:
            return MarketRegimeType.BREAKDOWN
        elif current_vol > avg_vol * self.breakout_threshold:
            return MarketRegimeType.HIGH_VOLATILITY
        elif current_vol < avg_vol / self.breakout_threshold:
            return MarketRegimeType.LOW_VOLATILITY
        elif trend_strength > self.regime_change_threshold:
            return MarketRegimeType.BULLISH_TRENDING if sma_50 > sma_200 else MarketRegimeType.BEARISH_TRENDING
        else:
            return MarketRegimeType.RANGING

    def _update_regime_history(self, new_regime: MarketRegimeType) -> None:
        """Update regime history and transition counts."""
        current_time = pd.Timestamp.now()
        self.regime_history.append((current_time, new_regime))
        
        if self._current_regime and self._current_regime != new_regime:
            self.condition_transitions[(self._current_regime, new_regime)] += 1
            
        self._current_regime = new_regime

    def _update_condition_stats(self, condition: str, data: pd.DataFrame) -> None:
        """Update statistics for the current market condition."""
        returns = data['close'].pct_change().dropna()
        
        # Calculate basic metrics
        stats = self.condition_stats[condition]
        stats.volatility = returns.std() * np.sqrt(252)
        stats.avg_return = returns.mean() * 252
        
        if len(returns) > 1:
            stats.sharpe_ratio = stats.avg_return / (stats.volatility + 1e-6)
        
        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        stats.max_drawdown = abs(drawdowns.min())
        
        # Update trade metrics
        stats.trade_count += 1
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        if len(positive_returns) > 0 and len(negative_returns) > 0:
            stats.profit_factor = abs(positive_returns.sum() / negative_returns.sum())
            stats.win_rate = len(positive_returns) / len(returns)

    def analyze_market_dynamics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive market dynamics analysis.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Dictionary containing market dynamics analysis
        """
        analysis = {}
        
        # Trend analysis
        analysis['trend'] = {
            'strength': self._calculate_trend_strength(data),
            'consistency': self._calculate_trend_consistency(data),
            'momentum': self._calculate_momentum_strength(data)
        }
        
        # Volatility analysis
        analysis['volatility'] = {
            'current': self._calculate_current_volatility(data),
            'regime': self._calculate_volatility_regime(data),
            'trend': self._calculate_volatility_trend(data)
        }
        
        # Volume analysis
        analysis['volume'] = {
            'trend': self._calculate_volume_trend(data),
            'relative_strength': self._calculate_volume_strength(data)
        }
        
        # Market efficiency
        analysis['efficiency'] = {
            'ratio': self._calculate_efficiency_ratio(data),
            'fractal_dimension': self._calculate_fractal_dimension(data)
        }
        
        return analysis

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength using multiple indicators."""
        ma_ratios = data['SMA_50'] / data['SMA_200']
        adx = data['ADX']
        
        trend_strength = (
            0.6 * (ma_ratios.iloc[-1] - 1) +
            0.4 * (adx.iloc[-1] / 100)
        )
        
        return abs(trend_strength)

    def _calculate_trend_consistency(self, data: pd.DataFrame) -> float:
        """Calculate trend consistency metric."""
        returns = data['close'].pct_change()
        positive_days = (returns > 0).rolling(window=20).mean()
        return abs(positive_days.iloc[-1] - 0.5) * 2

    def _calculate_momentum_strength(self, data: pd.DataFrame) -> float:
        """Calculate momentum strength using multiple indicators."""
        rsi = (data['RSI'].iloc[-1] - 50) / 50
        macd = data['MACD_diff'].iloc[-1] / data['close'].iloc[-1]
        stoch = (data['Stoch_K'].iloc[-1] - 50) / 50
        
        return np.mean([rsi, macd, stoch])

    def _calculate_current_volatility(self, data: pd.DataFrame) -> float:
        """Calculate current volatility level."""
        return data['ATR'].iloc[-1] / data['close'].iloc[-1]

    def _calculate_volatility_regime(self, data: pd.DataFrame) -> str:
        """Determine current volatility regime."""
        current_vol = self._calculate_current_volatility(data)
        historical_vol = data['ATR'].rolling(window=100).mean().iloc[-1] / data['close'].iloc[-1]
        
        if current_vol > historical_vol * 1.5:
            return "high"
        elif current_vol < historical_vol * 0.5:
            return "low"
        else:
            return "normal"

    def _calculate_volatility_trend(self, data: pd.DataFrame) -> float:
        """Calculate trend in volatility."""
        vol = data['ATR'] / data['close']
        vol_sma = vol.rolling(window=20).mean()
        return (vol_sma.iloc[-1] / vol_sma.iloc[-20]) - 1

    def _calculate_volume_trend(self, data: pd.DataFrame) -> float:
        """Calculate volume trend strength."""
        vol_sma = data['volume'].rolling(window=20).mean()
        return (vol_sma.iloc[-1] / vol_sma.iloc[-20]) - 1

    def _calculate_volume_strength(self, data: pd.DataFrame) -> float:
        """Calculate relative volume strength."""
        return data['Volume_Ratio'].iloc[-1] - 1

    def _calculate_efficiency_ratio(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate market efficiency ratio."""
        price_change = abs(data['close'].iloc[-1] - data['close'].iloc[-window])
        path_length = sum(abs(data['close'].diff().iloc[-window:]))
        return price_change / path_length if path_length != 0 else 0

    def _calculate_fractal_dimension(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate fractal dimension of price movement."""
        prices = data['close'].iloc[-window:]
        n = len(prices)
        
        if n < 2:
            return 1.0
            
        ranges = []
        for k in range(2, min(11, n)):
            subset_ranges = []
            for i in range(0, n-k+1):
                subset = prices.iloc[i:i+k]
                subset_ranges.append(max(subset) - min(subset))
            ranges.append(np.mean(subset_ranges))
            
        if not ranges:
            return 1.0
            
        x = np.log(range(2, len(ranges)+2))
        y = np.log(ranges)
        
        slope, _ = np.polyfit(x, y, 1)
        return 2 - slope