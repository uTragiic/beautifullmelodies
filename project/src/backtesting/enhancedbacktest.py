# Standard Library Imports
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
# Third-Party Imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor
from ta.trend import EMAIndicator, MACD, ADXIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
# Local Imports
from ..core.performance_metrics import PerformanceMetrics

from ..core.types import MarketRegime
from ..utils.validation import validate_dataframe
import config

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration parameters for backtesting"""
    initial_capital: float = 100000.0
    commission_rate: float = 0.001  # 0.1%
    slippage: float = 0.0001  # 0.01%
    n_splits: int = 5  # Number of splits for walk-forward
    test_size: float = 0.3  # 30% for testing
    n_jobs: int = -1  # Use all CPU cores
    min_samples: int = 252  # Minimum samples for training
    max_train_samples: int = 2520  # 10 years
    batch_size: int = 32
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    min_performance_threshold: float = 0.5
    max_correlation_threshold: float = 0.7
    risk_free_rate: float = 0.02  # 2% annual risk-free rate

class Backtest:
    """
    Enhanced backtesting system with walk-forward optimization
    and Monte Carlo simulation capabilities.
    """

    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        """
        Initialize the backtesting system.
        """
        
        try:
            # Validate data
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")
            
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            data.columns = data.columns.str.lower()
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Validate numeric data
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    raise ValueError(f"Column {col} must contain numeric data")
            
            # Validate index
            if not isinstance(data.index, pd.DatetimeIndex):
                raise ValueError("DataFrame must have DatetimeIndex")
                
            # Check for duplicate timestamps
            if data.index.duplicated().any():
                raise ValueError("DataFrame contains duplicate timestamps")
                
            # Check for chronological order
            if not data.index.is_monotonic_increasing:
                raise ValueError("DataFrame timestamps must be in chronological order")
                
            # Check for NaN values
            if data[required_columns].isna().any().any():
                raise ValueError("DataFrame contains NaN values in required columns")
                
            # Validate price integrity
            valid_prices = (
                (data['high'] >= data['low']).all() and
                (data['close'] >= data['low']).all() and
                (data['close'] <= data['high']).all() and
                (data['open'] >= data['low']).all() and
                (data['open'] <= data['high']).all()
            )
            if not valid_prices:
                raise ValueError("Invalid price relationships detected")
                
            # Minimum data requirement reduced but still validate
            if len(data) < 50:  # Reduced from previous requirement
                raise ValueError(f"Insufficient data points: {len(data)} < 50")

            self.data = data
            self.initial_capital = initial_capital
            self.results = None
            self.config = config or BacktestConfig()
            # Use already calculated indicators if they exist, otherwise calculate them
            required_indicators = [
                'SMA_50', 'SMA_200', 'RSI', 'RSI_Z', 'MACD', 'MACD_signal',
                'MACD_diff', 'MACD_Z', 'ADX', 'ATR', 'Volume_Ratio', 'OBV',
                'VWAP', 'Momentum', 'Stoch_K', 'Stoch_D', 'BB_width'
            ]
            
            missing_indicators = [ind for ind in required_indicators if ind not in data.columns]
            if missing_indicators:
                # If indicators are missing, calculate them
                from ..indicators.calculator import IndicatorCalculator
                calculator = IndicatorCalculator()
                self.data = calculator.calculate_indicators(self.data)
            
            logger.info(f"Backtest initialized with {len(data)} data points")
            
        except Exception as e:
            logger.error(f"Error initializing backtest system: {e}")
            raise
    def _calculate_simulation_metrics(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate metrics for Monte Carlo simulation.
        
        Args:
            returns: DataFrame of simulation returns
            
        Returns:
            Dictionary of simulation metrics
        """
        try:
            strategy_returns = returns['Strategy_Returns'].dropna()
            
            # Calculate cumulative returns
            cumulative_returns = (1 + strategy_returns).cumprod()
            
            # Calculate metrics
            total_return = cumulative_returns.iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            
            # Calculate Sharpe ratio
            risk_free_rate: float = 0.02
            excess_returns = strategy_returns - risk_free_rate/252
            sharpe_ratio = np.mean(excess_returns) / strategy_returns.std() * np.sqrt(252)
            
            # Calculate drawdown
            peak = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - peak) / peak
            max_drawdown = abs(drawdown.min())
            
            # Calculate win rate
            winning_trades = (strategy_returns > 0).sum()
            total_trades = len(strategy_returns[strategy_returns != 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profits = strategy_returns[strategy_returns > 0].sum()
            gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            return {
                'total_return': float(total_return),
                'annual_return': float(annual_return),
                'volatility': float(volatility),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'total_trades': int(total_trades)
            }
            
        except Exception as e:
            logger.error(f"Error calculating simulation metrics: {e}")
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_trades': 0
            }
    
    def calculate_sharpe_ratio(self, excess_returns: np.ndarray, 
                             strategy_returns: np.ndarray) -> float:
        """
        Calculate Sharpe ratio with safety checks for zero division
        """
        if len(strategy_returns) == 0:
            return 0.0
            
        std = strategy_returns.std()
        if std == 0:
            self.logger.warning("Standard deviation is zero, returning 0 for Sharpe ratio")
            return 0.0
            
        return np.mean(excess_returns) / std * np.sqrt(252)

    def validate_data(self, data: np.ndarray) -> bool:
        """
        Validate if data meets minimum requirements
        """
        if len(data) < self.min_required_rows:
            self.logger.error(
                f"Insufficient data: {len(data)} rows; at least {self.min_required_rows} required"
            )
            return False
        return True
    def _calculate_monte_carlo_metrics(self, returns: pd.DataFrame) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics for Monte Carlo simulation."""
        try:
            strategy_returns = returns['Strategy_Returns'].dropna()
            
            # Calculate basic metrics
            total_return = (1 + strategy_returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
            volatility = strategy_returns.std() * np.sqrt(252)
            
            # Calculate drawdown
            cum_returns = (1 + strategy_returns).cumprod()
            peak = cum_returns.expanding(min_periods=1).max()
            drawdown = (cum_returns - peak) / peak
            max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
            
            # Calculate win rate
            winning_trades = (strategy_returns > 0).sum()
            total_trades = len(strategy_returns[strategy_returns != 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profits = strategy_returns[strategy_returns > 0].sum()
            gross_losses = abs(strategy_returns[strategy_returns < 0].sum())
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            # Calculate risk-adjusted returns
            excess_returns = strategy_returns - (self.config.risk_free_rate / 252)
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / strategy_returns.std() if strategy_returns.std() != 0 else 0
            
            # Calculate additional metrics
            downside_returns = strategy_returns[strategy_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0
            calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else float('inf')
            
            # Create and return PerformanceMetrics object
            return PerformanceMetrics(
                sharpe_ratio=sharpe_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                volatility=volatility,
                total_return=total_return,
                annual_return=annual_return,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                total_trades=total_trades,
                alpha=excess_returns.mean() * 252,
                beta=self._calculate_beta(strategy_returns),
                information_ratio=self._calculate_information_ratio(strategy_returns),
                recovery_factor=abs(total_return / max_drawdown) if max_drawdown != 0 else float('inf')
            )
            
        except Exception as e:
            logger.error(f"Error calculating Monte Carlo metrics: {e}")
            raise
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate beta relative to market returns."""
        try:
            market_returns = self.data['close'].pct_change().dropna()
            if len(market_returns) != len(returns):
                market_returns = market_returns.iloc[-len(returns):]
            
            covariance = returns.cov(market_returns)
            market_variance = market_returns.var()
            
            return covariance / market_variance if market_variance != 0 else 1.0
            
        except Exception:
            return 1.0

    def calculate_trading_statistics(self) -> Dict[str, float]:
        """
        Calculate comprehensive trading statistics from backtest results.
        
        Returns:
            Dictionary containing various trading statistics
        """
        try:
            if self.results is None:
                raise ValueError("Backtest hasn't been run yet. Call run_backtest() first.")
                
            stats = {}
            returns = self.results['Strategy_Returns'].dropna()
            
            # Basic trade statistics
            trades = self.results[self.results['Signal'] != 0]
            stats['total_trades'] = len(trades)
            
            if stats['total_trades'] > 0:
                # Win rate and trade metrics
                winning_trades = returns[returns > 0]
                losing_trades = returns[returns < 0]
                
                stats['win_rate'] = len(winning_trades) / stats['total_trades']
                stats['avg_win'] = winning_trades.mean() if len(winning_trades) > 0 else 0
                stats['avg_loss'] = losing_trades.mean() if len(losing_trades) > 0 else 0
                
                # Calculate profit factor
                gross_profits = winning_trades.sum()
                gross_losses = abs(losing_trades.sum())
                stats['profit_factor'] = gross_profits / gross_losses if gross_losses != 0 else float('inf')
                
                # Trade duration
                trade_durations = []
                current_trade_start = None
                
                for idx, row in self.results.iterrows():
                    if row['Signal'] != 0 and current_trade_start is None:
                        current_trade_start = idx
                    elif row['Signal'] == 0 and current_trade_start is not None:
                        trade_duration = (idx - current_trade_start).days
                        trade_durations.append(trade_duration)
                        current_trade_start = None
                
                stats['avg_trade_duration'] = np.mean(trade_durations) if trade_durations else 0
                
                # Consecutive wins/losses
                consecutive_wins = 0
                consecutive_losses = 0
                max_consecutive_wins = 0
                max_consecutive_losses = 0
                current_streak = 0
                
                for ret in returns:
                    if ret > 0:
                        if current_streak >= 0:
                            current_streak += 1
                        else:
                            current_streak = 1
                        max_consecutive_wins = max(max_consecutive_wins, current_streak)
                    elif ret < 0:
                        if current_streak <= 0:
                            current_streak -= 1
                        else:
                            current_streak = -1
                        max_consecutive_losses = max(max_consecutive_losses, abs(current_streak))
                
                stats['max_consecutive_wins'] = max_consecutive_wins
                stats['max_consecutive_losses'] = max_consecutive_losses
                
                # Calculate recovery factor
                cumulative_returns = (1 + returns).cumprod()
                max_drawdown = self.results['Drawdown'].max()
                total_return = cumulative_returns.iloc[-1] - 1
                
                stats['recovery_factor'] = abs(total_return / max_drawdown) if max_drawdown != 0 else float('inf')
                
                # Risk-adjusted returns
                stats['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
                
                # Downside volatility
                downside_returns = returns[returns < 0]
                stats['downside_volatility'] = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
                
                # Calculate Sortino ratio
                stats['sortino_ratio'] = (returns.mean() * 252) / stats['downside_volatility'] if stats['downside_volatility'] != 0 else 0
                
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating trading statistics: {e}")
            raise
    def calculate_market_condition_statistics(self, backtest_results: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate performance statistics grouped by market conditions.
        
        Args:
            backtest_results: DataFrame containing backtest results with Market_Condition column
            
        Returns:
            DataFrame containing statistics for each market condition
        """
        try:
            # First group by market condition
            grouped = backtest_results.groupby('Market_Condition')
            
            # Initialize results dictionary
            stats_dict = {}
            
            # Calculate metrics for each market condition
            for condition in grouped.groups.keys():
                condition_data = backtest_results[backtest_results['Market_Condition'] == condition]
                
                # Calculate returns statistics
                returns = condition_data['Strategy_Returns']
                if len(returns) > 0:
                    avg_return = returns.mean() * 252  # Annualized
                    std_dev = returns.std() * np.sqrt(252)  # Annualized
                    
                    # Calculate Drawdown
                    cum_returns = (1 + returns).cumprod()
                    running_max = cum_returns.expanding().max()
                    drawdown = (cum_returns - running_max) / running_max
                    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
                    
                    # Calculate win rate
                    win_rate = (returns > 0).mean()
                    
                    # Store metrics
                    stats_dict[condition] = {
                        'Avg_Return': avg_return,
                        'Std_Dev': std_dev,
                        'Count': len(returns),
                        'Win_Rate': win_rate,
                        'Max_Drawdown': max_drawdown
                    }
                    
                    # Calculate Sharpe Ratio if std_dev is not zero
                    if std_dev != 0:
                        stats_dict[condition]['Sharpe_Ratio'] = avg_return / std_dev
                    else:
                        stats_dict[condition]['Sharpe_Ratio'] = np.nan
                
            # Convert dictionary to DataFrame
            market_condition_stats = pd.DataFrame.from_dict(stats_dict, orient='index')
            
            # Ensure all required columns exist
            required_columns = ['Avg_Return', 'Std_Dev', 'Count', 'Win_Rate', 'Max_Drawdown', 'Sharpe_Ratio']
            for col in required_columns:
                if col not in market_condition_stats.columns:
                    market_condition_stats[col] = np.nan
                    
            return market_condition_stats
            
        except Exception as e:
            logger.error(f"Error calculating market condition statistics: {e}")
            raise
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required technical indicators consistently with other classes.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated indicators
        """
        try:
            df = data.copy()
            
            # Moving Averages
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()
            
            # RSI and normalized RSI
            rsi = RSIIndicator(close=df['close'], window=14)
            df['RSI'] = rsi.rsi()
            # Calculate RSI Z-score
            df['RSI_Z'] = (df['RSI'] - df['RSI'].rolling(window=50).mean()) / df['RSI'].rolling(window=50).std()
            
            # MACD Components
            macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            # Calculate MACD Z-score
            df['MACD_Z'] = (df['MACD_diff'] - df['MACD_diff'].rolling(window=50).mean()) / df['MACD_diff'].rolling(window=50).std()
            
            # ADX
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['ADX'] = adx.adx()
            
            # ATR
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['ATR'] = atr.average_true_range()
            
            # Volume Indicators
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
            
            # On-Balance Volume
            obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
            df['OBV'] = obv.on_balance_volume()
            
            # VWAP
            vwap = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
            df['VWAP'] = vwap.volume_weighted_average_price()
            
            # Stochastic Oscillator
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']
            
            # Momentum
            df['Momentum'] = df['close'].pct_change(periods=20)
            
            # Forward fill any NaN values
            df = df.fillna(method='ffill')
            
            # Verify all required indicators are present
            required_indicators = [
                'SMA_50', 'SMA_200', 'RSI', 'RSI_Z', 'MACD', 'MACD_signal', 
                'MACD_diff', 'MACD_Z', 'ADX', 'ATR', 'OBV', 'VWAP', 'Stoch_K', 
                'Stoch_D', 'BB_width', 'Volume_Ratio', 'Momentum'
            ]
            
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            if missing_indicators:
                raise ValueError(f"Failed to calculate indicators: {missing_indicators}")
            
            # Log successful calculation
            logger.info("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise

    def run_backtest(self, strategy, n_splits: int = 5, n_jobs: int = -1) -> pd.DataFrame:
        """
        Run backtesting with walk-forward optimization.
        
        Args:
            strategy: Trading strategy to test
            n_splits: Number of time series splits
            n_jobs: Number of parallel jobs
                
        Returns:
            DataFrame containing backtest results
        """
        try:
            # Validate data length
            data_length = len(self.data)
            min_train_size = 200  # Minimum training size
            min_test_size = 50    # Minimum test size
            
            if data_length < (min_train_size + min_test_size):
                raise ValueError(
                    f"Insufficient data for backtesting: {data_length} points, "
                    f"need at least {min_train_size + min_test_size}"
                )

            # Calculate split sizes
            total_size = min_train_size + min_test_size
            split_size = total_size // n_splits
            
            # Initialize time series splits
            tscv = TimeSeriesSplit(
                n_splits=n_splits,
                test_size=split_size,
                gap=0
            )

            # Run parallel walk-forward optimization
            all_results = []
            for train_idx, test_idx in tscv.split(self.data):
                if len(train_idx) < min_train_size or len(test_idx) < min_test_size:
                    continue
                    
                try:
                    result = self._run_single_split(
                        train_idx, 
                        test_idx,
                        strategy
                    )
                    if result is not None and not result.empty:
                        all_results.append(result)
                except Exception as e:
                    logger.error(f"Error in split execution: {str(e)}")
                    continue

            # Ensure we have valid results
            if not all_results:
                raise ValueError("No valid results from any split")

            # Combine results and store
            self.results = pd.concat(all_results)
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error running backtest: {str(e)}")
            raise
    
    def _run_walk_forward(self, strategy: Any) -> pd.DataFrame:
        """
        Perform walk-forward optimization.
        
        Args:
            strategy: Trading strategy object
            
        Returns:
            DataFrame containing walk-forward results
        """
        try:
            tscv = TimeSeriesSplit(
                n_splits=self.config.n_splits,
                test_size=int(len(self.data) * self.config.test_size)
            )

            results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._run_single_split)(train, test, strategy)
                for train, test in tscv.split(self.data)
            )

            return pd.concat(results)

        except Exception as e:
            logger.error(f"Error in walk-forward optimization: {e}")
            raise

    def _run_single_split(self,
                         train_index: np.ndarray,
                         test_index: np.ndarray,
                         strategy: Any) -> Optional[pd.DataFrame]:
        """Run backtest on a single train/test split."""
        try:
            # Get train/test data
            train_data = self.data.iloc[train_index].copy()
            test_data = self.data.iloc[test_index].copy()
            
            # Train strategy
            strategy.train(train_data)
            
            # Generate signals
            signals = strategy.generate_signals(test_data)
            
            # Get market conditions
            market_conditions = self.classify_market_conditions(test_data)
            
            # Calculate returns
            returns = self.calculate_returns(test_data, signals)
            
            if returns is not None:
                returns['Market_Condition'] = market_conditions
                return returns
                
            return None
            
        except Exception as e:
            logger.error(f"Error in single split: {str(e)}")
            return None
    
    def _run_single_backtest(self, strategy: Any) -> pd.DataFrame:
        """
        Run a single backtest without walk-forward optimization.

        Args:
            strategy: Trading strategy object

        Returns:
            DataFrame containing backtest results
        """
        try:
            # Generate signals
            signals = strategy.generate_signals(self.data)

            # Classify market conditions
            market_conditions = self._classify_market_conditions(self.data)

            # Calculate returns
            returns = self._calculate_returns(self.data, signals)
            returns['Market_Condition'] = market_conditions

            return returns

        except Exception as e:
            logger.error(f"Error in single backtest: {e}")
            raise

    def classify_market_conditions(self, data: pd.DataFrame) -> pd.Series:
        """
        Classify market conditions based on technical indicators.
        
        Args:
            data: DataFrame containing market data
            
        Returns:
            Series containing market condition labels
        """
        try:
            # Create a copy of the data to avoid SettingWithCopyWarning
            df = data.copy()
            
            # Calculate returns first
            df['Returns'] = df['close'].pct_change()
            
            # Calculate necessary indicators using proper DataFrame assignment
            df.loc[:, 'SMA50'] = df['close'].rolling(window=50).mean()
            df.loc[:, 'SMA200'] = df['close'].rolling(window=200).mean()
            df.loc[:, 'Volatility'] = df['Returns'].rolling(window=20).std() * np.sqrt(252)
            df.loc[:, 'Volume_MA'] = df['volume'].rolling(window=20).mean()
            df.loc[:, 'Momentum'] = df['close'].pct_change(periods=20)

            conditions = []
            for i in range(len(df)):
                if i < 200:  # Not enough data for classification
                    conditions.append('Undefined')
                    continue

                current = df.iloc[i]
                
                # Trend
                if current['SMA50'] > current['SMA200']:
                    trend = 'Uptrend'
                elif current['SMA50'] < current['SMA200']:
                    trend = 'Downtrend'
                else:
                    trend = 'Sideways'

                # Volatility
                vol_quantiles = df['Volatility'].quantile([0.25, 0.75])
                if current['Volatility'] > vol_quantiles[0.75]:
                    volatility = 'High'
                elif current['Volatility'] < vol_quantiles[0.25]:
                    volatility = 'Low'
                else:
                    volatility = 'Medium'

                # Volume
                if current['volume'] > current['Volume_MA'] * 1.5:
                    volume = 'High'
                elif current['volume'] < current['Volume_MA'] * 0.5:
                    volume = 'Low'
                else:
                    volume = 'Normal'

                # Momentum
                if current['Momentum'] > 0.05:
                    momentum = 'Strong'
                elif current['Momentum'] < -0.05:
                    momentum = 'Weak'
                else:
                    momentum = 'Neutral'

                condition = f"{trend}-{volatility}_Volatility-{volume}_Volume-{momentum}_Momentum"
                conditions.append(condition)

            return pd.Series(conditions, index=df.index)

        except Exception as e:
            logger.error(f"Error classifying market conditions: {e}")
            raise    
    def calculate_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Calculate returns based on signals.
        
        Args:
            data: Market data DataFrame
            signals: Trading signals
            
        Returns:
            DataFrame containing calculated returns and metrics
        """
        try:
            # Create a new DataFrame for returns
            returns = pd.DataFrame(index=data.index)
            
            # Add required columns with proper DataFrame assignment
            returns.loc[:, 'Price'] = data['close']
            returns.loc[:, 'Signal'] = signals
            returns.loc[:, 'Returns'] = data['close'].pct_change().fillna(0)
            
            # Calculate strategy returns
            returns.loc[:, 'Strategy_Returns'] = returns['Signal'].shift(1) * returns['Returns']
            returns.loc[returns.index[0], 'Strategy_Returns'] = 0  # Set first row to 0
            
            # Calculate cumulative returns
            returns.loc[:, 'Cumulative_Returns'] = (1 + returns['Strategy_Returns']).cumprod()
            
            # Calculate drawdown
            rolling_max = returns['Cumulative_Returns'].expanding().max()
            returns.loc[:, 'Drawdown'] = (returns['Cumulative_Returns'] - rolling_max) / rolling_max
            
            return returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            raise

    def run_monte_carlo(self, 
                       strategy: Any, 
                       num_simulations: int = 1000,
                       simulation_length: int = 252) -> pd.DataFrame:
        """
        Run Monte Carlo simulation.
        
        Args:
            strategy: Trading strategy to test
            num_simulations: Number of simulations to run
            simulation_length: Length of each simulation in days
            
        Returns:
            DataFrame containing simulation results
        """
        try:
            # Create empty list to store simulation results
            simulation_results = []
            
            for i in range(num_simulations):
                try:
                    # Generate synthetic data
                    synthetic_data = self._generate_synthetic_data(simulation_length)
                    
                    # Generate signals using strategy
                    signals = strategy.generate_signals(synthetic_data)
                    
                    # Calculate returns
                    returns = self.calculate_returns(synthetic_data, signals)
                    
                    # Calculate metrics for this simulation
                    metrics = self._calculate_simulation_metrics(returns)
                    metrics['simulation_id'] = i
                    
                    # Store simulation metrics
                    simulation_results.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"Error in simulation {i}: {e}")
                    continue
                    
            # Convert results to DataFrame
            return pd.DataFrame(simulation_results)
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            return pd.DataFrame()
        
    def _generate_synthetic_data(self, days: int) -> pd.DataFrame:
        """Generate synthetic market data for simulation."""
        try:
            returns = self.data['close'].pct_change().dropna()
            
            # Calculate distribution parameters
            mu = returns.mean()
            sigma = returns.std()
            skew = returns.skew()
            kurt = returns.kurtosis()
            
            # Generate random returns with similar properties
            random_returns = np.random.normal(mu, sigma, days)
            
            # Generate synthetic prices
            last_price = self.data['close'].iloc[-1]
            synthetic_prices = last_price * np.exp(np.cumsum(random_returns))
            
            # Generate synthetic OHLCV data
            dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=days)
            synthetic_data = pd.DataFrame(index=dates)
            
            synthetic_data['close'] = synthetic_prices
            synthetic_data['open'] = synthetic_prices * np.random.uniform(0.99, 1.01, days)
            synthetic_data['high'] = synthetic_data[['open', 'close']].max(axis=1) * np.random.uniform(1.001, 1.02, days)
            synthetic_data['low'] = synthetic_data[['open', 'close']].min(axis=1) * np.random.uniform(0.98, 0.999, days)
            synthetic_data['volume'] = np.random.lognormal(np.log(self.data['volume'].mean()), 
                                                         self.data['volume'].std(), 
                                                         days)
            
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise
    
    def calculate_performance_metrics(self, data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Calculate comprehensive performance metrics.

        Args:
            data: Optional DataFrame to calculate metrics for. If None, uses self.results

        Returns:
            Dictionary containing various performance metrics
        """
        try:
            if self.results is None and data is None:
                raise ValueError("Backtest hasn't been run yet. Call run_backtest() first.")
                
            # Use provided data or fall back to self.results
            results_df = data if data is not None else self.results
            
            if results_df.empty:
                return {
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'sharpe_ratio': 0.0,
                    'volatility': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'profit_factor': 0.0
                }

            returns = results_df['Strategy_Returns'].dropna()
            
            # Calculate returns
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)
            risk_free_rate: float = 0.02 
            # Calculate Sharpe ratio
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
            
            # Calculate maximum drawdown
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdowns.min()) if len(drawdowns) > 0 else 0
            
            # Calculate win rate
            winning_trades = (returns > 0).sum()
            total_trades = len(returns[returns != 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profits = returns[returns > 0].sum()
            gross_losses = abs(returns[returns < 0].sum())
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            risk_free_rate: float = 0.02
            # Calculate sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - risk_free_rate) / downside_std if downside_std != 0 else 0
            
            # Calculate Calmar ratio
            calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else float('inf')
            
            return {
                'total_return': total_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'avg_return': returns.mean(),
                'avg_win': returns[returns > 0].mean() if len(returns[returns > 0]) > 0 else 0,
                'avg_loss': returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else 0,
                'best_trade': returns.max(),
                'worst_trade': returns.min(),
                'recovery_factor': abs(total_return / max_drawdown) if max_drawdown != 0 else float('inf'),
                'risk_adjusted_return': annual_return / volatility if volatility != 0 else 0
            }

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise
    
    def generate_report(self, 
                       output_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Generate comprehensive backtest report.

        Args:
            output_path: Optional path to save report

        Returns:
            Dictionary containing report data

        Raises:
            ValueError: If results are not available
        """
        try:
            if self.results is None:
                raise ValueError("Backtest hasn't been run yet")

            # Generate report components
            report = {
                'performance_metrics': self.performance_metrics.to_dict(),
                'market_condition_analysis': self._analyze_market_conditions(),
                'trade_analysis': self._analyze_trades(),
                'risk_analysis': self._analyze_risk_metrics()
            }

            if self.monte_carlo_results is not None:
                report['monte_carlo_analysis'] = self._analyze_monte_carlo()

            # Generate and save visualizations
            if output_path:
                self._generate_visualizations(output_path)

            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            raise

    def _analyze_market_conditions(self) -> Dict[str, Any]:
        """Analyze performance across different market conditions."""
        try:
            analysis = {}
            for condition in self.results['Market_Condition'].unique():
                condition_data = self.results[
                    self.results['Market_Condition'] == condition
                ]
                analysis[condition] = {
                    'count': len(condition_data),
                    'win_rate': (
                        condition_data['Strategy_Returns'] > 0
                    ).mean(),
                    'avg_return': condition_data['Strategy_Returns'].mean(),
                    'sharpe': (
                        condition_data['Strategy_Returns'].mean() / 
                        condition_data['Strategy_Returns'].std() * 
                        np.sqrt(252)
                    )
                }
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing market conditions: {e}")
            raise

    def _analyze_trades(self) -> Dict[str, Any]:
        """Analyze individual trades."""
        try:
            signals = self.results['Signal']
            trades = []
            current_trade = None

            for date, signal in signals.items():
                if current_trade is None and signal != 0:
                    current_trade = {
                        'entry_date': date,
                        'entry_price': self.results.loc[date, 'Price'],
                        'direction': 'long' if signal > 0 else 'short'
                    }
                elif current_trade is not None and signal == 0:
                    current_trade['exit_date'] = date
                    current_trade['exit_price'] = self.results.loc[
                        date, 'Price'
                    ]
                    current_trade['return'] = (
                        (current_trade['exit_price'] / 
                         current_trade['entry_price'] - 1) *
                        (1 if current_trade['direction'] == 'long' else -1)
                    )
                    trades.append(current_trade)
                    current_trade = None

            return {
                'total_trades': len(trades),
                'avg_trade_duration': np.mean([
                    (t['exit_date'] - t['entry_date']).days 
                    for t in trades
                ]),
                'avg_trade_return': np.mean([t['return'] for t in trades]),
                'best_trade': max(trades, key=lambda x: x['return']),
                'worst_trade': min(trades, key=lambda x: x['return'])
            }

        except Exception as e:
            logger.error(f"Error analyzing trades: {e}")
            raise
    def _analyze_risk_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics.

        Returns:
            Dictionary containing calculated risk metrics

        Raises:
            ValueError: If results data is invalid
        """
        try:
            returns = self.results['Strategy_Returns'].dropna()
            
            # Calculate Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            # Calculate Conditional Value at Risk (CVaR/Expected Shortfall)
            cvar_95 = returns[returns <= var_95].mean()
            cvar_99 = returns[returns <= var_99].mean()
            
            # Calculate Sortino Ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (
                (returns.mean() * 252 - self.config.risk_free_rate) / 
                downside_std if downside_std != 0 else np.inf
            )
            
            # Calculate Calmar Ratio
            max_dd = self.results['Drawdown'].max()
            calmar_ratio = (
                returns.mean() * 252 / max_dd if max_dd != 0 else np.inf
            )
            
            # Calculate Omega Ratio
            threshold = 0
            omega_ratio = (
                returns[returns > threshold].sum() / 
                abs(returns[returns < threshold].sum())
                if len(returns[returns < threshold]) > 0 else np.inf
            )

            return {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'omega_ratio': omega_ratio,
                'max_drawdown': max_dd,
                'max_drawdown_duration': self._calculate_max_drawdown_duration(),
                'skewness': returns.skew(),
                'kurtosis': returns.kurtosis(),
                'tail_ratio': abs(np.percentile(returns, 95)) / abs(np.percentile(returns, 5))
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            raise

    def _calculate_max_drawdown_duration(self) -> int:
        """
        Calculate the maximum drawdown duration in days.

        Returns:
            Number of days of longest drawdown period
        """
        try:
            drawdown = self.results['Drawdown']
            max_duration = 0
            current_duration = 0
            
            for dd in drawdown:
                if dd == 0:
                    current_duration = 0
                else:
                    current_duration += 1
                    max_duration = max(max_duration, current_duration)
                    
            return max_duration

        except Exception as e:
            logger.error(f"Error calculating max drawdown duration: {e}")
            raise

    def _analyze_monte_carlo(self) -> Dict[str, Any]:
        """
        Analyze Monte Carlo simulation results.

        Returns:
            Dictionary containing Monte Carlo analysis results

        Raises:
            ValueError: If Monte Carlo results are not available
        """
        try:
            if self.monte_carlo_results is None:
                raise ValueError("Monte Carlo simulation hasn't been run")

            # Calculate confidence intervals
            confidence_intervals = {}
            metrics = ['sharpe_ratio', 'max_drawdown', 'total_return']
            
            for metric in metrics:
                values = self.monte_carlo_results[metric].sort_values()
                confidence_intervals[metric] = {
                    '95%': (values.quantile(0.025), values.quantile(0.975)),
                    '99%': (values.quantile(0.005), values.quantile(0.995))
                }

            # Calculate probability of various scenarios
            total_sims = len(self.monte_carlo_results)
            
            return {
                'confidence_intervals': confidence_intervals,
                'probability_positive_sharpe': (
                    (self.monte_carlo_results['sharpe_ratio'] > 0).sum() / 
                    total_sims
                ),
                'probability_target_return': (
                    (self.monte_carlo_results['total_return'] > 
                     config.TARGET_ANNUAL_RETURN).sum() / total_sims
                ),
                'probability_max_drawdown': (
                    (self.monte_carlo_results['max_drawdown'] < 
                     config.MAX_ACCEPTABLE_DRAWDOWN).sum() / total_sims
                ),
                'worst_case_metrics': {
                    metric: self.monte_carlo_results[metric].min()
                    for metric in metrics
                },
                'best_case_metrics': {
                    metric: self.monte_carlo_results[metric].max()
                    for metric in metrics
                }
            }

        except Exception as e:
            logger.error(f"Error analyzing Monte Carlo results: {e}")
            raise

    def _generate_visualizations(self, output_path: Path) -> None:
        """
        Generate and save visualization plots.

        Args:
            output_path: Path to save visualization files

        Raises:
            ValueError: If results data is invalid
        """
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Generate equity curve
            self._plot_equity_curve(output_path / 'equity_curve.html')
            
            # Generate drawdown plot
            self._plot_drawdown(output_path / 'drawdown.html')
            
            # Generate market condition performance
            self._plot_market_condition_performance(
                output_path / 'market_conditions.html'
            )
            
            # Generate Monte Carlo distribution
            if self.monte_carlo_results is not None:
                self._plot_monte_carlo_distribution(
                    output_path / 'monte_carlo.html'
                )

            logger.info(f"Visualizations saved to {output_path}")

        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise

    def _plot_equity_curve(self, filepath: Path) -> None:
        """Generate interactive equity curve plot."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.results.index,
                y=self.results['Cumulative_Returns'],
                mode='lines',
                name='Strategy',
                line=dict(color='blue')
            ))
            
            # Add buy/sell markers
            signals = self.results['Signal']
            buys = signals[signals > 0].index
            sells = signals[signals < 0].index
            
            fig.add_trace(go.Scatter(
                x=buys,
                y=self.results.loc[buys, 'Cumulative_Returns'],
                mode='markers',
                name='Buy',
                marker=dict(color='green', size=8)
            ))
            
            fig.add_trace(go.Scatter(
                x=sells,
                y=self.results.loc[sells, 'Cumulative_Returns'],
                mode='markers',
                name='Sell',
                marker=dict(color='red', size=8)
            ))
            
            fig.update_layout(
                title='Equity Curve',
                xaxis_title='Date',
                yaxis_title='Cumulative Return',
                template='plotly_white'
            )
            
            fig.write_html(str(filepath))

        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
            raise

    def _plot_drawdown(self, filepath: Path) -> None:
        """Generate interactive drawdown plot."""
        try:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=self.results.index,
                y=-self.results['Drawdown'] * 100,  # Convert to percentage
                fill='tozeroy',
                mode='lines',
                name='Drawdown',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title='Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white'
            )
            
            fig.write_html(str(filepath))

        except Exception as e:
            logger.error(f"Error plotting drawdown: {e}")
            raise

    def _plot_market_condition_performance(self, filepath: Path) -> None:
        """Generate market condition performance comparison plot."""
        try:
            market_condition_analysis = self._analyze_market_conditions()
            
            conditions = list(market_condition_analysis.keys())
            win_rates = [
                market_condition_analysis[c]['win_rate'] * 100 
                for c in conditions
            ]
            avg_returns = [
                market_condition_analysis[c]['avg_return'] * 100 
                for c in conditions
            ]
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Win Rate by Market Condition',
                              'Average Return by Market Condition')
            )
            
            fig.add_trace(
                go.Bar(x=conditions, y=win_rates, name='Win Rate'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=conditions, y=avg_returns, name='Avg Return'),
                row=1, col=2
            )
            
            fig.update_layout(
                title='Performance by Market Condition',
                template='plotly_white',
                showlegend=False
            )
            
            fig.write_html(str(filepath))

        except Exception as e:
            logger.error(f"Error plotting market condition performance: {e}")
            raise

    def _plot_monte_carlo_distribution(self, filepath: Path) -> None:
        """Generate Monte Carlo simulation distribution plot."""
        try:
            if self.monte_carlo_results is None:
                raise ValueError("No Monte Carlo results available")

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Return Distribution',
                              'Sharpe Ratio Distribution',
                              'Max Drawdown Distribution',
                              'Monte Carlo Paths')
            )
            
            # Return distribution
            fig.add_trace(
                go.Histogram(
                    x=self.monte_carlo_results['total_return'],
                    name='Return'
                ),
                row=1, col=1
            )
            
            # Sharpe ratio distribution
            fig.add_trace(
                go.Histogram(
                    x=self.monte_carlo_results['sharpe_ratio'],
                    name='Sharpe'
                ),
                row=1, col=2
            )
            
            # Max drawdown distribution
            fig.add_trace(
                go.Histogram(
                    x=self.monte_carlo_results['max_drawdown'],
                    name='Drawdown'
                ),
                row=2, col=1
            )
            
            # Sample of Monte Carlo paths
            sample_paths = self.monte_carlo_results.sample(
                min(100, len(self.monte_carlo_results))
            )
            
            for _, path in sample_paths.iterrows():
                fig.add_trace(
                    go.Scatter(
                        y=path['equity_curve'],
                        mode='lines',
                        line=dict(width=1, color='rgba(0,0,255,0.1)'),
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Monte Carlo Simulation Results',
                template='plotly_white',
                showlegend=False
            )
            
            fig.write_html(str(filepath))

        except Exception as e:
            logger.error(f"Error plotting Monte Carlo distribution: {e}")
            raise
