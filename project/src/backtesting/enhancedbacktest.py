# Standard Library Imports
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

# Third-Party Imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import TimeSeriesSplit
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
class EnhancedBacktest:
    """
    Enhanced backtesting system with walk-forward optimization
    and Monte Carlo simulation capabilities.
    """

    def __init__(self, 
             data: pd.DataFrame, 
             initial_capital: float = 100000,
             config: Optional[BacktestConfig] = None):
        """
        Initialize the backtesting system.

        Args:
            data: DataFrame containing OHLCV price data
            initial_capital: Initial capital amount
            config: Backtesting configuration
        """
        try:
            # Define required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Validate the DataFrame
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input data must be a pandas DataFrame")
                
            # Convert column names to lowercase
            data.columns = data.columns.str.lower()
            
            # Check required columns
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
                
            # Check minimum data points
            if len(data) < 100:  # Minimum required for backtesting
                raise ValueError(f"Insufficient data points: {len(data)} < 100")
                
            # Store validated data
            self.data = data.copy()
            self.initial_capital = initial_capital
            self.results = None
            
            # Store configuration
            self.config = config or BacktestConfig()
            
            logger.info(f"Backtest initialized with {len(data)} data points")
            
        except Exception as e:
            logger.error(f"Error initializing backtest system: {e}")
            raise ValueError(f"Invalid input data: {str(e)}")


    def run_backtest(self, strategy: Any, n_splits: Optional[int] = None) -> pd.DataFrame:
        """
        Run the backtest with walk-forward optimization.
        
        Args:
            strategy: Trading strategy object with generate_signals method
            n_splits: Optional number of splits (will be calculated automatically if None)
            
        Returns:
            DataFrame containing backtest results
        """
        try:
            # Calculate optimal number of splits based on data size
            data_length = len(self.data)
            min_samples_per_split = 100  # Minimum samples needed for reliable training
            
            if n_splits is None:
                # Dynamically calculate number of splits
                n_splits = max(2, min(5, data_length // (2 * min_samples_per_split)))
            
            # Ensure we have enough data for the requested splits
            min_required_samples = n_splits * min_samples_per_split * 2  # *2 for train/test
            if data_length < min_required_samples:
                logger.warning(
                    f"Insufficient data for {n_splits} splits. "
                    f"Reducing to {data_length // (2 * min_samples_per_split)} splits"
                )
                n_splits = max(2, data_length // (2 * min_samples_per_split))
            
            # Calculate test size
            test_size = data_length // (n_splits + 1)  # Ensure test size is proportional
            
            # Initialize TimeSeriesSplit
            tscv = TimeSeriesSplit(
                n_splits=n_splits,
                test_size=test_size,
                gap=0
            )
            
            # Run walk-forward optimization
            all_results = []
            for train_index, test_index in tscv.split(self.data):
                try:
                    split_results = self._run_single_split(train_index, test_index, strategy)
                    all_results.append(split_results)
                except Exception as e:
                    logger.error(f"Error in split: {e}")
                    continue
            
            if not all_results:
                raise ValueError("No valid results from any split")
            
            self.results = pd.concat(all_results)
            return self.results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
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
                     strategy: Any) -> pd.DataFrame:
        """
        Run backtest on a single train/test split.
        
        Args:
            train_index: Indices for training data
            test_index: Indices for testing data
            strategy: Trading strategy object
            
        Returns:
            DataFrame containing results for this split
        """
        try:
            # Get train/test data
            train_data = self.data.iloc[train_index]
            test_data = self.data.iloc[test_index]
            
            # Ensure minimum data requirements
            if len(train_data) < 50 or len(test_data) < 20:  # Example minimum requirements
                raise ValueError(
                    f"Insufficient data in split: train={len(train_data)}, test={len(test_data)}"
                )
            
            # Train strategy
            strategy.train(train_data)
            
            # Generate signals
            signals = strategy.generate_signals(test_data)
            
            # Get market conditions
            market_conditions = self.classify_market_conditions(test_data)
            
            # Calculate returns
            returns = self.calculate_returns(test_data, signals)
            returns['Market_Condition'] = market_conditions
            
            return returns
            
        except Exception as e:
            logger.error(f"Error in single split: {e}")
            raise

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

    def calculate_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Calculate returns and performance metrics for trades.
        
        Args:
            data: Price data
            signals: Trading signals
            
        Returns:
            DataFrame containing calculated returns
        """
        try:
            returns = pd.DataFrame(index=data.index)
            returns['Price'] = data['close']
            returns['Signal'] = signals
            returns['Returns'] = np.log(data['close'] / data['close'].shift(1))
            
            # Calculate transaction costs
            transaction_costs = (
                abs(returns['Signal'].diff()) * 
                (self.config.commission_rate + self.config.slippage)
            )
            
            # Calculate strategy returns
            returns['Strategy_Returns'] = (
                returns['Signal'].shift(1) * returns['Returns'] - 
                transaction_costs
            )
            
            # Calculate cumulative returns and drawdown
            returns['Cumulative_Returns'] = (
                (1 + returns['Strategy_Returns']).cumprod()
            )
            returns['Drawdown'] = (
                (returns['Cumulative_Returns'].cummax() - 
                returns['Cumulative_Returns']) / 
                returns['Cumulative_Returns'].cummax()
            )

            return returns

        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            raise

    def _run_monte_carlo(self,
                        strategy: Any,
                        n_simulations: int) -> pd.DataFrame:
        """
        Run Monte Carlo simulations.

        Args:
            strategy: Trading strategy object
            n_simulations: Number of simulations to run

        Returns:
            DataFrame containing simulation results
        """
        try:
            simulation_results = []
            
            for i in range(n_simulations):
                # Generate synthetic price data
                synthetic_data = self._generate_synthetic_data()
                
                # Run backtest on synthetic data
                signals = strategy.generate_signals(synthetic_data)
                returns = self._calculate_returns(synthetic_data, signals)
                
                # Calculate performance metrics
                metrics = self._calculate_simulation_metrics(returns)
                metrics['simulation_id'] = i
                simulation_results.append(metrics)
                
                if i % 100 == 0:
                    logger.info(f"Completed {i} Monte Carlo simulations")

            return pd.DataFrame(simulation_results)

        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {e}")
            raise

    def _generate_synthetic_data(self) -> pd.DataFrame:
        """
        Generate synthetic price data for Monte Carlo simulation.

        Returns:
            DataFrame containing synthetic price data
        """
        try:
            # Get historical parameters
            returns = np.log(
                self.data['close'] / self.data['close'].shift(1)
            ).dropna()
            
            mu = returns.mean()
            sigma = returns.std()
            
            # Generate random returns
            n_days = len(self.data)
            random_returns = np.random.normal(mu, sigma, n_days)
            
            # Generate synthetic prices
            synthetic_prices = self.data['close'].iloc[0] * np.exp(
                np.cumsum(random_returns)
            )
            
            # Create synthetic DataFrame
            synthetic_data = pd.DataFrame(index=self.data.index)
            synthetic_data['close'] = synthetic_prices
            synthetic_data['open'] = synthetic_prices * np.random.uniform(
                0.99, 1.01, n_days
            )
            synthetic_data['high'] = synthetic_prices * np.random.uniform(
                1.001, 1.02, n_days
            )
            synthetic_data['low'] = synthetic_prices * np.random.uniform(
                0.98, 0.999, n_days
            )
            synthetic_data['volume'] = np.random.randint(
                100000, 1000000, n_days
            )

            return synthetic_data

        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            raise

    def _calculate_performance_metrics(self) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.

        Returns:
            PerformanceMetrics object containing calculated metrics

        Raises:
            ValueError: If results are not available
        """
        try:
            if self.results is None:
                raise ValueError("Backtest hasn't been run yet")

            returns = self.results['Strategy_Returns'].dropna()
            
            # Calculate key metrics
            total_return = (1 + returns).prod() - 1
            annual_return = (1 + total_return) ** (
                252 / len(returns)
            ) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe = (
                (annual_return - self.config.risk_free_rate) / 
                volatility
            )
            
            # Calculate other metrics
            max_drawdown = self.results['Drawdown'].max()
            win_rate = (returns > 0).mean()
            profit_factor = abs(
                returns[returns > 0].sum() / 
                returns[returns < 0].sum()
            )

            return PerformanceMetrics(
                sharpe_ratio=sharpe,
                win_rate=win_rate,
                profit_factor=profit_factor,
                max_drawdown=max_drawdown,
                volatility=volatility
            )

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
