# Standard Library Imports
import json
import logging
import multiprocessing
import os
import sqlite3
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

# Third-Party Imports
import arch
import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yfinance as yf
from arch import arch_model
from joblib import Parallel, delayed
from scipy.stats import kurtosis, skew, t
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler, StandardScaler
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, MACD, SMAIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.volatility import AverageTrueRange, BollingerBands
from tqdm import tqdm

# Local Imports
import project.config as config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DatabaseHandler:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def load_market_data(self, ticker: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM '{ticker}'"
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading market data: {e}")
            raise

class EnhancedBacktest:
    def __init__(self, data: pd.DataFrame, initial_capital: float = 100000):
        self.data = data
        self.initial_capital = initial_capital
        self.results = None

    def run_backtest(self, strategy, n_splits: int = 5, n_jobs: int = -1) -> pd.DataFrame:
        tscv = TimeSeriesSplit(n_splits=n_splits)
        all_results = Parallel(n_jobs=n_jobs)(
            delayed(self._run_single_split)(train, test, strategy)
            for train, test in tscv.split(self.data)
        )
        self.results = pd.concat(all_results)
        return self.results

    def _run_single_split(self, train_index, test_index, strategy):
        train_data = self.data.iloc[train_index]
        test_data = self.data.iloc[test_index]
        strategy.train(train_data)
        signals = strategy.generate_signals(test_data)
        market_conditions = self.classify_market_conditions(test_data)
        returns = self.calculate_returns(test_data, signals)
        returns['Market_Condition'] = market_conditions
        return returns

    def calculate_returns(self, data: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        returns = pd.DataFrame(index=data.index)
        returns['Price'] = data['Close']
        returns['Signal'] = signals
        returns['Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        returns['Strategy_Returns'] = returns['Signal'].shift(1) * returns['Returns']
        returns['Cumulative_Returns'] = (1 + returns['Strategy_Returns']).cumprod()
        returns['Drawdown'] = (returns['Cumulative_Returns'].cummax() - returns['Cumulative_Returns']) / returns['Cumulative_Returns'].cummax()
        return returns

    def classify_market_conditions(self, data: pd.DataFrame) -> pd.Series:
        # Calculate necessary indicators
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['SMA200'] = data['Close'].rolling(window=200).mean()
        data['Volatility'] = data['Returns'].rolling(window=20).std() * np.sqrt(252)
        data['Volume_MA'] = data['Volume'].rolling(window=20).mean()
        data['Momentum'] = data['Close'].pct_change(periods=20)

        conditions = []
        for i in range(len(data)):
            if i < 200:  # Not enough data for classification
                conditions.append('Undefined')
                continue

            current = data.iloc[i]
            
            # Trend
            if current['SMA50'] > current['SMA200']:
                trend = 'Uptrend'
            elif current['SMA50'] < current['SMA200']:
                trend = 'Downtrend'
            else:
                trend = 'Sideways'

            # Volatility
            if current['Volatility'] > data['Volatility'].quantile(0.75):
                volatility = 'High'
            elif current['Volatility'] < data['Volatility'].quantile(0.25):
                volatility = 'Low'
            else:
                volatility = 'Medium'

            # Volume
            if current['Volume'] > current['Volume_MA'] * 1.5:
                volume = 'High'
            elif current['Volume'] < current['Volume_MA'] * 0.5:
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

        return pd.Series(conditions, index=data.index)

    def calculate_performance_metrics(self) -> dict:
        if self.results is None:
            raise ValueError("Backtest hasn't been run yet. Call run_backtest() first.")

        metrics = {}
        returns = self.results['Strategy_Returns']
        cumulative_returns = self.results['Cumulative_Returns']

        metrics['Total Return'] = cumulative_returns.iloc[-1] - 1
        metrics['Annualized Return'] = (1 + metrics['Total Return']) ** (252 / len(returns)) - 1
        metrics['Sharpe Ratio'] = np.sqrt(252) * returns.mean() / returns.std()
        
        downside_returns = returns[returns < 0]
        metrics['Sortino Ratio'] = np.sqrt(252) * returns.mean() / downside_returns.std()
        
        metrics['Max Drawdown'] = self.results['Drawdown'].max()
        metrics['Win Rate'] = (returns > 0).mean()
        metrics['Volatility'] = returns.std() * np.sqrt(252)

        # Calculate metrics per market condition
        for condition in self.results['Market_Condition'].unique():
            condition_returns = returns[self.results['Market_Condition'] == condition]
            metrics[f'{condition} Return'] = condition_returns.mean()
            metrics[f'{condition} Sharpe'] = np.sqrt(252) * condition_returns.mean() / condition_returns.std()

        return metrics

    def calculate_confidence_score(self, current_condition: str) -> float:
        if self.results is None:
            raise ValueError("Backtest hasn't been run yet. Call run_backtest() first.")

        # Calculate overall performance
        overall_return = self.results['Strategy_Returns'].mean()
        overall_sharpe = np.sqrt(252) * overall_return / self.results['Strategy_Returns'].std()

        # Calculate performance for the current market condition
        condition_returns = self.results[self.results['Market_Condition'] == current_condition]['Strategy_Returns']
        condition_return = condition_returns.mean()
        condition_sharpe = np.sqrt(252) * condition_return / condition_returns.std()

        # Calculate confidence score
        # This is a simple example; you might want to adjust the weights or add more factors
        confidence_score = 0.5 * (condition_return / overall_return) + 0.5 * (condition_sharpe / overall_sharpe)

        return min(max(confidence_score, 0), 1)  # Ensure the score is between 0 and 1

    def generate_synthetic_data(self, days: int = 252) -> pd.DataFrame:
        returns = self.data['Close'].pct_change().dropna()
        
        # Fit GARCH model
        model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
        results = model.fit(disp='off')

        # Simulate returns
        sim_returns = results.forecast(horizon=days, method='simulation').simulations.values[-1, :]
        
        # Generate prices from returns
        last_price = self.data['Close'].iloc[-1]
        sim_prices = last_price * np.exp(np.cumsum(sim_returns))
        
        dates = pd.date_range(start=self.data.index[-1] + pd.Timedelta(days=1), periods=days)
        
        synthetic_data = pd.DataFrame({
            'Close': sim_prices,
            'Open': sim_prices * np.random.uniform(0.99, 1.01, days),
            'High': sim_prices * np.random.uniform(1.001, 1.02, days),
            'Low': sim_prices * np.random.uniform(0.98, 0.999, days),
            'Volume': np.random.randint(100000, 1000000, days)
        }, index=dates)

        return synthetic_data

    def run_monte_carlo(self, strategy, num_simulations: int = 1000, simulation_length: int = 252) -> pd.DataFrame:
        monte_carlo_results = []

        for _ in tqdm(range(num_simulations), desc="Running Monte Carlo Simulations"):
            synthetic_data = self.generate_synthetic_data(simulation_length)
            signals = strategy.generate_signals(synthetic_data)
            returns = self.calculate_returns(synthetic_data, signals)
            market_conditions = self.classify_market_conditions(synthetic_data)
            returns['Market_Condition'] = market_conditions
            
            metrics = self.calculate_performance_metrics_for_simulation(returns)
            monte_carlo_results.append(metrics)

        return pd.DataFrame(monte_carlo_results)

    def calculate_performance_metrics_for_simulation(self, returns: pd.DataFrame) -> dict:
        metrics = {}
        strategy_returns = returns['Strategy_Returns']

        metrics['Total Return'] = (1 + strategy_returns).prod() - 1
        metrics['Annualized Return'] = (1 + metrics['Total Return']) ** (252 / len(strategy_returns)) - 1
        metrics['Sharpe Ratio'] = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        metrics['Max Drawdown'] = returns['Drawdown'].max()
        metrics['Win Rate'] = (strategy_returns > 0).mean()
        
        for condition in returns['Market_Condition'].unique():
            condition_returns = strategy_returns[returns['Market_Condition'] == condition]
            metrics[f'{condition} Return'] = condition_returns.mean()
            metrics[f'{condition} Sharpe'] = np.sqrt(252) * condition_returns.mean() / condition_returns.std()

        return metrics

@dataclass
class PerformanceMetrics:
    """Data class for storing model performance metrics"""
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    volatility: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format"""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility
        }

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PerformanceMetrics':
        """Create PerformanceMetrics instance from dictionary"""
        return cls(
            sharpe_ratio=data['sharpe_ratio'],
            win_rate=data['win_rate'],
            profit_factor=data['profit_factor'],
            max_drawdown=data['max_drawdown'],
            volatility=data['volatility']
        )

class IndicatorCalculator:
    @staticmethod
    def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
        try:
            df = data.copy()
            df = add_all_ta_features(df, open="open", high="high", low="low", close="close", volume="volume")
            
            # Additional custom indicators
            df['SMA_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['SMA_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
            
            macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            
            df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            
            df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
            
            df['ADX'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
            
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']
            
            df['VWAP'] = VolumeWeightedAveragePrice(high=df['high'], low=df['low'], close=df['close'], volume=df['volume']).volume_weighted_average_price()
            
            # Vectorized operations for performance
            df['RSI_Z'] = (df['RSI'] - df['RSI'].rolling(window=50).mean()) / df['RSI'].rolling(window=50).std()
            df['MACD_Z'] = (df['MACD_diff'] - df['MACD_diff'].rolling(window=50).mean()) / df['MACD_diff'].rolling(window=50).std()
            df['Momentum'] = df['close'].pct_change(periods=20)
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
            
            return df
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise

class MarketConditionAnalyzer:
    @staticmethod
    def determine_market_condition(data: pd.DataFrame) -> str:
        try:
            last_row = data.iloc[-1]
            trend = "Uptrend" if last_row['SMA_50'] > last_row['SMA_200'] else "Downtrend"
            volatility = "High" if last_row['ATR'] > data['ATR'].mean() else "Low"
            volume = "High" if last_row['Volume_Ratio'] > 1.5 else "Low"
            momentum = "Positive" if last_row['Momentum'] > 0 else "Negative"
            return f"{trend}-{volatility} Volatility-{volume} Volume-{momentum} Momentum"
        except Exception as e:
            logger.error(f"Error determining market condition: {e}")
            raise

class ParameterAdjuster:
    def __init__(self):
        self.asset_characteristics = {}

    def adjust_parameters(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        try:
            volatility = data['close'].pct_change().rolling(window=20).std().iloc[-1]
            
            if ticker not in self.asset_characteristics:
                self.asset_characteristics[ticker] = self.calculate_asset_characteristics(data)
            
            asset_volatility = self.asset_characteristics[ticker]['volatility']
            market_volatility = self.calculate_market_volatility()
            
            # Adjust indicator parameters
            rsi_window = int(config.RSI_WINDOW * (1 + volatility) * (asset_volatility / market_volatility))
            macd_fast = int(config.MACD_FAST * (1 - volatility) * (market_volatility / asset_volatility))
            macd_slow = int(config.MACD_SLOW * (1 + volatility) * (asset_volatility / market_volatility))
            adx_window = int(config.ADX_WINDOW * (1 + volatility) * (asset_volatility / market_volatility))
            atr_window = int(config.ATR_WINDOW * (1 + volatility) * (asset_volatility / market_volatility))
            
            # Recalculate indicators with new parameters
            data['RSI'] = RSIIndicator(close=data['close'], window=rsi_window).rsi()
            macd = MACD(close=data['close'], window_slow=macd_slow, window_fast=macd_fast, window_sign=9)
            data['MACD'] = macd.macd()
            data['MACD_signal'] = macd.macd_signal()
            data['MACD_diff'] = macd.macd_diff()
            data['ADX'] = ADXIndicator(high=data['high'], low=data['low'], close=data['close'], window=adx_window).adx()
            data['ATR'] = AverageTrueRange(high=data['high'], low=data['low'], close=data['close'], window=atr_window).average_true_range()
            
            return data
        except Exception as e:
            logger.error(f"Error adjusting parameters: {e}")
            raise

    def calculate_asset_characteristics(self, data: pd.DataFrame) -> Dict[str, float]:
        return {
            'volatility': data['close'].pct_change().std() * np.sqrt(252),
            'avg_volume': data['volume'].mean(),
        }


    def calculate_market_volatility(self, market_data: Optional[pd.DataFrame] = None, window: int = 252, decay_factor: float = 0.94) -> float:
        try:
            if market_data is None:
                # If no market data is provided, fetch S&P 500 data
                market_data = self.fetch_sp500_data()
            
            # Calculate daily returns
            returns = market_data['close'].pct_change().dropna()
            
            # Calculate exponentially weighted volatility
            weights = np.array([(decay_factor ** i) for i in range(window)])
            weights = weights / weights.sum()
            
            squared_returns = returns.rolling(window=window).apply(lambda x: np.sum(weights * x**2))
            volatility = np.sqrt(squared_returns * 252)  # Annualize the volatility
            
            # Return the most recent volatility value
            return volatility.iloc[-1]
        except Exception as e:
            logger.error(f"Error calculating market volatility: {e}")
            # Fallback to a default value if calculation fails
            return 0.2  # 20% annual volatility as a fallback

    def fetch_sp500_data(self) -> pd.DataFrame:
        """
        Fetch live S&P 500 data. Falls back to cached data if live fetch fails.
        
        Returns:
            pd.DataFrame: DataFrame with S&P 500 daily data including OHLCV prices
        """
        backup_file = 'sp500_backup.csv'
        
        def fetch_live_data() -> pd.DataFrame:
            """Helper to fetch and validate live data"""
            start_date = (pd.Timestamp.now() - pd.DateOffset(months=18)).strftime('%Y-%m-%d')
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            sp500 = yf.download('^GSPC', start=start_date, end=end_date, progress=False)
            
            if len(sp500) < 252:
                raise ValueError(f"Insufficient S&P 500 data: only {len(sp500)} days retrieved")
                
            sp500.columns = sp500.columns.str.lower()
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(sp500.columns):
                missing_cols = required_columns - set(sp500.columns)
                raise ValueError(f"Missing required columns: {missing_cols}")
                
            sp500['returns'] = sp500['close'].pct_change()
            sp500 = sp500.dropna()
            
            if len(sp500) < 252:
                raise ValueError("Insufficient clean data points")
                
            return sp500

        try:
            # Try to get live data
            live_data = fetch_live_data()
            
            # Save it as backup
            live_data.to_csv(backup_file)
            logger.info(f"Successfully saved new data to {backup_file}")
            
            return live_data
            
        except Exception as e:
            logger.warning(f"Failed to fetch live data: {e}")
            
            # If live data fails, try to use backup
            if os.path.exists(backup_file):
                try:
                    backup_data = pd.read_csv(backup_file, index_col=0, parse_dates=True)
                    logger.info("Successfully loaded backup data")
                    return backup_data
                except Exception as backup_e:
                    logger.error(f"Failed to load backup data: {backup_e}")
                    
            # If both live and backup fail, raise the original error
            raise Exception("Failed to fetch live data and no valid backup exists") from e

class MachineLearningModel:
    def __init__(self, lookback_period: int = 500):
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=5,
            min_samples_leaf=20,
            random_state=42
        )
        self.feature_columns = [
            'RSI', 'MACD_diff', 'ADX', 'ATR', 'Volume_Ratio', 
            'Momentum', 'Stoch_K', 'OBV', 'BB_width', 'VWAP'
        ]
        
        # Initialize overfitting controller
        self.overfitting_controller = OverfittingController(
            base_lookback_period=252,
            min_samples=100,
            max_complexity_score=0.8,
            parameter_stability_threshold=0.3
        )
        
        self.current_market_regime = "normal"
        self.performance_history = []

    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target for model training."""
        X = data[self.feature_columns].iloc[-self.lookback_period:]
        y = np.where(data['close'].pct_change().shift(-1).iloc[-self.lookback_period:] > 0, 1, 0)
        return X, y

    def calculate_performance_metrics(self, predictions: np.ndarray, actuals: np.ndarray, 
                                   returns: np.ndarray) -> PerformanceMetrics:
        """Calculate performance metrics for the model."""
        accuracy = accuracy_score(actuals, predictions)
        
        strategy_returns = returns * np.where(predictions == 1, 1, -1)
        sharpe_ratio = np.mean(strategy_returns) / (np.std(strategy_returns) + 1e-6) * np.sqrt(252)
        
        win_rate = np.mean(np.sign(strategy_returns) == np.sign(returns))
        
        # Calculate profit factor
        profitable_trades = strategy_returns[strategy_returns > 0]
        losing_trades = strategy_returns[strategy_returns < 0]
        profit_factor = (np.sum(profitable_trades) / (-np.sum(losing_trades))) if len(losing_trades) > 0 else np.inf
        
        # Calculate max drawdown
        cumulative_returns = np.cumprod(1 + strategy_returns)
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = np.min(drawdowns)
        
        return PerformanceMetrics(
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            volatility=np.std(strategy_returns) * np.sqrt(252)
        )

    def train(self, data: pd.DataFrame):
        """Train the model with overfitting protection."""
        try:
            # Split data into training and validation sets
            train_size = int(len(data) * 0.7)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:]

            # Prepare features
            X_train, y_train = self.prepare_features(train_data)
            X_val, y_val = self.prepare_features(val_data)

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            # Train model
            self.model.fit(X_train_scaled, y_train)

            # Get predictions
            train_predictions = self.model.predict(X_train_scaled)
            val_predictions = self.model.predict(X_val_scaled)

            # Calculate performance metrics
            train_returns = train_data['close'].pct_change().dropna()
            val_returns = val_data['close'].pct_change().dropna()

            in_sample_metrics = self.calculate_performance_metrics(
                train_predictions, y_train, train_returns
            )
            out_sample_metrics = self.calculate_performance_metrics(
                val_predictions, y_val, val_returns
            )

            # Get model parameters for overfitting detection
            model_parameters = {
                'n_features': len(self.feature_columns),
                'max_depth': self.model.max_depth,
                'n_estimators': self.model.n_estimators,
                'min_samples_leaf': self.model.min_samples_leaf
            }

            # Check for overfitting
            is_overfitting, overfitting_scores = self.overfitting_controller.detect_overfitting(
                in_sample_metrics=in_sample_metrics,
                out_sample_metrics=out_sample_metrics,
                market_regime=self.current_market_regime,
                model_parameters=model_parameters
            )

            if is_overfitting:
                # Adjust model parameters
                adjusted_params = self.overfitting_controller.adjust_model(
                    model=self.model,
                    overfitting_scores=overfitting_scores,
                    market_regime=self.current_market_regime
                )

                # Update model with adjusted parameters
                self.model.set_params(**adjusted_params)
                
                # Retrain with adjusted parameters
                self.model.fit(X_train_scaled, y_train)

                logger.info("Model adjusted due to detected overfitting")

            # Generate and save report
            report = self.overfitting_controller.generate_report(
                in_sample_metrics=in_sample_metrics,
                out_sample_metrics=out_sample_metrics,
                market_regime=self.current_market_regime,
                overfitting_scores=overfitting_scores
            )

            # Save report to file or database
            self._save_training_report(report)

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def predict(self, data: pd.DataFrame) -> float:
        """Generate predictions with the trained model."""
        try:
            X, _ = self.prepare_features(data)
            X_scaled = self.scaler.transform(X.iloc[-1].values.reshape(1, -1))
            return self.model.predict_proba(X_scaled)[0][1]
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def update(self, new_data: pd.DataFrame, market_regime: str):
        """Update the model with new data."""
        try:
            self.current_market_regime = market_regime
            X_new, y_new = self.prepare_features(new_data)
            X_scaled = self.scaler.transform(X_new)
            
            # Get current predictions
            predictions = self.model.predict(X_scaled)
            returns = new_data['close'].pct_change().dropna()
            
            # Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics(
                predictions, y_new, returns
            )
            
            # Store performance history
            self.performance_history.append(performance_metrics)
            
            # Update the model
            self.model.partial_fit(X_scaled, y_new)
            
            logger.info("Model successfully updated with new data")
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return dict(zip(self.feature_columns, self.model.feature_importances_))

    def _save_training_report(self, report: Dict[str, Any]) -> None:
        """Save training report to file or database."""
        try:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"training_reports/model_report_{timestamp}.json"
            
            os.makedirs("training_reports", exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            logger.info(f"Training report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving training report: {e}")

    def save_model(self, path: str) -> None:
        """Save model state and overfitting controller state."""
        try:
            model_state = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'lookback_period': self.lookback_period,
                'current_market_regime': self.current_market_regime
            }
            
            # Save model state
            joblib.dump(model_state, f"{path}_model.joblib")
            
            # Save overfitting controller state
            self.overfitting_controller.save_state(f"{path}_overfitting_controller.json")
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """Load model state and overfitting controller state."""
        try:
            # Load model state
            model_state = joblib.load(f"{path}_model.joblib")
            
            self.model = model_state['model']
            self.scaler = model_state['scaler']
            self.feature_columns = model_state['feature_columns']
            self.lookback_period = model_state['lookback_period']
            self.current_market_regime = model_state['current_market_regime']
            
            # Load overfitting controller state
            self.overfitting_controller.load_state(f"{path}_overfitting_controller.json")
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

class PerformanceLagMitigation:
    def __init__(self, base_adjustment_period: int = 20):
        self.base_adjustment_period = base_adjustment_period
        self.performance_history = []
        self.last_adjustment_time = None
        self.learning_rate = 1.0
        self.learning_rate_decay = 0.995  # Decay factor for learning rate
        
    def should_adjust(self, current_metrics: Dict[str, float], 
                     threshold: float = 0.15) -> bool:
        """
        Determine if adjustment is needed based on performance change magnitude
        and time since last adjustment.
        """
        if not self.performance_history:
            return False
            
        # Calculate performance change
        recent_perf = np.mean([m['sharpe_ratio'] for m in self.performance_history[-5:]])
        current_perf = current_metrics['sharpe_ratio']
        perf_change = abs(current_perf - recent_perf) / max(abs(recent_perf), 1e-6)
        
        # Check if change exceeds threshold
        if perf_change > threshold:
            if self.last_adjustment_time is None:
                return True
            
            # Ensure minimum time between adjustments
            time_since_last = len(self.performance_history) - self.last_adjustment_time
            return time_since_last >= self.base_adjustment_period
            
        return False
        
    def calculate_adjustment_factor(self, 
                                  short_term_metrics: Dict[str, float],
                                  long_term_metrics: Dict[str, float],
                                  volatility: float) -> float:
        """
        Calculate adjustment factor using hybrid approach and current volatility.
        """
        # Weight recent performance more in high volatility
        short_term_weight = min(0.7, 0.4 + volatility)
        long_term_weight = 1.0 - short_term_weight
        
        short_term_factor = self._calculate_performance_factor(short_term_metrics)
        long_term_factor = self._calculate_performance_factor(long_term_metrics)
        
        # Combine factors with volatility-adjusted weights
        adjustment = (short_term_weight * short_term_factor + 
                     long_term_weight * long_term_factor)
        
        # Apply learning rate decay
        self.learning_rate *= self.learning_rate_decay
        adjustment *= self.learning_rate
        
        # Limit maximum adjustment
        return np.clip(adjustment, -0.2, 0.2)
    
    def _calculate_performance_factor(self, metrics: Dict[str, float]) -> float:
        """
        Calculate performance factor from metrics.
        """
        sharpe_contribution = metrics.get('sharpe_ratio', 0) * 0.4
        win_rate_contribution = (metrics.get('win_rate', 0.5) - 0.5) * 0.3
        dd_contribution = -abs(metrics.get('max_drawdown', 0)) * 0.3
        
        return sharpe_contribution + win_rate_contribution + dd_contribution
    
    def update_history(self, metrics: Dict[str, float]):
        """
        Update performance history and manage history length.
        """
        self.performance_history.append(metrics)
        if len(self.performance_history) > self.base_adjustment_period * 3:
            self.performance_history.pop(0)

class SignalGenerator:
    def __init__(self, db_path: str, model_path: str):
        self.db_handler = DatabaseHandler(db_path)
        self.indicator_calculator = IndicatorCalculator()
        self.market_condition_analyzer = MarketConditionAnalyzer()
        self.parameter_adjuster = ParameterAdjuster()
        self.ml_model = MachineLearningModel()
        self.model_path = model_path
        
        # Add lag mitigation system
        self.lag_mitigation = PerformanceLagMitigation()
        
        # Track performance metrics
        self.short_term_metrics = {}
        self.long_term_metrics = {}
        self.current_volatility = 0.0

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = self.indicator_calculator.calculate_indicators(data)
        return self.parameter_adjuster.adjust_parameters(df, data.iloc[-1]['symbol'])

    def check_signal(self, data: pd.DataFrame) -> int:
        # Update volatility estimate
        self.current_volatility = self._calculate_volatility(data)
        
        # Calculate performance metrics
        self._update_performance_metrics(data)
        
        # Check if parameters need adjustment
        if self.lag_mitigation.should_adjust(self.short_term_metrics):
            adjustment_factor = self.lag_mitigation.calculate_adjustment_factor(
                self.short_term_metrics,
                self.long_term_metrics,
                self.current_volatility
            )
            self._adjust_parameters(adjustment_factor)
        
        signal_probability = self.ml_model.predict(data)
        market_condition = self.market_condition_analyzer.determine_market_condition(data)
        thresholds = self.calculate_dynamic_thresholds(market_condition, data)
        
        if signal_probability > thresholds['buy']:
            return 1
        elif signal_probability < thresholds['sell']:
            return -1
        return 0

    def get_indicator_values(self, data: pd.DataFrame) -> Dict[str, Tuple[float, float, float]]:
        """Get the latest indicator values along with their means and standard deviations."""
        indicators = {
            'RSI': (data['RSI'].iloc[-1], data['RSI'].mean(), data['RSI'].std()),
            'MACD': (data['MACD'].iloc[-1], data['MACD'].mean(), data['MACD'].std()),
            'ATR': (data['ATR'].iloc[-1], data['ATR'].mean(), data['ATR'].std()),
            'Stochastic': (data['Stoch_K'].iloc[-1], data['Stoch_K'].mean(), data['Stoch_K'].std()),
            'BB_width': (data['BB_width'].iloc[-1], data['BB_width'].mean(), data['BB_width'].std()),
        }
        return indicators

    def get_market_condition_win_probability(self, market_condition: str) -> float:
        """Get the win probability for a given market condition based on historical data."""
        condition_probabilities = {
            'Uptrend-High Volatility': 0.6,
            'Uptrend-Low Volatility': 0.7,
            'Downtrend-High Volatility': 0.4,
            'Downtrend-Low Volatility': 0.5,
            'Ranging-High Volatility': 0.5,
            'Ranging-Low Volatility': 0.55
        }
        return condition_probabilities.get(market_condition, 0.5)

    def get_recent_signals(self, data: pd.DataFrame, lookback: int = 50) -> List[int]:
        recent_data = data.tail(lookback)
        return [self.check_signal(recent_data.iloc[:i+1]) for i in range(len(recent_data))]

    def calculate_dynamic_thresholds(self, market_condition: str, data: pd.DataFrame) -> Dict[str, float]:
        base_threshold = {'buy': 0.7, 'sell': 0.3}
        
        # Adjust based on market condition
        if 'Uptrend' in market_condition:
            base_threshold['buy'] *= 0.9  # More sensitive to buy signals
            base_threshold['sell'] *= 1.1  # Less sensitive to sell signals
        elif 'Downtrend' in market_condition:
            base_threshold['buy'] *= 1.1  # Less sensitive to buy signals
            base_threshold['sell'] *= 0.9  # More sensitive to sell signals
            
        # Adjust based on volatility
        if self.current_volatility > data['ATR'].mean():
            base_threshold['buy'] *= 1.1  # More conservative in high volatility
            base_threshold['sell'] *= 1.1
            
        return base_threshold

    def update_model(self, new_data: pd.DataFrame):
        self.ml_model.update(new_data)
        joblib.dump(self.ml_model, self.model_path)
        logger.info(f"Model updated and saved")

    def get_feature_importance(self) -> Dict[str, float]:
        return self.ml_model.get_feature_importance()

    def calculate_confidence_score(self, data: pd.DataFrame, market_condition: str) -> float:
        lwr = self.calculate_recent_performance(data)['win_rate']
        bp = self.get_market_condition_win_probability(market_condition)
        mcwp = self.get_market_condition_win_probability(market_condition)
        indicators = self.get_indicator_values(data)
        ssm = self.calculate_ssm(indicators)
        current_volatility = data['ATR'].iloc[-1]
        average_volatility = data['ATR'].mean()
        vaf = self.calculate_vaf(current_volatility, average_volatility)
        
        confidence_calculator = ConfidenceScoreCalculator()
        return confidence_calculator.calculate_confidence_score(lwr, bp, mcwp, ssm, vaf)

    def calculate_ssm(self, indicators: Dict[str, Tuple[float, float, float]]) -> float:
        weighted_sum = sum(
            (indicator[0] - indicator[1]) / indicator[2] * weight
            for indicator, weight in zip(indicators.values(), [0.3, 0.3, 0.2, 0.1, 0.1])
        )
        return 1 / (1 + np.exp(-weighted_sum))

    def calculate_vaf(self, current_volatility: float, average_volatility: float) -> float:
        return 1 - abs((current_volatility - average_volatility) / average_volatility)

    def calculate_recent_performance(self, data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
        recent_data = data.tail(window)
        returns = recent_data['close'].pct_change()
        return {
            'avg_return': returns.mean(),
            'win_rate': (returns > 0).mean(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        }

    def run(self, ticker: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], pd.DataFrame]:
        data = self.db_handler.load_market_data(ticker)
        data = data[(data.index >= start_date) & (data.index <= end_date)]
        
        results = []
        for i in range(len(data)):
            if i < 200:  # Skip the first 200 rows to have enough data for indicators
                continue
            
            current_data = data.iloc[max(0, i-200):i+1]
            signal = self.check_signal(current_data)
            
            if signal != 0:
                market_condition = self.market_condition_analyzer.determine_market_condition(current_data)
                confidence_score = self.calculate_confidence_score(current_data, market_condition)
                
                results.append({
                    'date': current_data.index[-1],
                    'signal': signal,
                    'market_condition': market_condition,
                    'confidence_score': confidence_score
                })
        
        results_df = pd.DataFrame(results)
        performance_metrics = self.calculate_performance_metrics(results_df)
        recent_performance = self.calculate_recent_performance(data)
        
        return results_df, performance_metrics, recent_performance, self.perform_walk_forward_optimization(data)

    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """Calculate recent volatility."""
        returns = data['close'].pct_change().dropna()
        return returns.std() * np.sqrt(252)

    def _update_performance_metrics(self, data: pd.DataFrame):
        """Update short and long-term performance metrics."""
        recent_data = data.tail(20)  # Short-term window
        long_term_data = data.tail(100)  # Long-term window
        
        self.short_term_metrics = {
            'sharpe_ratio': self._calculate_sharpe(recent_data),
            'win_rate': self._calculate_win_rate(recent_data),
            'max_drawdown': self._calculate_max_drawdown(recent_data)
        }
        
        self.long_term_metrics = {
            'sharpe_ratio': self._calculate_sharpe(long_term_data),
            'win_rate': self._calculate_win_rate(long_term_data),
            'max_drawdown': self._calculate_max_drawdown(long_term_data)
        }
        
        self.lag_mitigation.update_history(self.short_term_metrics)

    def _adjust_parameters(self, adjustment_factor: float):
        """Apply adjustment factor to strategy parameters."""
        current_params = self.ml_model.get_parameters()
        
        adjusted_params = {}
        for param, value in current_params.items():
            if param in ['learning_rate', 'threshold', 'window_size']:
                adjusted_params[param] = value * (1 + adjustment_factor)
        
        adjusted_params = self._apply_parameter_bounds(adjusted_params)
        self.ml_model.set_parameters(adjusted_params)

    def _apply_parameter_bounds(self, params: Dict[str, float]) -> Dict[str, float]:
        bounds = {
            'learning_rate': (0.0001, 0.1),
            'threshold': (0.1, 0.9),
            'window_size': (10, 200)
        }
        
        bounded_params = {}
        for param, value in params.items():
            if param in bounds:
                bounded_params[param] = np.clip(value, bounds[param][0], bounds[param][1])
            else:
                bounded_params[param] = value
                
        return bounded_params

    def _calculate_sharpe(self, data: pd.DataFrame) -> float:
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.0
        return np.sqrt(252) * returns.mean() / (returns.std() + 1e-6)

    def _calculate_win_rate(self, data: pd.DataFrame) -> float:
        returns = data['close'].pct_change().dropna()
        if len(returns) == 0:
            return 0.5
        return (returns > 0).mean()

    def _calculate_max_drawdown(self, data: pd.DataFrame) -> float:
        prices = data['close']
        peak = prices.expanding(min_periods=1).max()
        drawdown = (prices - peak) / peak
        return abs(drawdown.min())

    def calculate_performance_metrics(self, results: pd.DataFrame) -> Dict[str, float]:
        if results.empty:
            return {'accuracy': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}
        
        signals = results['signal']
        returns = pd.Series(index=results.index)
        
        # Calculate returns based on signals
        for i in range(1, len(signals)):
            returns.iloc[i] = signals.iloc[i-1] * (results.index[i] - results.index[i-1]).days
        
        return {
            'accuracy': (signals != 0).mean(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
            'max_drawdown': self._calculate_max_drawdown(returns.cumsum().to_frame('close'))
        }
    def perform_walk_forward_optimization(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Perform walk-forward optimization using the existing OverfittingController.
        
        Args:
            data: Historical market data
            
        Returns:
            DataFrame containing optimization results
        """
        # Use ML model's overfitting controller which already has this functionality
        return self.ml_model.overfitting_controller.perform_walk_forward_analysis(data)


class RiskManagement:
    def __init__(self, backtest_results: pd.DataFrame, market_conditions_file: str, db_path: str, market_index: str = 'SPY'):
        self.backtest_results = backtest_results
        self.scaler = RobustScaler()
        self.volatility_model = None
        self.market_conditions_df = self.load_market_conditions(market_conditions_file)
        self.db_path = db_path
        self.market_index = market_index
        self.market_returns = self.load_market_returns()
        self.train_volatility_model()

    def load_market_conditions(self, file_path: str) -> pd.DataFrame:
        """Load Market Conditions Overview from a CSV file."""
        return pd.read_csv(file_path, encoding='ISO-8859-1')

    def load_market_data(self, ticker: str) -> pd.DataFrame:
        """Load Market Data for a specific ticker from the SQLite Database."""
        conn = sqlite3.connect(self.db_path)
        market_data_df = pd.read_sql_query(f"SELECT * FROM '{ticker}'", conn)
        conn.close()
        return market_data_df

    def load_market_returns(self) -> pd.Series:
        """Load and calculate returns for the market index."""
        market_data = self.load_market_data(self.market_index)
        market_data['return'] = market_data['close'].pct_change()
        return market_data['return']

    def train_volatility_model(self):
        X = self.backtest_results[['Market_Condition_ID', 'Avg_Return', 'Max_Drawdown']]
        y = self.backtest_results['Volatility']
        self.volatility_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.volatility_model.fit(X, y)

    def predict_volatility(self, market_condition_id: int, avg_return: float, max_drawdown: float) -> float:
        X = np.array([[market_condition_id, avg_return, max_drawdown]])
        X_scaled = self.scaler.fit_transform(X)
        return self.volatility_model.predict(X_scaled)[0]

    def adjust_risk_based_on_market_condition(self, market_condition: str, live_performance: Dict[str, float], confidence_score: float) -> Dict[str, float]:
        condition_id = self.get_market_condition_id(market_condition)
        avg_return = live_performance.get('avg_return', 0)
        max_drawdown = live_performance.get('drawdown', 0)
        predicted_volatility = self.predict_volatility(condition_id, avg_return, max_drawdown)

        base_sl = config.BASE_STOP_LOSS
        base_tp = config.BASE_TAKE_PROFIT

        volatility_factor = predicted_volatility / self.backtest_results['Volatility'].mean()
        confidence_factor = 1 + (confidence_score - 0.5)  # Adjust based on confidence score

        adjusted_sl = base_sl * volatility_factor / confidence_factor
        adjusted_tp = base_tp * volatility_factor * confidence_factor

        return {
            'stop_loss': min(max(adjusted_sl, config.MIN_STOP_LOSS), config.MAX_STOP_LOSS),
            'take_profit': min(max(adjusted_tp, config.MIN_TAKE_PROFIT), config.MAX_TAKE_PROFIT)
        }

    def get_market_condition_id(self, market_condition: str) -> Optional[int]:
        """Get the Market Condition ID based on the Market Condition Description."""
        condition_row = self.market_conditions_df[self.market_conditions_df['Market Condition Description'] == market_condition]
        
        if not condition_row.empty:
            return condition_row['Market Condition ID'].values[0]
        else:
            logger.warning(f"No match found for condition: {market_condition}")
            return None

    def calculate_dynamic_risk_reward_ratio(self, market_condition: str, confidence_score: float) -> float:
        base_ratio = config.BASE_RISK_REWARD_RATIO
        
        # Adjust based on market condition
        if "Uptrend" in market_condition and "Strong" in market_condition:
            base_ratio *= 1.2
        elif "Downtrend" in market_condition and "Strong" in market_condition:
            base_ratio *= 0.8
        
        # Adjust based on confidence score
        confidence_factor = 1 + (confidence_score - 0.5)
        
        return base_ratio * confidence_factor

    def calculate_position_size(self, account_balance: float, risk_per_trade: float, entry_price: float, stop_loss: float) -> float:
        risk_amount = account_balance * risk_per_trade
        risk_per_share = abs(entry_price - stop_loss)
        return risk_amount / risk_per_share

    def calculate_dynamic_position_size(self, account_balance: float, risk_per_trade: float, entry_price: float, stop_loss: float, confidence_score: float, asset_volatility: float, market_volatility: float) -> float:
        """Calculate dynamic position size based on confidence score and volatility."""
        base_position_size = self.calculate_position_size(account_balance, risk_per_trade, entry_price, stop_loss)
        volatility_factor = market_volatility / asset_volatility
        confidence_factor = 1 + (confidence_score - 0.5)
        return base_position_size * volatility_factor * confidence_factor

    def adjust_for_partial_exit(self, initial_stop_loss: float, current_price: float, exit_percentage: float) -> float:
        """Adjust stop loss for partial exit strategy."""
        if exit_percentage > 0:
            # Move stop loss to break-even or better after partial exit
            return max(initial_stop_loss, current_price)
        return initial_stop_loss

    def calculate_trailing_stop(self, initial_stop_loss: float, highest_price: float, atr: float, multiplier: float = 2) -> float:
        """Calculate trailing stop loss."""
        trailing_stop = highest_price - (multiplier * atr)
        return max(trailing_stop, initial_stop_loss)

    def evaluate_exit_condition(self, current_price: float, entry_price: float, stop_loss: float, take_profit: float, trailing_stop: float, max_holding_period: int, trade_duration: int) -> Tuple[bool, Optional[str]]:
        """Evaluate if any exit condition is met."""
        if current_price <= stop_loss:
            return True, "Stop Loss"
        if current_price >= take_profit:
            return True, "Take Profit"
        if current_price <= trailing_stop:
            return True, "Trailing Stop"
        if trade_duration >= max_holding_period:
            return True, "Max Holding Period"
        return False, None

    def calculate_var_cvar(self, returns: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
        """Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)."""
        sorted_returns = np.sort(returns)
        index = int((1 - confidence_level) * len(sorted_returns))
        var = -sorted_returns[index]
        cvar = -sorted_returns[:index].mean()
        return var, cvar

    def calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe Ratio."""
        excess_returns = returns - risk_free_rate
        return np.mean(excess_returns) / np.std(excess_returns)

    def calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino Ratio."""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = np.std(downside_returns)
        return np.mean(excess_returns) / downside_deviation if downside_deviation != 0 else np.inf

    def perform_stress_test(self, strategy_returns: np.ndarray, stress_scenario: Dict[str, float]) -> Dict[str, float]:
        """Perform stress testing on the strategy."""
        stressed_returns = strategy_returns * stress_scenario.get('market_shock', 1)
        stressed_sharpe = self.calculate_sharpe_ratio(stressed_returns)
        stressed_sortino = self.calculate_sortino_ratio(stressed_returns)
        stressed_var, stressed_cvar = self.calculate_var_cvar(stressed_returns)
        
        # Calculate beta under stress
        market_returns_stress = self.market_returns * stress_scenario.get('market_shock', 1)
        beta_stress = np.cov(stressed_returns, market_returns_stress)[0, 1] / np.var(market_returns_stress)
        
        return {
            'stressed_sharpe_ratio': stressed_sharpe,
            'stressed_sortino_ratio': stressed_sortino,
            'stressed_var': stressed_var,
            'stressed_cvar': stressed_cvar,
            'stressed_beta': beta_stress
        }

    def calculate_max_drawdown(self, cumulative_returns: np.ndarray) -> float:
        """Calculate the maximum drawdown."""
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        return np.min(drawdown)

    def monitor_drawdown(self, current_drawdown: float) -> bool:
        """Monitor drawdown and return True if it exceeds the acceptable threshold."""
        return current_drawdown > config.MAX_ACCEPTABLE_DRAWDOWN

    def adjust_for_liquidity(self, position_size: float, average_volume: float) -> float:
        """Adjust position size based on liquidity considerations."""
        max_position_size = average_volume * config.MAX_VOLUME_PERCENTAGE
        return min(position_size, max_position_size)

    def calculate_kelly_criterion(self, win_rate: float, win_loss_ratio: float) -> float:
        """Calculate the optimal fraction to invest using the Kelly Criterion."""
        return (win_rate * win_loss_ratio - (1 - win_rate)) / win_loss_ratio

    def apply_kelly_criterion(self, position_size: float, kelly_fraction: float) -> float:
        """Apply Kelly Criterion to adjust position size."""
        return position_size * kelly_fraction * config.KELLY_FRACTION_LIMIT

    def diversification_check(self, current_positions: Dict[str, float], new_position: float, asset: str) -> bool:
        """Check if adding a new position maintains proper diversification."""
        total_exposure = sum(current_positions.values()) + new_position
        asset_exposure = (current_positions.get(asset, 0) + new_position) / total_exposure
        return asset_exposure <= config.MAX_ASSET_EXPOSURE

    def adjust_for_correlation(self, position_size: float, asset: str, correlations: Dict[str, float]) -> float:
        """Adjust position size based on correlation with existing positions."""
        avg_correlation = sum(correlations.values()) / len(correlations) if correlations else 0
        correlation_factor = 1 - (avg_correlation * config.CORRELATION_IMPACT_FACTOR)
        return position_size * correlation_factor

    def calculate_risk_adjusted_return(self, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """Calculate risk-adjusted return (e.g., Treynor Ratio)."""
        excess_returns = returns - risk_free_rate
        market_excess_returns = self.market_returns - risk_free_rate
        beta = np.cov(excess_returns, market_excess_returns)[0, 1] / np.var(market_excess_returns)
        return np.mean(excess_returns) / beta if beta != 0 else np.inf

    def update_risk_model(self, new_data: pd.DataFrame):
        """Update the risk model with new data."""
        X = new_data[['Market_Condition_ID', 'Avg_Return', 'Max_Drawdown']]
        y = new_data['Volatility']
        self.volatility_model.partial_fit(X, y)

    def calculate_beta(self, returns: np.ndarray) -> float:
        """Calculate beta of the strategy relative to the market."""
        covariance = np.cov(returns, self.market_returns)[0, 1]
        market_variance = np.var(self.market_returns)
        return covariance / market_variance if market_variance != 0 else np.inf

    def generate_risk_report(self, positions: Dict[str, float], returns: np.ndarray, market_conditions: Dict[str, str]) -> Dict[str, float]:
        """Generate a comprehensive risk report."""
        var, cvar = self.calculate_var_cvar(returns)
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        max_drawdown = self.calculate_max_drawdown(np.cumprod(1 + returns))
        beta = self.calculate_beta(returns)
        risk_adjusted_return = self.calculate_risk_adjusted_return(returns)
        
        return {
            'total_exposure': sum(positions.values()),
            'var': var,
            'cvar': cvar,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'beta': beta,
            'risk_adjusted_return': risk_adjusted_return,
            'current_market_conditions': market_conditions
        }

    def get_backtest_win_rate(self, market_condition: str) -> float:
        """Get the win rate from backtest results for a specific market condition."""
        condition_id = self.get_market_condition_id(market_condition)
        if condition_id is not None:
            condition_results = self.backtest_results[self.backtest_results['Market_Condition_ID'] == condition_id]
            if not condition_results.empty:
                return condition_results['Win_Rate'].values[0]
        return 0.5  # Default win rate if no data is found

class TakeProfitStopLoss:
    def __init__(self, atr_multiplier: float = 2, 
                 confidence_factor: float = 0.2, 
                 volatility_factor: float = 0.1):
        self.atr_multiplier = atr_multiplier
        self.confidence_factor = confidence_factor
        self.volatility_factor = volatility_factor

    def calculate_tp_sl(self, entry_price: float, atr: float, 
                        risk_parameters: Dict[str, float], 
                        trade_direction: str, 
                        confidence_score: float,
                        current_volatility: float,
                        average_volatility: float) -> Tuple[float, float]:
        """
        Calculate initial stop loss and take profit levels.

        :param entry_price: The entry price for the trade
        :param atr: Average True Range
        :param risk_parameters: Dictionary containing base stop loss and take profit multipliers
        :param trade_direction: 'long' or 'short'
        :param confidence_score: The confidence score for the trade (0-1)
        :param current_volatility: Current market volatility
        :param average_volatility: Average historical volatility
        :return: Tuple of (stop_loss, take_profit)
        """
        base_sl = risk_parameters['stop_loss']
        base_tp = risk_parameters['take_profit']

        # Adjust based on confidence score
        confidence_adjustment = 1 + (confidence_score - 0.5) * self.confidence_factor
        
        # Adjust based on volatility
        volatility_ratio = current_volatility / average_volatility
        volatility_adjustment = 1 + (volatility_ratio - 1) * self.volatility_factor

        adjusted_atr = atr * confidence_adjustment * volatility_adjustment

        if trade_direction == 'long':
            stop_loss = entry_price - (base_sl * adjusted_atr * self.atr_multiplier)
            take_profit = entry_price + (base_tp * adjusted_atr * self.atr_multiplier)
        else:  # short
            stop_loss = entry_price + (base_sl * adjusted_atr * self.atr_multiplier)
            take_profit = entry_price - (base_tp * adjusted_atr * self.atr_multiplier)

        return round(stop_loss, 2), round(take_profit, 2)

    def adjust_tp_sl(self, current_price: float, entry_price: float, 
                     stop_loss: float, take_profit: float, atr: float, 
                     trade_direction: str, market_condition: str) -> Tuple[float, float]:
        """
        Adjust stop loss and take profit levels based on price movement and market conditions.

        :param current_price: Current price of the asset
        :param entry_price: Original entry price
        :param stop_loss: Current stop loss level
        :param take_profit: Current take profit level
        :param atr: Current Average True Range
        :param trade_direction: 'long' or 'short'
        :param market_condition: Current market condition (e.g., 'trending', 'ranging')
        :return: Tuple of (new_stop_loss, new_take_profit)
        """
        # Implement advanced trailing stop loss
        if trade_direction == 'long':
            new_stop_loss = max(stop_loss, self._calculate_trailing_stop(current_price, atr, 'long', market_condition))
        else:  # short
            new_stop_loss = min(stop_loss, self._calculate_trailing_stop(current_price, atr, 'short', market_condition))

        # Adjust take profit dynamically
        new_take_profit = self._adjust_take_profit(current_price, entry_price, take_profit, atr, trade_direction, market_condition)

        return round(new_stop_loss, 2), round(new_take_profit, 2)

    def _calculate_trailing_stop(self, current_price: float, atr: float, 
                                 trade_direction: str, market_condition: str) -> float:
        """
        Calculate trailing stop based on market conditions.

        :param current_price: Current price of the asset
        :param atr: Current Average True Range
        :param trade_direction: 'long' or 'short'
        :param market_condition: Current market condition
        :return: New trailing stop level
        """
        base_distance = self.atr_multiplier * atr
        
        if market_condition == 'trending':
            # In trending markets, we might want a wider trailing stop
            distance = base_distance * 1.5
        elif market_condition == 'ranging':
            # In ranging markets, we might want a tighter trailing stop
            distance = base_distance * 0.8
        else:
            distance = base_distance

        if trade_direction == 'long':
            return current_price - distance
        else:  # short
            return current_price + distance

    def _adjust_take_profit(self, current_price: float, entry_price: float, 
                            take_profit: float, atr: float, 
                            trade_direction: str, market_condition: str) -> float:
        """
        Dynamically adjust take profit based on price movement and market conditions.

        :param current_price: Current price of the asset
        :param entry_price: Original entry price
        :param take_profit: Current take profit level
        :param atr: Current Average True Range
        :param trade_direction: 'long' or 'short'
        :param market_condition: Current market condition
        :return: New take profit level
        """
        price_movement = abs(current_price - entry_price)
        
        if market_condition == 'trending':
            # In trending markets, we might want to let profits run
            tp_distance = max(price_movement, self.atr_multiplier * atr * 2)
        elif market_condition == 'ranging':
            # In ranging markets, we might want to take profits quicker
            tp_distance = min(price_movement, self.atr_multiplier * atr)
        else:
            tp_distance = self.atr_multiplier * atr

        if trade_direction == 'long':
            new_take_profit = max(take_profit, entry_price + tp_distance)
        else:  # short
            new_take_profit = min(take_profit, entry_price - tp_distance)

        return new_take_profit

    def calculate_risk_reward_ratio(self, entry_price: float, stop_loss: float, take_profit: float) -> float:
        """
        Calculate the risk-reward ratio for a trade.

        :param entry_price: Entry price of the trade
        :param stop_loss: Stop loss level
        :param take_profit: Take profit level
        :return: Risk-reward ratio
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        return reward / risk if risk != 0 else 0

    def adjust_for_market_volatility(self, stop_loss: float, take_profit: float, 
                                     entry_price: float, current_volatility: float, 
                                     average_volatility: float, trade_direction: str) -> Tuple[float, float]:
        """
        Adjust stop loss and take profit levels based on current market volatility.

        :param stop_loss: Current stop loss level
        :param take_profit: Current take profit level
        :param entry_price: Entry price of the trade
        :param current_volatility: Current market volatility
        :param average_volatility: Average historical volatility
        :param trade_direction: 'long' or 'short'
        :return: Tuple of (adjusted_stop_loss, adjusted_take_profit)
        """
        volatility_ratio = current_volatility / average_volatility
        adjustment_factor = np.clip(volatility_ratio, 0.5, 2.0)  # Limit adjustment to 0.5x to 2x

        sl_distance = abs(entry_price - stop_loss)
        tp_distance = abs(take_profit - entry_price)

        adjusted_sl_distance = sl_distance * adjustment_factor
        adjusted_tp_distance = tp_distance * adjustment_factor

        if trade_direction == 'long':
            adjusted_stop_loss = entry_price - adjusted_sl_distance
            adjusted_take_profit = entry_price + adjusted_tp_distance
        else:  # short
            adjusted_stop_loss = entry_price + adjusted_sl_distance
            adjusted_take_profit = entry_price - adjusted_tp_distance

        return round(adjusted_stop_loss, 2), round(adjusted_take_profit, 2)

class PositionSizing:
    def __init__(self, account_balance: float, max_risk_per_trade: float, 
                 min_risk_reward_ratio: float = 2.0, 
                 volatility_adjustment_factor: float = 0.5):
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.min_risk_reward_ratio = min_risk_reward_ratio
        self.volatility_adjustment_factor = volatility_adjustment_factor

    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                                take_profit: float, confidence_score: float, 
                                current_volatility: float, average_volatility: float, 
                                market_data: Dict[str, Any]) -> float:
        """
        Calculate the position size based on various factors including the confidence score.
        
        :param entry_price: The entry price for the trade
        :param stop_loss: The stop loss price
        :param take_profit: The take profit price
        :param confidence_score: The confidence score for the trade (0-1)
        :param current_volatility: The current market volatility
        :param average_volatility: The average historical volatility
        :param market_data: Additional market data for advanced calculations
        :return: The calculated position size
        """
        # Calculate base risk amount
        risk_amount = self.account_balance * self.max_risk_per_trade

        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)

        # Check risk-reward ratio
        reward_per_share = abs(take_profit - entry_price)
        risk_reward_ratio = reward_per_share / risk_per_share
        if risk_reward_ratio < self.min_risk_reward_ratio:
            return 0  # Don't take the trade if it doesn't meet the minimum risk-reward ratio

        # Calculate base position size
        base_position_size = risk_amount / risk_per_share

        # Adjust position size based on confidence score
        confidence_adjusted_size = base_position_size * confidence_score

        # Volatility-based adjustment
        volatility_ratio = current_volatility / average_volatility
        volatility_adjustment = 1 + (volatility_ratio - 1) * self.volatility_adjustment_factor
        volatility_adjusted_size = confidence_adjusted_size / volatility_adjustment

        # Ensure position size doesn't exceed account balance
        max_affordable_shares = self.account_balance / entry_price
        final_position_size = min(volatility_adjusted_size, max_affordable_shares)

        # Apply any additional adjustments based on market data
        final_position_size = self._apply_market_adjustments(final_position_size, market_data)

        return round(final_position_size, 2)

    def _apply_market_adjustments(self, position_size: float, market_data: Dict[str, Any]) -> float:
        """
        Apply additional adjustments based on market conditions.
        
        :param position_size: The current calculated position size
        :param market_data: Additional market data for advanced calculations
        :return: The adjusted position size
        """
        # Example: Reduce position size in high volatility conditions
        if market_data.get('high_volatility', False):
            position_size *= 0.8

        # Example: Increase position size in strong trend conditions
        if market_data.get('strong_trend', False):
            position_size *= 1.2

        return position_size

    def update_account_balance(self, new_balance: float) -> None:
        """
        Update the account balance.
        
        :param new_balance: The new account balance
        """
        self.account_balance = new_balance

    def calculate_dynamic_risk_per_trade(self, performance_metric: float) -> None:
        """
        Dynamically adjust the maximum risk per trade based on recent performance.
        
        :param performance_metric: A metric representing recent trading performance (e.g., Sharpe ratio)
        """
        base_risk = 0.01  # 1% base risk
        max_risk = 0.02  # 2% maximum risk
        self.max_risk_per_trade = np.clip(base_risk * performance_metric, base_risk, max_risk)

    def adjust_position_size_for_confidence(self, base_size: float, confidence_score: float) -> float:
        """
        Adjust the position size based on the confidence score.
        
        :param base_size: The base position size
        :param confidence_score: The confidence score (0-1)
        :return: The adjusted position size
        """
        return base_size * (0.5 + confidence_score / 2)  # Scale between 50% and 100% of base size

    def calculate_position_value(self, position_size: float, current_price: float) -> float:
        """
        Calculate the current value of a position.
        
        :param position_size: The size of the position
        :param current_price: The current price of the asset
        :return: The current value of the position
        """
        return position_size * current_price

    def calculate_portfolio_heat(self, open_positions: Dict[str, Dict[str, float]]) -> float:
        """
        Calculate the current portfolio heat (total risk exposure).
        
        :param open_positions: A dictionary of open positions with their sizes and current prices
        :return: The total portfolio heat as a percentage of account balance
        """
        total_risk = sum(
            self.calculate_position_value(pos['size'], pos['current_price']) * self.max_risk_per_trade
            for pos in open_positions.values()
        )
        return (total_risk / self.account_balance) * 100

class ConfidenceScoreCalculator:
    def __init__(self, lookback_period: int = 20):
        self.lookback_period = lookback_period

    def calculate_confidence_score(self, lwr: float, bp: float, mcwp: float, ssm: float, vaf: float) -> float:
        weights = self._calculate_weights(lwr, bp)
        cs = (
            weights['lwr'] * lwr +
            weights['bp'] * bp +
            weights['mcwp'] * mcwp +
            weights['ssm'] * ssm +
            weights['vaf'] * vaf
        )
        return min(max(cs, 0), 1)  # Ensure score is between 0 and 1

    def _calculate_weights(self, lwr: float, bp: float) -> Dict[str, float]:
        performance_factor = lwr / bp if bp != 0 else 1
        w_lwr = 0.4 * performance_factor
        w_bp = 0.3 / performance_factor
        w_mcwp = 0.2
        w_ssm = 0.1
        w_vaf = 0.0

        total = w_lwr + w_bp + w_mcwp + w_ssm + w_vaf
        return {
            'lwr': w_lwr / total,
            'bp': w_bp / total,
            'mcwp': w_mcwp / total,
            'ssm': w_ssm / total,
            'vaf': w_vaf / total
        }

    def calculate_lwr(self, data: pd.DataFrame) -> float:
        recent_trades = data.tail(self.lookback_period)
        return (recent_trades['profit'] > 0).mean()

    def calculate_ssm(self, indicators: Dict[str, float]) -> float:
        weighted_sum = sum(
            (indicator - indicators[f'{name}_mean']) / indicators[f'{name}_std'] * weight
            for name, (indicator, weight) in indicators.items()
        )
        return 1 / (1 + np.exp(-weighted_sum))  # Sigmoid normalization

    def calculate_vaf(self, current_volatility: float, average_volatility: float) -> float:
        return 1 - abs((current_volatility - average_volatility) / average_volatility)

class MarketRegime(Enum):
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    RANGING = "ranging"
    NORMAL = "normal"

@dataclass
class PerformanceMetrics:
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    volatility: float

    def to_dict(self) -> Dict[str, float]:
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility
        }

class OverfittingController:
    """
    Enhanced overfitting controller with market regime awareness and adaptive thresholds.
    """
    
    def __init__(self, 
                 base_lookback_period: int = 252,
                 min_samples: int = 100,
                 max_complexity_score: float = 0.8,
                 parameter_stability_threshold: float = 0.3):
        """
        Initialize the overfitting controller with configurable parameters.

        Args:
            base_lookback_period: Base period for historical analysis
            min_samples: Minimum number of samples required for analysis
            max_complexity_score: Maximum allowable complexity score
            parameter_stability_threshold: Threshold for parameter stability
        """
        self.base_lookback_period = base_lookback_period
        self.min_samples = min_samples
        self.max_complexity_score = max_complexity_score
        self.parameter_stability_threshold = parameter_stability_threshold
        
        # Core metrics to track
        self.key_metrics = ['sharpe_ratio', 'win_rate', 'profit_factor', 'max_drawdown', 'volatility']
        
        # Historical data for adaptation
        self.performance_history: List[PerformanceMetrics] = []
        self.regime_history: List[str] = []
        self.adjustment_history: List[Dict[str, float]] = []

    def detect_overfitting(self, 
                          in_sample_metrics: PerformanceMetrics,
                          out_sample_metrics: PerformanceMetrics,
                          market_regime: str,
                          model_parameters: Dict[str, Any]) -> Tuple[bool, Dict[str, float]]:
        """
        Main method to detect overfitting using multiple indicators.

        Args:
            in_sample_metrics: Performance metrics from training data
            out_sample_metrics: Performance metrics from test data
            market_regime: Current market regime
            model_parameters: Current model parameters

        Returns:
            Tuple of (is_overfitting: bool, detailed_scores: Dict[str, float])
        """
        try:
            # Get regime-specific thresholds
            thresholds = self._get_regime_thresholds(market_regime)
            
            # Calculate individual overfitting scores
            performance_score = self._calculate_performance_degradation(
                in_sample_metrics, out_sample_metrics, thresholds
            )
            
            stability_score = self._calculate_parameter_stability(
                model_parameters, thresholds
            )
            
            regime_score = self._calculate_regime_consistency(
                in_sample_metrics, market_regime
            )
            
            complexity_score = self._calculate_complexity_score(model_parameters)

            # Combine scores with regime-specific weights
            weights = self._get_regime_weights(market_regime)
            final_score = (
                weights['performance'] * performance_score +
                weights['stability'] * stability_score +
                weights['regime'] * regime_score +
                weights['complexity'] * complexity_score
            )

            detailed_scores = {
                'performance_score': performance_score,
                'stability_score': stability_score,
                'regime_score': regime_score,
                'complexity_score': complexity_score,
                'final_score': final_score,
                'threshold': thresholds['final']
            }

            # Update historical data
            self._update_history(in_sample_metrics, market_regime, final_score)

            return final_score > thresholds['final'], detailed_scores

        except Exception as e:
            logger.error(f"Error in overfitting detection: {e}")
            # Return conservative estimate in case of error
            return True, {'error': str(e)}

    def _get_regime_thresholds(self, market_regime: str) -> Dict[str, float]:
        """
        Get threshold values adjusted for the current market regime.
        """
        base_thresholds = {
            'performance': 0.3,
            'stability': 0.4,
            'regime': 0.5,
            'complexity': 0.7,
            'final': 0.6
        }

        multipliers = {
            MarketRegime.HIGH_VOLATILITY.value: {
                'performance': 1.2,
                'stability': 1.3,
                'regime': 1.1,
                'complexity': 1.0,
                'final': 1.2
            },
            MarketRegime.TRENDING.value: {
                'performance': 0.9,
                'stability': 1.1,
                'regime': 1.2,
                'complexity': 1.0,
                'final': 1.0
            },
            MarketRegime.RANGING.value: {
                'performance': 1.1,
                'stability': 0.9,
                'regime': 0.9,
                'complexity': 1.0,
                'final': 1.1
            }
        }

        regime_multiplier = multipliers.get(market_regime, {})
        return {
            k: v * regime_multiplier.get(k, 1.0)
            for k, v in base_thresholds.items()
        }

    def _get_regime_weights(self, market_regime: str) -> Dict[str, float]:
        """
        Get scoring weights adjusted for the current market regime.
        """
        base_weights = {
            'performance': 0.4,
            'stability': 0.3,
            'regime': 0.2,
            'complexity': 0.1
        }

        if market_regime == MarketRegime.HIGH_VOLATILITY.value:
            return {
                'performance': 0.3,
                'stability': 0.4,
                'regime': 0.2,
                'complexity': 0.1
            }
        elif market_regime == MarketRegime.TRENDING.value:
            return {
                'performance': 0.45,
                'stability': 0.25,
                'regime': 0.2,
                'complexity': 0.1
            }
        elif market_regime == MarketRegime.RANGING.value:
            return {
                'performance': 0.35,
                'stability': 0.35,
                'regime': 0.2,
                'complexity': 0.1
            }
        
        return base_weights

    def _calculate_performance_degradation(self,
                                         in_sample: PerformanceMetrics,
                                         out_sample: PerformanceMetrics,
                                         thresholds: Dict[str, float]) -> float:
        """
        Calculate performance degradation between in-sample and out-of-sample results.
        """
        degradation_scores = []
        
        for metric in self.key_metrics:
            in_sample_value = getattr(in_sample, metric)
            out_sample_value = getattr(out_sample, metric)
            
            if abs(in_sample_value) > 1e-6:  # Avoid division by zero
                degradation = abs(in_sample_value - out_sample_value) / abs(in_sample_value)
                degradation_scores.append(min(degradation, 1.0))
        
        return np.mean(degradation_scores) if degradation_scores else 1.0

    def _calculate_parameter_stability(self,
                                     current_parameters: Dict[str, Any],
                                     thresholds: Dict[str, float]) -> float:
        """
        Calculate parameter stability score based on historical changes.
        """
        if not self.adjustment_history:
            return 0.0

        stability_scores = []
        for param_name, current_value in current_parameters.items():
            historical_values = [
                h.get(param_name, current_value)
                for h in self.adjustment_history[-self.base_lookback_period:]
            ]
            
            if len(historical_values) >= self.min_samples:
                variation = np.std(historical_values) / (np.mean(historical_values) + 1e-6)
                stability_scores.append(min(variation, 1.0))

        return np.mean(stability_scores) if stability_scores else 1.0

    def _calculate_regime_consistency(self,
                                    metrics: PerformanceMetrics,
                                    current_regime: str) -> float:
        """
        Calculate consistency score across different market regimes.
        """
        if len(self.performance_history) < self.min_samples:
            return 0.0

        # Get historical performance for the current regime
        regime_performances = [
            p.sharpe_ratio
            for p, r in zip(self.performance_history, self.regime_history)
            if r == current_regime
        ]

        if len(regime_performances) >= self.min_samples:
            current_performance = metrics.sharpe_ratio
            regime_mean = np.mean(regime_performances)
            regime_std = np.std(regime_performances) + 1e-6
            
            # Calculate z-score of current performance
            z_score = abs(current_performance - regime_mean) / regime_std
            return min(z_score / 3.0, 1.0)  # Normalize to [0,1]
        
        return 0.5  # Default to moderate score if insufficient data

    def _calculate_complexity_score(self, model_parameters: Dict[str, Any]) -> float:
        """
        Calculate model complexity score based on parameters.
        """
        complexity_factors = []

        # Check number of features
        if 'n_features' in model_parameters:
            n_features = model_parameters['n_features']
            complexity_factors.append(min(n_features / 100, 1.0))

        # Check model depth
        if 'max_depth' in model_parameters:
            depth = model_parameters['max_depth']
            complexity_factors.append(min(depth / 10, 1.0))

        # Check number of parameters
        n_params = len(model_parameters)
        complexity_factors.append(min(n_params / 50, 1.0))

        return np.mean(complexity_factors) if complexity_factors else 0.5

    def adjust_model(self, 
                    model: Any,
                    overfitting_scores: Dict[str, float],
                    market_regime: str) -> Dict[str, Any]:
        """
        Adjust model parameters based on overfitting detection results.

        Args:
            model: The machine learning model to adjust
            overfitting_scores: Detailed scores from overfitting detection
            market_regime: Current market regime

        Returns:
            Dict containing the adjusted parameters
        """
        try:
            current_params = model.get_parameters()
            
            # Calculate adjustment factors
            adjustment_strength = min(overfitting_scores['final_score'], 0.5)
            
            # Adjust parameters based on specific scores
            adjusted_params = current_params.copy()
            
            if overfitting_scores['complexity_score'] > self.max_complexity_score:
                adjusted_params = self._reduce_complexity(adjusted_params, adjustment_strength)
            
            if overfitting_scores['stability_score'] > self.parameter_stability_threshold:
                adjusted_params = self._increase_regularization(adjusted_params, adjustment_strength)
            
            # Apply regime-specific adjustments
            adjusted_params = self._apply_regime_adjustments(adjusted_params, market_regime)
            
            # Update adjustment history
            self.adjustment_history.append(adjusted_params)
            
            return adjusted_params

        except Exception as e:
            logger.error(f"Error adjusting model: {e}")
            return current_params

    def _reduce_complexity(self, 
                          parameters: Dict[str, Any], 
                          adjustment_strength: float) -> Dict[str, Any]:
        """
        Reduce model complexity by adjusting relevant parameters.
        """
        adjusted = parameters.copy()
        
        # Adjust tree depth if present
        if 'max_depth' in adjusted:
            adjusted['max_depth'] = max(
                3,  # Minimum depth
                int(adjusted['max_depth'] * (1 - adjustment_strength))
            )
        
        # Adjust number of estimators if present
        if 'n_estimators' in adjusted:
            adjusted['n_estimators'] = max(
                50,  # Minimum estimators
                int(adjusted['n_estimators'] * (1 - adjustment_strength * 0.5))
            )
        
        # Adjust minimum samples per leaf if present
        if 'min_samples_leaf' in adjusted:
            adjusted['min_samples_leaf'] = max(
                1,
                int(adjusted['min_samples_leaf'] * (1 + adjustment_strength))
            )
        
        return adjusted

    def _increase_regularization(self, 
                               parameters: Dict[str, Any], 
                               adjustment_strength: float) -> Dict[str, Any]:
        """
        Increase model regularization to combat instability.
        """
        adjusted = parameters.copy()
        
        # Adjust L1 regularization if present
        if 'l1_ratio' in adjusted:
            adjusted['l1_ratio'] = min(
                1.0,
                adjusted['l1_ratio'] * (1 + adjustment_strength)
            )
        
        # Adjust L2 regularization if present
        if 'l2_ratio' in adjusted:
            adjusted['l2_ratio'] = min(
                1.0,
                adjusted['l2_ratio'] * (1 + adjustment_strength)
            )
        
        # Adjust dropout if present
        if 'dropout_rate' in adjusted:
            adjusted['dropout_rate'] = min(
                0.5,
                adjusted['dropout_rate'] * (1 + adjustment_strength)
            )
        
        return adjusted
    def _apply_regime_adjustments(self,
                                parameters: Dict[str, Any],
                                market_regime: str) -> Dict[str, Any]:
        """
        Apply market regime-specific parameter adjustments.
        """
        adjusted = parameters.copy()
        
        if market_regime == MarketRegime.TRENDING.value:
            # Favor longer-term patterns in trending markets
            if 'lookback_period' in adjusted:
                adjusted['lookback_period'] = int(adjusted['lookback_period'] * 1.2)
            if 'momentum_period' in adjusted:
                adjusted['momentum_period'] = int(adjusted['momentum_period'] * 1.3)
            if 'learning_rate' in adjusted:
                adjusted['learning_rate'] *= 0.8  # More conservative learning
                
        elif market_regime == MarketRegime.RANGING.value:
            # Favor faster adaptation in ranging markets
            if 'lookback_period' in adjusted:
                adjusted['lookback_period'] = int(adjusted['lookback_period'] * 0.8)
            if 'learning_rate' in adjusted:
                adjusted['learning_rate'] *= 1.2  # Faster learning
            if 'mean_reversion_threshold' in adjusted:
                adjusted['mean_reversion_threshold'] *= 0.9
                
        elif market_regime == MarketRegime.HIGH_VOLATILITY.value:
            # More conservative settings in high volatility
            if 'learning_rate' in adjusted:
                adjusted['learning_rate'] *= 0.7
            if 'position_size_multiplier' in adjusted:
                adjusted['position_size_multiplier'] *= 0.8
            if 'stop_loss_multiplier' in adjusted:
                adjusted['stop_loss_multiplier'] *= 1.2
                
        elif market_regime == MarketRegime.LOW_VOLATILITY.value:
            # More aggressive settings in low volatility
            if 'learning_rate' in adjusted:
                adjusted['learning_rate'] *= 1.1
            if 'position_size_multiplier' in adjusted:
                adjusted['position_size_multiplier'] *= 1.2
            if 'stop_loss_multiplier' in adjusted:
                adjusted['stop_loss_multiplier'] *= 0.9

        return adjusted

    def _update_history(self,
                       metrics: PerformanceMetrics,
                       market_regime: str,
                       overfitting_score: float) -> None:
        """
        Update historical tracking of performance and regimes.
        """
        self.performance_history.append(metrics)
        self.regime_history.append(market_regime)
        
        # Maintain fixed history length
        if len(self.performance_history) > self.base_lookback_period:
            self.performance_history.pop(0)
            self.regime_history.pop(0)

    def generate_report(self, 
                       in_sample_metrics: PerformanceMetrics,
                       out_sample_metrics: PerformanceMetrics,
                       market_regime: str,
                       overfitting_scores: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate a comprehensive report on overfitting analysis.

        Args:
            in_sample_metrics: Performance metrics from training data
            out_sample_metrics: Performance metrics from test data
            market_regime: Current market regime
            overfitting_scores: Detailed scores from overfitting detection

        Returns:
            Dict containing detailed analysis and recommendations
        """
        try:
            report = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'market_regime': market_regime,
                'performance_comparison': {
                    'in_sample': in_sample_metrics.to_dict(),
                    'out_sample': out_sample_metrics.to_dict(),
                    'degradation': {
                        metric: abs(getattr(in_sample_metrics, metric) - 
                                  getattr(out_sample_metrics, metric))
                        for metric in self.key_metrics
                    }
                },
                'overfitting_analysis': overfitting_scores,
                'regime_specific_thresholds': self._get_regime_thresholds(market_regime),
                'recommendations': self._generate_recommendations(overfitting_scores, market_regime)
            }

            # Add historical context
            if len(self.performance_history) >= self.min_samples:
                report['historical_context'] = {
                    'performance_trend': self._analyze_performance_trend(),
                    'regime_stability': self._analyze_regime_stability(),
                    'parameter_stability': self._analyze_parameter_stability()
                }

            return report

        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {'error': str(e)}

    def _generate_recommendations(self,
                                overfitting_scores: Dict[str, float],
                                market_regime: str) -> List[str]:
        """
        Generate specific recommendations based on overfitting analysis.
        """
        recommendations = []

        # Check complexity
        if overfitting_scores['complexity_score'] > self.max_complexity_score:
            recommendations.append(
                "Model complexity is high. Consider reducing the number of features "
                "or simplifying model architecture."
            )

        # Check stability
        if overfitting_scores['stability_score'] > self.parameter_stability_threshold:
            recommendations.append(
                "Parameter stability is low. Consider increasing regularization "
                "or implementing ensemble methods."
            )

        # Performance degradation
        if overfitting_scores['performance_score'] > 0.3:
            recommendations.append(
                "Significant performance degradation detected. Consider implementing "
                "more robust cross-validation techniques."
            )

        # Regime-specific recommendations
        if market_regime == MarketRegime.HIGH_VOLATILITY.value:
            recommendations.append(
                "In high volatility regime: Consider implementing adaptive position "
                "sizing and stronger risk controls."
            )
        elif market_regime == MarketRegime.TRENDING.value:
            recommendations.append(
                "In trending regime: Consider extending lookback periods and "
                "momentum indicators."
            )

        return recommendations

    def _analyze_performance_trend(self) -> Dict[str, float]:
        """
        Analyze historical performance trends.
        """
        if len(self.performance_history) < self.min_samples:
            return {}

        recent_window = self.performance_history[-20:]
        older_window = self.performance_history[-40:-20]

        trend_analysis = {}
        for metric in self.key_metrics:
            recent_avg = np.mean([getattr(p, metric) for p in recent_window])
            older_avg = np.mean([getattr(p, metric) for p in older_window])
            
            if abs(older_avg) > 1e-6:
                trend_analysis[f'{metric}_trend'] = (recent_avg - older_avg) / abs(older_avg)
            else:
                trend_analysis[f'{metric}_trend'] = 0.0

        return trend_analysis

    def _analyze_regime_stability(self) -> Dict[str, float]:
        """
        Analyze stability of market regime predictions.
        """
        if len(self.regime_history) < self.min_samples:
            return {}

        # Calculate regime transition frequency
        transitions = sum(1 for i in range(1, len(self.regime_history))
                        if self.regime_history[i] != self.regime_history[i-1])
        
        transition_rate = transitions / len(self.regime_history)

        # Calculate regime distribution
        regime_counts = pd.Series(self.regime_history).value_counts()
        regime_distribution = (regime_counts / len(self.regime_history)).to_dict()

        return {
            'transition_rate': transition_rate,
            'regime_distribution': regime_distribution
        }

    def _analyze_parameter_stability(self) -> Dict[str, float]:
        """
        Analyze stability of model parameters over time.
        """
        if len(self.adjustment_history) < self.min_samples:
            return {}

        stability_metrics = {}
        for param_name in self.adjustment_history[0].keys():
            param_values = [adj[param_name] for adj in self.adjustment_history 
                          if param_name in adj]
            
            if param_values:
                stability_metrics[param_name] = {
                    'mean': float(np.mean(param_values)),
                    'std': float(np.std(param_values)),
                    'cv': float(np.std(param_values) / (np.mean(param_values) + 1e-6))
                }

        return stability_metrics

    def save_state(self, filepath: str) -> None:
        """
        Save the current state of the overfitting controller.
        """
        state = {
            'base_lookback_period': self.base_lookback_period,
            'min_samples': self.min_samples,
            'max_complexity_score': self.max_complexity_score,
            'parameter_stability_threshold': self.parameter_stability_threshold,
            'performance_history': [p.to_dict() for p in self.performance_history],
            'regime_history': self.regime_history,
            'adjustment_history': self.adjustment_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f)

    def load_state(self, filepath: str) -> None:
        """
        Load a previously saved state.
        """
        with open(filepath, 'r') as f:
            state = json.load(f)
            
        self.base_lookback_period = state['base_lookback_period']
        self.min_samples = state['min_samples']
        self.max_complexity_score = state['max_complexity_score']
        self.parameter_stability_threshold = state['parameter_stability_threshold']
        
        self.performance_history = [
            PerformanceMetrics(**p) for p in state['performance_history']
        ]
        self.regime_history = state['regime_history']
        self.adjustment_history = state['adjustment_history']

class TradingSystem:
    def __init__(self, db_path: str, account_balance: float, max_risk_per_trade: float):
        self.db_path = db_path
        self.account_balance = account_balance
        self.max_risk_per_trade = max_risk_per_trade
        self.signal_generator = SignalGenerator(config.MARKET_DATA_DB)
        self.backtest = None
        self.performance_metrics = None
        self.market_condition_stats = None

    def load_historical_data(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM '{ticker}' WHERE date BETWEEN '{start_date}' AND '{end_date}'"
            df = pd.read_sql_query(query, conn)
            conn.close()
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading historical data: {e}")
            raise

    def run_backtest(self, ticker: str, start_date: str, end_date: str, n_splits: int = 5) -> pd.DataFrame:
        historical_data = self.load_historical_data(ticker, start_date, end_date)
        self.backtest = EnhancedBacktest(historical_data, initial_capital=self.account_balance)
        backtest_results = self.backtest.run_backtest(self.signal_generator, n_splits=n_splits)
        self.performance_metrics = self.backtest.calculate_performance_metrics()
        self.market_condition_stats = self.calculate_market_condition_statistics(backtest_results)
        return backtest_results

    def calculate_market_condition_statistics(self, backtest_results: pd.DataFrame) -> pd.DataFrame:
        market_condition_stats = backtest_results.groupby('Market_Condition').agg({
            'Strategy_Returns': ['mean', 'std', 'count'],
            'Drawdown': 'max'
        })
        market_condition_stats.columns = ['Avg_Return', 'Std_Dev', 'Count', 'Max_Drawdown']
        market_condition_stats['Sharpe_Ratio'] = (
            market_condition_stats['Avg_Return'] / market_condition_stats['Std_Dev'] * np.sqrt(252)
        )
        return market_condition_stats

    def get_current_market_condition(self, data: pd.DataFrame) -> str:
        return self.backtest.classify_market_conditions(data).iloc[-1]

    def calculate_confidence_score(self, current_condition: str) -> float:
        return self.backtest.calculate_confidence_score(current_condition)

    def run_monte_carlo(self, ticker: str, num_simulations: int = 1000, simulation_length: int = 252) -> pd.DataFrame:
        if self.backtest is None:
            raise ValueError("Backtest hasn't been run yet. Call run_backtest() first.")
        
        monte_carlo_results = self.backtest.run_monte_carlo(
            self.signal_generator, num_simulations, simulation_length
        )
        return monte_carlo_results

    def generate_trading_signal(self, current_data: pd.DataFrame) -> int:
        signal = self.signal_generator.generate_signal(current_data)
        current_condition = self.get_current_market_condition(current_data)
        confidence_score = self.calculate_confidence_score(current_condition)
        
        # Adjust signal based on confidence score
        if confidence_score < 0.3:
            return 0  # No trade due to low confidence
        elif confidence_score < 0.7:
            return signal // 2  # Reduce position size for medium confidence
        else:
            return signal  # Full signal for high confidence

    def calculate_position_size(self, signal: int, current_price: float, stop_loss: float) -> float:
        risk_amount = self.account_balance * self.max_risk_per_trade
        risk_per_share = abs(current_price - stop_loss)
        position_size = risk_amount / risk_per_share
        return position_size * abs(signal)

    def execute_trade(self, ticker: str, signal: int, current_price: float, stop_loss: float) -> Dict[str, float]:
        position_size = self.calculate_position_size(signal, current_price, stop_loss)
        trade_value = position_size * current_price
        
        if trade_value > self.account_balance:
            logging.warning("Insufficient funds to execute the trade.")
            return None
        
        self.account_balance -= trade_value
        
        return {
            'ticker': ticker,
            'signal': signal,
            'position_size': position_size,
            'entry_price': current_price,
            'stop_loss': stop_loss
        }

    def update_trade(self, trade: Dict[str, float], current_price: float) -> Tuple[Dict[str, float], float]:
        if (trade['signal'] > 0 and current_price <= trade['stop_loss']) or \
           (trade['signal'] < 0 and current_price >= trade['stop_loss']):
            # Close the trade
            profit_loss = (current_price - trade['entry_price']) * trade['position_size'] * trade['signal']
            self.account_balance += (trade['position_size'] * current_price)
            return None, profit_loss
        return trade, 0

    def run_trading_session(self, ticker: str, start_date: str, end_date: str) -> List[Dict[str, float]]:
        data = self.load_historical_data(ticker, start_date, end_date)
        trades = []
        current_trade = None
        
        for date, row in data.iterrows():
            if current_trade:
                current_trade, profit_loss = self.update_trade(current_trade, row['close'])
                if not current_trade:
                    trades.append({'exit_date': date, 'profit_loss': profit_loss})
            
            if not current_trade:
                signal = self.generate_trading_signal(data.loc[:date])
                if signal != 0:
                    stop_loss = row['close'] * (0.95 if signal > 0 else 1.05)  # 5% stop loss
                    trade = self.execute_trade(ticker, signal, row['close'], stop_loss)
                    if trade:
                        current_trade = trade
                        trades.append({'entry_date': date, **trade})

        return trades

    def get_performance_summary(self) -> Dict[str, float]:
        if not self.performance_metrics:
            raise ValueError("Backtest hasn't been run yet. Call run_backtest() first.")
        return self.performance_metrics

    def get_market_condition_summary(self) -> pd.DataFrame:
        if self.market_condition_stats is None:
            raise ValueError("Backtest hasn't been run yet. Call run_backtest() first.")
        return self.market_condition_stats

# Main execution
if __name__ == "__main__":
    try:
        db_path = config.DB_PATH
        model_path = config.MODEL_PATH
        signal_generator = SignalGenerator(db_path, model_path)

        ticker = "AAPL"  # Example ticker
        start_date = "2022-01-01"
        end_date = "2023-01-01"

        backtest_results, performance_metrics, recent_performance, wfo_results = signal_generator.run(ticker, start_date, end_date)

        logger.info(f"Performance Metrics: {performance_metrics}")
        logger.info(f"Recent Performance: {recent_performance}")

        market_condition_stats = signal_generator.update_market_condition_statistics(backtest_results)
        logger.info(f"Market Condition Statistics:\n{market_condition_stats}")

    except Exception as e:
        logger.error(f"Error in main execution: {e}")