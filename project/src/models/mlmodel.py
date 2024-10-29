# Standard Library Imports
import logging
import json
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

# Third-Party Imports
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib
from ta.trend import EMAIndicator, MACD, ADXIndicator, SMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Local Imports
from ..core.types import MarketRegime
from ..core.performance_metrics import PerformanceMetrics
from .overfittingcontrol import OverfittingController

# Set up logging
logger = logging.getLogger(__name__)

class MachineLearningModel:
    """
    Machine learning model for market prediction with overfitting protection.
    """
    
    def __init__(self, lookback_period: int = 300):
        """
        Initialize the machine learning model with parameters from Durandal Trading Strategy Document.

        Args:
            lookback_period: Number of periods for historical analysis
        """
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()
        # Initialize preprocessing pipeline
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        # Initialize model with document-specified parameters
        self.model = RandomForestClassifier(
            n_estimators=100,  # Can be tuned during validation
            max_depth=5,       # Prevents overfitting
            min_samples_leaf=20,  # Ensures stable splits
            random_state=42
        )

        # Feature columns from document section 2.1
        self.feature_columns = [
            # Moving Averages
            'SMA_50', 'SMA_200',
            
            # RSI Components
            'RSI', 'RSI_Z',
            
            # MACD Components
            'MACD', 'MACD_signal', 'MACD_diff', 'MACD_Z',
            
            # Trend and Volatility
            'ADX', 'ATR',
            
            # Volume Indicators
            'Volume_Ratio', 'volume',
            'OBV', 'VWAP',
            
            # Momentum and Oscillators
            'Momentum',
            'Stoch_K', 'Stoch_D',
            'BB_width'
        ]

        # Initialize overfitting controller from document
        self.overfitting_controller = OverfittingController(
            base_lookback_period=252,  # One year of trading data
            min_samples=100,           # Minimum samples for reliable statistics
            max_complexity_score=0.8,  # Prevents excessive model complexity
            parameter_stability_threshold=0.3  # Maximum allowed parameter instability
        )

        # Market condition parameters from document section 2.2
        self.market_params = {
            'trend_threshold': 0.02,        # For sideways market detection
            'volatility_percentile': 0.8,   # 80th percentile for high volatility
            'volume_threshold': 1.5,        # 50% above average for high volume
            'atr_window': 14,              # For volatility calculation
            'momentum_window': 20          # For momentum calculation
        }

        # Signal generation thresholds from document
        self.signal_thresholds = {
            'base_bull': 0.7,
            'base_bear': 0.3,
            'alpha': 0.1,  # Bullish threshold reduction factor
            'beta': 0.3,   # Bearish threshold increase factor
            'min_adx': 25,  # Minimum ADX for trend confirmation
            'rsi_overbought': 70,
            'rsi_oversold': 30
        }

        # Performance tracking
        self.current_market_regime = "normal"
        self.performance_history = []
        
        # Tracking windows from document
        self.tracking_windows = {
            'short_term': 20,   # 20-day window for recent performance
            'medium_term': 60,  # 60-day window for medium-term trends
            'long_term': 252    # 252-day window for long-term analysis
        }

        # Store indicators for threshold adjustments
        self.indicator_thresholds = {}  # Will be updated during training

        logger.info("MachineLearningModel initialized with Durandal Trading Strategy parameters")
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required technical indicators from Durandal Trading Strategy Document.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all calculated indicators
        """
        try:
            df = data.copy()

            # Simple Moving Averages
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()

            # RSI and normalized RSI (RSI_Z)
            rsi = RSIIndicator(close=df['close'], window=14)
            df['RSI'] = rsi.rsi()
            # Calculate RSI Z-score over 50-day window
            df['RSI_Z'] = (df['RSI'] - df['RSI'].rolling(window=50).mean()) / df['RSI'].rolling(window=50).std()

            # MACD Components
            macd = MACD(
                close=df['close'],
                window_slow=26,
                window_fast=12,
                window_sign=9
            )
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            # Calculate MACD Z-score over 50-day window
            df['MACD_Z'] = (df['MACD_diff'] - df['MACD_diff'].rolling(window=50).mean()) / \
                            df['MACD_diff'].rolling(window=50).std()

            # ADX for trend strength
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['ADX'] = adx.adx()

            # ATR for volatility
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['ATR'] = atr.average_true_range()

            # Volume indicators
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
            
            # On-Balance Volume
            obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
            df['OBV'] = obv.on_balance_volume()

            # VWAP
            vwap = VolumeWeightedAveragePrice(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            )
            df['VWAP'] = vwap.volume_weighted_average_price()

            # Momentum (20-day price change)
            df['Momentum'] = df['close'].pct_change(periods=20)

            # Stochastic Oscillator
            stoch = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14,
                smooth_window=3
            )
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()

            # Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']

            # Forward fill any NaN values caused by lookback windows
            df = df.fillna(method='ffill')

            # Verify all required indicators are calculated
            required_indicators = [
                'SMA_50', 'SMA_200',
                'RSI', 'RSI_Z',
                'MACD', 'MACD_signal', 'MACD_diff', 'MACD_Z',
                'ADX', 'ATR',
                'Volume_Ratio', 
                'OBV', 'VWAP',
                'Momentum',
                'Stoch_K', 'Stoch_D',
                'BB_width'
            ]

            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            if missing_indicators:
                raise ValueError(f"Failed to calculate indicators: {missing_indicators}")

            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise

    def _get_base_thresholds(self) -> Dict[str, float]:
        """Get base signal thresholds."""
        return {
            'bull': 0.7,  # Base bullish signal threshold
            'bear': 0.3,  # Base bearish signal threshold
        }

    def _adjust_thresholds(self, 
                        base_thresholds: Dict[str, float],
                        market_condition: str,
                        data: pd.DataFrame) -> Dict[str, float]:
        """
        Adjust thresholds based on market conditions as per document section on
        Market Condition Adjustments.
        """
        adjusted = base_thresholds.copy()
        
        # Alpha and beta from document
        alpha = 0.1  # Factor for bullish threshold reduction
        beta = 0.3   # Factor for bearish threshold increase
        
        if "Uptrend" in market_condition:
            # T_bull_up = T_bull_base × (1 - α)
            adjusted['bull'] *= (1 - alpha)
            # T_bear_up = T_bear_base × (1 + β)
            adjusted['bear'] *= (1 + beta)
            
        elif "Downtrend" in market_condition:
            # T_bull_down = T_bull_base × (1 + β)
            adjusted['bull'] *= (1 + beta)
            # T_bear_down = T_bear_base × (1 - α)
            adjusted['bear'] *= (1 - alpha)
            
        # Trend strength adjustment
        trend_factor = self._calculate_trend_strength(data)
        if "Strong" in market_condition:
            for key in adjusted:
                adjusted[key] *= (1 + trend_factor * 0.2)
                
        return adjusted

    def _calculate_trend_strength(self, data: pd.DataFrame) -> float:
        """Calculate trend strength as per document."""
        sma_diff = (data['SMA_50'].iloc[-1] - data['SMA_200'].iloc[-1]) / data['SMA_200'].iloc[-1]
        return abs(sma_diff)

    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for model training with enhanced validation.
        
        Args:
            data: DataFrame with market data and indicators
            
        Returns:
            Tuple of (features, target)
            
        Raises:
            ValueError: If data preparation fails
        """
        try:
            # Validate input data
            if len(data) < self.lookback_period:
                raise ValueError(
                    f"Insufficient data points: {len(data)} < {self.lookback_period}"
                )
            
            # Verify all feature columns exist
            missing_features = [col for col in self.feature_columns if col not in data.columns]
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
                
            # Extract features
            X = data[self.feature_columns].copy()
            
            # Calculate future returns for target
            future_returns = data['close'].pct_change().shift(-1)
            y = np.where(future_returns > 0, 1, 0)
            
            # Remove any rows with NaN values
            valid_mask = ~(X.isna().any(axis=1) | pd.isna(y))
            X = X[valid_mask]
            y = y[valid_mask]
            
            # Ensure sufficient data remains after cleaning
            if len(X) < 100:  # Minimum required samples
                raise ValueError(
                    f"Insufficient valid samples after cleaning: {len(X)} < 100"
                )
                
            # Convert to numpy arrays
            X = X.values
            y = y[:(len(X))]  # Align target with features
            
            # Add additional validation
            if X.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Feature/target shape mismatch: {X.shape[0]} != {y.shape[0]}"
                )
                
            if np.isnan(X).any() or np.isnan(y).any():
                raise ValueError("NaN values found after preparation")
                
            logger.info(
                f"Prepared {X.shape[0]} samples with {X.shape[1]} features"
            )
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {str(e)}")
            raise

    def _calculate_performance_metrics(self, returns, predictions, threshold=0):
        """
        Calculate comprehensive performance metrics for the model.
        
        Args:
            returns (np.ndarray or pd.Series): Actual returns
            predictions (np.ndarray or pd.Series): Model predictions
            threshold (float): Threshold for binary classification
            
        Returns:
            dict: Dictionary containing various performance metrics
            
        Raises:
            Exception: If there's an error in calculation
        """
        try:
            # Ensure minimum length and align data
            min_length = min(len(returns), len(predictions))
            
            # Handle different input types
            if isinstance(returns, pd.Series):
                returns = returns[-min_length:].values
            else:
                returns = returns[-min_length:]
                
            if isinstance(predictions, pd.Series):
                predictions = predictions[-min_length:].values
            else:
                predictions = predictions[-min_length:]

            # Calculate classification metrics
            actual_signals = returns > threshold
            predicted_signals = predictions > threshold
            
            accuracy = accuracy_score(actual_signals, predicted_signals)
            precision = precision_score(actual_signals, predicted_signals)
            recall = recall_score(actual_signals, predicted_signals)
            f1 = f1_score(actual_signals, predicted_signals)
            
            # Calculate trading metrics
            cumulative_returns = np.cumsum(returns)
            max_drawdown = self._calculate_max_drawdown(cumulative_returns)
            
            # Calculate volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252)
            
            # Calculate Sharpe Ratio (annualized)
            excess_returns = returns - self.risk_free_rate/252  # Daily risk-free rate
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
            
            # Calculate win rate
            profitable_trades = np.sum(returns > 0)
            total_trades = len(returns)
            win_rate = profitable_trades / total_trades if total_trades > 0 else 0
            
            # Calculate profit factor
            gross_profits = np.sum(returns[returns > 0])
            gross_losses = abs(np.sum(returns[returns < 0]))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
            
            # Calculate average return
            avg_return = np.mean(returns)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'max_drawdown': max_drawdown,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'avg_return': avg_return,
                'total_trades': total_trades,
                'profitable_trades': profitable_trades
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise


    def _calculate_max_drawdown(self, cumulative_returns):
        """
        Calculate the maximum drawdown from peak to trough.
        
        Args:
            cumulative_returns (np.ndarray): Array of cumulative returns
            
        Returns:
            float: Maximum drawdown as a positive percentage
        """
        rolling_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        return abs(np.min(drawdowns))

    def train(self, data: pd.DataFrame) -> None:
        """
        Train the model with enhanced validation and error handling.
        
        Args:
            data: Training data DataFrame
            
        Raises:
            ValueError: If training fails
        """
        try:
            logger.info("Starting model training...")
            
            # Split data into training and validation sets
            train_size = int(len(data) * 0.7)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:]
            
            # Prepare features with validation
            X_train, y_train = self.prepare_features(train_data)
            X_val, y_val = self.prepare_features(val_data)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train model with validation
            logger.info("Training model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Get predictions for overfitting check
            train_predictions = self.model.predict(X_train_scaled)
            val_predictions = self.model.predict(X_val_scaled)
            
            # Calculate performance metrics
            train_metrics = self._calculate_performance_metrics(
                train_data['close'].pct_change().dropna(),
                train_predictions,
                y_train
            )
            val_metrics = self._calculate_performance_metrics(
                val_data['close'].pct_change().dropna(),
                val_predictions,
                y_val
            )
            
            # Check for overfitting
            is_overfitting = self._detect_overfitting(
                train_metrics, val_metrics
            )
            
            if is_overfitting:
                logger.warning("Overfitting detected, adjusting model parameters...")
                self._adjust_for_overfitting()
                
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise

    def predict(self, data: pd.DataFrame) -> float:
        """
        Generate predictions with the trained model.
        
        Args:
            data: Market data with indicators
            
        Returns:
            Prediction probability
        """
        try:
            X, _ = self.prepare_features(data)
            if len(X) == 0:
                raise ValueError("No valid features after preparation")
                
            X_transformed = self.pipeline.transform(X.iloc[-1].values.reshape(1, -1))
            return self.model.predict_proba(X_transformed)[0][1]
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise

    def update(self, new_data: pd.DataFrame, market_regime: str) -> None:
        """
        Update the model with new data.
        
        Args:
            new_data: New market data
            market_regime: Current market regime
        """
        try:
            self.current_market_regime = market_regime
            X_new, y_new = self.prepare_features(new_data)
            
            if len(X_new) == 0:
                raise ValueError("No valid data for update")
                
            X_transformed = self.pipeline.transform(X_new)
            
            # Get current predictions
            predictions = self.model.predict(X_transformed)
            returns = new_data['close'].pct_change().dropna()
            
            # Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics(
                predictions, y_new, returns
            )
            
            # Store performance history
            self.performance_history.append(performance_metrics)
            
            # Update the model
            self.model.fit(X_transformed, y_new)
            
            logger.info("Model successfully updated with new data")
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        return dict(zip(self.feature_columns, self.model.feature_importances_))

    def get_parameters(self) -> Dict[str, Any]:
        """Get current model parameters."""
        return {
            'n_features': len(self.feature_columns),
            'max_depth': self.model.max_depth,
            'n_estimators': self.model.n_estimators,
            'min_samples_leaf': self.model.min_samples_leaf
        }

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """Set model parameters."""
        self.model.set_params(**parameters)

    def _save_training_report(self, report: Dict[str, Any]) -> None:
        """Save training report to file."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"training_reports/model_report_{timestamp}.json"
            
            os.makedirs("training_reports", exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
                
            logger.info(f"Training report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving training report: {e}")

    def save_model(self, path: str) -> None:
        """Save model state and components."""
        try:
            model_state = {
                'model': self.model,
                'pipeline': self.pipeline,
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
        """Load model state and components."""
        try:
            # Load model state
            model_state = joblib.load(f"{path}_model.joblib")
            
            self.model = model_state['model']
            self.pipeline = model_state['pipeline']
            self.feature_columns = model_state['feature_columns']
            self.lookback_period = model_state['lookback_period']
            self.current_market_regime = model_state['current_market_regime']
            
            # Load overfitting controller state
            self.overfitting_controller.load_state(f"{path}_overfitting_controller.json")
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise