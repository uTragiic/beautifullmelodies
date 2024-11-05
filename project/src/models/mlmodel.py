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
from sklearn.utils.validation import check_is_fitted

# Local Imports
from ..core.types import MarketRegime
from ..core.performance_metrics import PerformanceMetrics
from .overfittingcontrol import OverfittingController
from ..indicators.calculator import IndicatorCalculator

# Set up logging
logger = logging.getLogger(__name__)

class MachineLearningModel:
    """
    Machine learning model for market prediction with overfitting protection.
    """
    
    def __init__(self, lookback_period: int = 100):
        """
        Initialize the machine learning model with parameters from Durandal Trading Strategy Document.

        Args:
            lookback_period: Number of periods for historical analysis
        """
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()
        self.risk_free_rate = 0.02
        model: Optional[RandomForestClassifier] = None,
        pipeline: Optional[Pipeline] = None

        # Initialize preprocessing pipeline
    # Initialize or assign the model
        if model is not None:
            self.model = model
        else:
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=20,
                random_state=42
            )

        # Initialize or assign the pipeline
        if pipeline is not None:
            self.pipeline = pipeline
        else:
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

        # Initialize IndicatorCalculator
        self.indicator_calculator = IndicatorCalculator()
        
        logger.info("MachineLearningModel initialized with Durandal Trading Strategy parameters")

    def _get_base_thresholds(self) -> Dict[str, float]:
        """Get base signal thresholds."""
        return {
            'bull': 0.7,  # Base bullish signal threshold
            'bear': 0.3,  # Base bearish signal threshold
        }

    def _is_pipeline_fitted(self) -> bool:
        """
        Check if the pipeline has been fitted.

        Returns:
            True if the pipeline is fitted, False otherwise.
        """
        try:
            # Check if each step in the pipeline is fitted
            for name, estimator in self.pipeline.named_steps.items():
                check_is_fitted(estimator)
            return True
        except Exception as e:
            logger.error(f"Pipeline not fitted: {str(e)}")
            return False

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
    def _calculate_performance_metrics(self, predictions: np.ndarray, y: np.ndarray, returns: np.ndarray) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for the model with proper error handling.
        
        Args:
            predictions: Model predictions
            y: Actual target values 
            returns: Actual returns
            
        Returns:
            PerformanceMetrics object containing calculated metrics
        """
        try:
            # Ensure arrays are the same length
            min_length = min(len(returns), len(predictions), len(y))
            returns = returns[-min_length:]
            predictions = predictions[-min_length:]
            y = y[-min_length:]

            # Calculate strategy returns
            strategy_returns = returns * np.where(predictions > 0.5, 1, -1)

            # Calculate cumulative returns safely
            cumulative_returns = np.cumprod(1 + strategy_returns)

            # Calculate winning trades and win rate
            winning_trades = np.sum(strategy_returns > 0)
            total_trades = len(strategy_returns[strategy_returns != 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Calculate profit factor safely
            gross_profits = np.sum(strategy_returns[strategy_returns > 0])
            gross_losses = abs(np.sum(strategy_returns[strategy_returns < 0]))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')

            # Calculate volatility and Sharpe ratio safely
            volatility = np.std(strategy_returns) * np.sqrt(252)
            if volatility == 0:  # Handle zero volatility case
                sharpe_ratio = 0.0
            else:
                sharpe_ratio = np.mean(strategy_returns) / volatility * np.sqrt(252)

            # Calculate max drawdown safely
            rolling_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0

            return PerformanceMetrics(
                sharpe_ratio=float(sharpe_ratio),
                win_rate=float(win_rate),
                profit_factor=float(profit_factor),
                max_drawdown=float(max_drawdown),
                volatility=float(volatility)
            )

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            # Return default metrics instead of raising
            return PerformanceMetrics(
                sharpe_ratio=0.0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                volatility=0.0
            )
        
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
            excess_returns = strategy_returns - self.risk_free_rate/252
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
            logger.error(f"Error calculating simulation metrics: {str(e)}")
            raise
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
            X = X.loc[:, X.apply(pd.Series.nunique) != 1]
            self.feature_columns = X.columns.tolist()
            
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

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all required technical indicators with adjusted minimum data requirements.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with calculated indicators

        Raises:
            ValueError: If there is insufficient data
        """
        try:
            df = data.copy()

            # Determine the minimum required length based on largest window size
            window_sizes = [50, 14, 26, 20]  # Reduced from previous sizes
            min_required_length = max(window_sizes)

            if len(df) < min_required_length:
                raise ValueError(
                    f"Insufficient data: {len(df)} rows; at least {min_required_length} required."
                )

            # Moving Averages
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=50).mean()  # Temporarily use SMA_50 instead of 200

            # RSI and normalized RSI
            rsi = RSIIndicator(close=df['close'], window=14)
            df['RSI'] = rsi.rsi()
            df['RSI_Z'] = (
                df['RSI'] - df['RSI'].rolling(window=20).mean()
            ) / df['RSI'].rolling(window=20).std()

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
            df['MACD_Z'] = (
                df['MACD_diff'] - df['MACD_diff'].rolling(window=20).mean()
            ) / df['MACD_diff'].rolling(window=20).std()

            # ADX
            adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['ADX'] = adx.adx()

            # ATR
            atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
            df['ATR'] = atr.average_true_range()

            # Volume indicators
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']

            # On-Balance Volume
            obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
            df['OBV'] = obv.on_balance_volume()

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

            # Momentum
            df['Momentum'] = df['close'].pct_change(periods=20)

            # Forward fill any NaN values
            df = df.fillna(method='ffill').fillna(method='bfill')

            # Verify all required indicators are present
            required_indicators = [
                'SMA_50', 'RSI', 'RSI_Z', 'MACD', 'MACD_signal', 'MACD_diff', 
                'MACD_Z', 'ADX', 'ATR', 'Volume_Ratio', 'OBV', 'Stoch_K', 
                'Stoch_D', 'BB_width', 'Momentum'
            ]

            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            if missing_indicators:
                raise ValueError(f"Failed to calculate indicators: {missing_indicators}")

            return df

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise  
    def train(self, data: pd.DataFrame) -> None:
        try:
            logger.info("Starting model training...")
            
            # Calculate indicators using IndicatorCalculator
            data = self.indicator_calculator.calculate_indicators(data)
            
            # Split data into training and validation sets
            train_size = int(len(data) * 0.7)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:]
            
            # Prepare features
            X_train, y_train = self.prepare_features(train_data)
            X_val, y_val = self.prepare_features(val_data)
            
            # Fit the pipeline on training data and transform
            X_train_scaled = self.pipeline.fit_transform(X_train)
            X_val_scaled = self.pipeline.transform(X_val)
            
            # Train model
            logger.info("Training model...")
            self.model.fit(X_train_scaled, y_train)
            
            # Get predictions
            train_predictions = self.model.predict_proba(X_train_scaled)[:, 1]
            val_predictions = self.model.predict_proba(X_val_scaled)[:, 1]
            
            # Calculate returns
            train_returns = train_data['close'].pct_change().dropna().values
            val_returns = val_data['close'].pct_change().dropna().values
            
            # Ensure alignment of returns and predictions
            train_returns = train_returns[-len(train_predictions):]
            val_returns = val_returns[-len(val_predictions):]
            
            # Calculate performance metrics
            in_sample_metrics = self._calculate_performance_metrics(
                train_predictions,
                y_train,
                train_returns
            )
            out_sample_metrics = self._calculate_performance_metrics(
                val_predictions,
                y_val,
                val_returns
            )
            
            # Get model parameters
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
                logger.warning("Overfitting detected, adjusting model parameters...")
                adjusted_params = self.overfitting_controller.adjust_model(
                    model=self.model,
                    overfitting_scores=overfitting_scores,
                    market_regime=self.current_market_regime
                )
                
                self.model.set_params(**adjusted_params)
                self.model.fit(X_train_scaled, y_train)
                
                # Recalculate metrics after adjustment
                train_predictions = self.model.predict_proba(X_train_scaled)[:, 1]
                val_predictions = self.model.predict_proba(X_val_scaled)[:, 1]
                
                # Re-align returns
                train_returns = train_data['close'].pct_change().dropna().values
                val_returns = val_data['close'].pct_change().dropna().values
                train_returns = train_returns[-len(train_predictions):]
                val_returns = val_returns[-len(val_predictions):]
                
                in_sample_metrics = self._calculate_performance_metrics(
                    train_predictions,
                    y_train,
                    train_returns
                )
                out_sample_metrics = self._calculate_performance_metrics(
                    val_predictions,
                    y_val,
                    val_returns
                )

            # Generate report
            report = self.overfitting_controller.generate_report(
                in_sample_metrics=in_sample_metrics,
                out_sample_metrics=out_sample_metrics,
                market_regime=self.current_market_regime,
                overfitting_scores=overfitting_scores
            )
            
            # Save report
            self._save_training_report(report)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    
    def predict(self, data: pd.DataFrame) -> float:
        """
        Generate predictions with the trained model.
        
        Args:
            data: Market data
        
        Returns:
            Prediction probability
        """
        try:
            # Calculate indicators using IndicatorCalculator
            data = self.indicator_calculator.calculate_indicators(data)
            
            X, _ = self.prepare_features(data)
            if len(X) == 0:
                raise ValueError("No valid features after preparation")
            
            # Check if the pipeline is fitted
            if not self._is_pipeline_fitted():
                raise ValueError("Pipeline is not fitted. Cannot transform data.")
                
            # Use the fitted pipeline to transform the data
            X_transformed = self.pipeline.transform(X[-1].reshape(1, -1))
            prediction = self.model.predict_proba(X_transformed)[0][1]
            return prediction
            
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

            # Calculate indicators using IndicatorCalculator
            new_data = self.indicator_calculator.calculate_indicators(new_data)

            X_new, y_new = self.prepare_features(new_data)

            if len(X_new) == 0:
                raise ValueError("No valid data for update")

            # Check if the pipeline is fitted
            if not self._is_pipeline_fitted():
                logger.info("Pipeline not fitted. Fitting pipeline with new data...")
                # Fit the pipeline with new data
                X_new = self.pipeline.fit_transform(X_new)
            else:
                # Transform new data using the fitted pipeline
                X_new = self.pipeline.transform(X_new)

            # Get current predictions
            predictions = self.model.predict_proba(X_new)[:, 1]
            returns = new_data['close'].pct_change().dropna().values

            # Ensure alignment of returns and predictions
            returns = returns[-len(predictions):]

            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(
                predictions, y_new, returns
            )

            # Store performance history
            self.performance_history.append(performance_metrics)

            # Update the model with new data
            self.model.fit(X_new, y_new)

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
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on model predictions.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            Series of trading signals (-1, 0, 1)
        """
        try:
            # Get prediction probabilities
            probabilities = []
            
            # Process data in chunks to generate signals
            for i in range(len(data)):
                if i < self.lookback_period:
                    probabilities.append(0.5)  # Default to neutral for insufficient data
                    continue
                    
                # Get data slice for prediction
                data_slice = data.iloc[max(0, i-self.lookback_period):i+1]
                
                # Get prediction probability
                prob = self.predict(data_slice)
                probabilities.append(prob)
                
            # Convert probabilities to signals
            signal_threshold = 0.75  # High conviction threshold
            signals = pd.Series(index=data.index, dtype=float)
            
            for i, prob in enumerate(probabilities):
                if prob > signal_threshold:
                    signals.iloc[i] = 1  # Long signal
                elif prob < (1 - signal_threshold):
                    signals.iloc[i] = -1  # Short signal
                else:
                    signals.iloc[i] = 0  # No signal
                    
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            # Return neutral signals in case of error
            return pd.Series(0, index=data.index)

    def load_model(self, path: str) -> None:
        """Load model state and components."""
        try:
            # Load model state
            model_state = joblib.load(f"{path}_model.joblib")
            
            self.model = model_state['model']
            self.pipeline = model_state['pipeline']  # Load the fitted pipeline
            self.feature_columns = model_state['feature_columns']
            self.lookback_period = model_state['lookback_period']
            self.current_market_regime = model_state['current_market_regime']
            
            # Load overfitting controller state
            self.overfitting_controller.load_state(f"{path}_overfitting_controller.json")
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise