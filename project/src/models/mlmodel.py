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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Local Imports
# Adjust the import paths according to your project structure
from .overfittingcontrol import OverfittingController
from ..indicators.calculator import IndicatorCalculator
from ..core.performance_metrics import PerformanceMetrics

# Set up logging
logger = logging.getLogger(__name__)

class MachineLearningModel:
    """
    Machine learning model for market prediction with overfitting protection.
    """
    def __init__(
        self,
        lookback_period: int = 300,
        model: Optional[RandomForestClassifier] = None,
        pipeline: Optional[Pipeline] = None
    ):
        """
        Initialize the machine learning model with parameters.

        Args:
            lookback_period: Number of periods for historical analysis.
            model: Optional pre-trained model.
            pipeline: Optional pre-fitted pipeline.
        """
        try:
            self.lookback_period = lookback_period
            self.risk_free_rate = 0.02

            # Use the provided model or initialize a new one
            if model is not None:
                self.model = model
            else:
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=5,
                    min_samples_leaf=20,
                    random_state=42
                )

            # Use the provided pipeline or initialize a new one
            if pipeline is not None:
                self.pipeline = pipeline
            else:
                self.pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='mean')),
                    ('scaler', StandardScaler())
                ])

            # Feature columns
            self.feature_columns = [
                'SMA_50', 'SMA_200',
                'RSI', 'RSI_Z',
                'MACD', 'MACD_signal', 'MACD_diff', 'MACD_Z',
                'ADX', 'ATR',
                'Volume_Ratio', 'volume',
                'OBV', 'VWAP',
                'Momentum',
                'Stoch_K', 'Stoch_D',
                'BB_width'
            ]

            # Initialize overfitting controller
            self.overfitting_controller = OverfittingController(
                base_lookback_period=252,
                min_samples=100,
                max_complexity_score=0.8,
                parameter_stability_threshold=0.3
            )

            # Market condition parameters
            self.market_params = {
                'trend_threshold': 0.02,
                'volatility_percentile': 0.8,
                'volume_threshold': 1.5,
                'atr_window': 14,
                'momentum_window': 20
            }

            # Signal generation thresholds
            self.signal_thresholds = {
                'base_bull': 0.7,
                'base_bear': 0.3,
                'alpha': 0.1,
                'beta': 0.3,
                'min_adx': 25,
                'rsi_overbought': 70,
                'rsi_oversold': 30
            }

            # Performance tracking
            self.current_market_regime = "normal"
            self.performance_history = []

            # Tracking windows
            self.tracking_windows = {
                'short_term': 20,
                'medium_term': 60,
                'long_term': 252
            }

            # Store indicators for threshold adjustments
            self.indicator_thresholds = {}

            # Initialize IndicatorCalculator
            self.indicator_calculator = IndicatorCalculator()

            # Initialize variables to store training data (for updates)
            self.X_train = None
            self.y_train = None

            logger.info("MachineLearningModel initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing MachineLearningModel: {str(e)}")
            raise
    def _is_pipeline_fitted(self) -> bool:
        """
        Check if the pipeline has been fitted.

        Returns:
            True if the pipeline is fitted, False otherwise.
        """
        try:
            # Check if the scaler in the pipeline has the 'scale_' attribute
            return hasattr(self.pipeline.named_steps['scaler'], 'scale_')
        except Exception:
            return False

    def _validate_data(self, data: pd.DataFrame) -> None:
        """
        Validate input data structure and contents.

        Args:
            data: Input DataFrame to validate

        Raises:
            ValueError: If data validation fails
        """
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")

            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            missing_columns = required_columns - set(data.columns)

            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            if data.empty:
                raise ValueError("DataFrame is empty")

            if data.isnull().any().any():
                logger.warning("DataFrame contains NaN values")

        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            raise
    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for model training with enhanced validation.

        Args:
            data: DataFrame with market data

        Returns:
            Tuple of (features, target)

        Raises:
            ValueError: If data preparation fails
        """
        try:
            # Validate input data
            self._validate_data(data)

            if len(data) < self.lookback_period:
                raise ValueError(
                    f"Insufficient data points: {len(data)} < {self.lookback_period}"
                )

            # Verify all feature columns exist
            missing_features = [col for col in self.feature_columns if col not in data.columns]
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
                # Calculate missing indicators if possible
                data = self.indicator_calculator.calculate_indicators(data)

                # Recheck for missing features
                missing_features = [col for col in self.feature_columns if col not in data.columns]
                if missing_features:
                    raise ValueError(f"Could not calculate required features: {missing_features}")

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
            try:
                X = X.values
                y = np.array(y)
                y = y[:len(X)]  # Align target with features
            except Exception as e:
                logger.error(f"Error converting to numpy arrays: {str(e)}")
                raise ValueError("Error converting data to numpy arrays")

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
    def train(self, data: pd.DataFrame) -> None:
        """
        Train the model with comprehensive error handling and validation.

        Args:
            data: Market data for training
        """
        try:
            logger.info("Starting model training...")

            # Validate input data
            self._validate_data(data)

            # Calculate indicators using IndicatorCalculator
            try:
                data = self.indicator_calculator.calculate_indicators(data)
            except Exception as e:
                logger.error(f"Error calculating indicators: {str(e)}")
                raise ValueError("Failed to calculate indicators")

            # Split data into training and validation sets
            try:
                train_size = int(len(data) * 0.7)
                if train_size < self.lookback_period:
                    raise ValueError(f"Insufficient training data: {train_size} < {self.lookback_period}")

                train_data = data.iloc[:train_size]
                val_data = data.iloc[train_size:]
            except Exception as e:
                logger.error(f"Error splitting data: {str(e)}")
                raise ValueError("Failed to split data")

            # Prepare features
            try:
                X_train, y_train = self.prepare_features(train_data)
                X_val, y_val = self.prepare_features(val_data)
            except Exception as e:
                logger.error(f"Error preparing features: {str(e)}")
                raise ValueError("Failed to prepare features")

            # Store training data for future updates
            self.X_train = X_train
            self.y_train = y_train

            # Log types and shapes
            logger.debug(f"X_train type: {type(X_train)}, shape: {X_train.shape}")
            logger.debug(f"y_train type: {type(y_train)}, shape: {y_train.shape}")
            logger.debug(f"X_val type: {type(X_val)}, shape: {X_val.shape}")
            logger.debug(f"y_val type: {type(y_val)}, shape: {y_val.shape}")

            # Ensure y_train and y_val are array-like and not scalars
            if isinstance(y_train, (int, float)) or np.isscalar(y_train):
                y_train = np.array([y_train])
            if isinstance(y_val, (int, float)) or np.isscalar(y_val):
                y_val = np.array([y_val])

            # Fit the pipeline and transform data
            try:
                X_train_scaled = self.pipeline.fit_transform(X_train)
                X_val_scaled = self.pipeline.transform(X_val)
            except Exception as e:
                logger.error(f"Error in data transformation: {str(e)}")
                raise ValueError("Failed to transform data")

            # Ensure that the lengths of X and y match
            if len(X_train_scaled) != len(y_train):
                logger.error(f"Length mismatch: X_train_scaled has {len(X_train_scaled)} samples, y_train has {len(y_train)} samples")
                raise ValueError("Feature and target lengths do not match in training data")
            if len(X_val_scaled) != len(y_val):
                logger.error(f"Length mismatch: X_val_scaled has {len(X_val_scaled)} samples, y_val has {len(y_val)} samples")
                raise ValueError("Feature and target lengths do not match in validation data")

            # Train model
            logger.info("Training model...")
            try:
                self.model.fit(X_train_scaled, y_train)
            except Exception as e:
                logger.error(f"Error fitting model: {str(e)}")
                raise ValueError("Failed to fit model")

            # Get predictions
            try:
                train_predictions = self.model.predict_proba(X_train_scaled)[:, 1]
                val_predictions = self.model.predict_proba(X_val_scaled)[:, 1]
            except Exception as e:
                logger.error(f"Error generating predictions: {str(e)}")
                raise ValueError("Failed to generate predictions")

            # Calculate returns
            try:
                train_returns = train_data['close'].pct_change().shift(-1).dropna().values
                val_returns = val_data['close'].pct_change().shift(-1).dropna().values

                # Ensure alignment of returns and predictions
                min_len_train = min(len(train_predictions), len(train_returns), len(y_train))
                train_predictions = train_predictions[-min_len_train:]
                train_returns = train_returns[-min_len_train:]
                y_train = y_train[-min_len_train:]

                min_len_val = min(len(val_predictions), len(val_returns), len(y_val))
                val_predictions = val_predictions[-min_len_val:]
                val_returns = val_returns[-min_len_val:]
                y_val = y_val[-min_len_val:]
            except Exception as e:
                logger.error(f"Error calculating returns: {str(e)}")
                raise ValueError("Failed to calculate returns")

            # Calculate performance metrics
            try:
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
            except Exception as e:
                logger.error(f"Error calculating metrics: {str(e)}")
                raise ValueError("Failed to calculate performance metrics")

            # Get model parameters
            model_parameters = self.get_parameters()

            # Initialize overfitting_scores
            overfitting_scores = {}

            # Check for overfitting
            try:
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

                    # Recalculate predictions and metrics after adjustment
                    train_predictions = self.model.predict_proba(X_train_scaled)[:, 1]
                    val_predictions = self.model.predict_proba(X_val_scaled)[:, 1]

                    # Re-align data
                    min_len_train = min(len(train_predictions), len(train_returns), len(y_train))
                    train_predictions = train_predictions[-min_len_train:]
                    train_returns = train_returns[-min_len_train:]
                    y_train = y_train[-min_len_train:]

                    min_len_val = min(len(val_predictions), len(val_returns), len(y_val))
                    val_predictions = val_predictions[-min_len_val:]
                    val_returns = val_returns[-min_len_val:]
                    y_val = y_val[-min_len_val:]

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
            except Exception as e:
                logger.error(f"Error in overfitting control: {str(e)}")
                raise ValueError("Failed in overfitting control")

            # Generate and save report
            try:
                report = self.overfitting_controller.generate_report(
                    in_sample_metrics=in_sample_metrics,
                    out_sample_metrics=out_sample_metrics,
                    market_regime=self.current_market_regime,
                    overfitting_scores=overfitting_scores
                )
                self._save_training_report(report)
            except Exception as e:
                logger.error(f"Error generating/saving report: {str(e)}")

            logger.info("Model training completed successfully")

        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
    def predict(self, data: pd.DataFrame) -> float:
        """
        Generate predictions with comprehensive error handling.

        Args:
            data: Market data

        Returns:
            Prediction probability
        """
        try:
            # Validate input data
            self._validate_data(data)

            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")

            if not self._is_pipeline_fitted():
                raise ValueError("Pipeline not fitted. Please train the model first.")

            # Calculate indicators using IndicatorCalculator
            try:
                data = self.indicator_calculator.calculate_indicators(data)
            except Exception as e:
                logger.error(f"Error calculating indicators for prediction: {str(e)}")
                raise ValueError("Failed to calculate indicators")

            # Prepare features
            try:
                X, _ = self.prepare_features(data)
                if len(X) == 0:
                    raise ValueError("No valid features after preparation")
            except Exception as e:
                logger.error(f"Error preparing features for prediction: {str(e)}")
                raise ValueError("Failed to prepare features")

            # Transform features
            try:
                X_transformed = self.pipeline.transform(X)
            except Exception as e:
                logger.error(f"Error transforming features: {str(e)}")
                raise ValueError("Failed to transform features")

            # Generate prediction
            try:
                prediction = self.model.predict_proba(X_transformed[-1].reshape(1, -1))[0][1]
                logger.info(f"Generated prediction: {prediction:.4f}")
                return prediction
            except Exception as e:
                logger.error(f"Error generating prediction: {str(e)}")
                raise ValueError("Failed to generate prediction")

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            raise
    def update(self, new_data: pd.DataFrame, market_regime: str) -> None:
        """
        Update the model with new data and market regime information.

        Args:
            new_data: New market data
            market_regime: Current market regime
        """
        try:
            # Validate input
            self._validate_data(new_data)
            if not isinstance(market_regime, str):
                raise ValueError("market_regime must be a string")

            self.current_market_regime = market_regime

            # Calculate indicators
            try:
                new_data = self.indicator_calculator.calculate_indicators(new_data)
            except Exception as e:
                logger.error(f"Error calculating indicators for update: {str(e)}")
                raise ValueError("Failed to calculate indicators")

            # Prepare features
            try:
                X_new, y_new = self.prepare_features(new_data)
                if len(X_new) == 0 or len(y_new) == 0:
                    raise ValueError("No valid data for update")
            except Exception as e:
                logger.error(f"Error preparing features for update: {str(e)}")
                raise ValueError("Failed to prepare features")

            # Transform new data
            if not self._is_pipeline_fitted():
                raise ValueError("Pipeline not fitted. Cannot update model without a fitted pipeline.")

            try:
                X_transformed = self.pipeline.transform(X_new)
            except Exception as e:
                logger.error(f"Error transforming update data: {str(e)}")
                raise ValueError("Failed to transform update data")

            # Get predictions for performance calculation
            try:
                predictions = self.model.predict_proba(X_transformed)[:, 1]
                returns = new_data['close'].pct_change().shift(-1).dropna().values
                # Ensure alignment of returns and predictions
                min_length = min(len(predictions), len(returns), len(y_new))
                predictions = predictions[-min_length:]
                returns = returns[-min_length:]
                y_new = y_new[-min_length:]
            except Exception as e:
                logger.error(f"Error calculating predictions/returns: {str(e)}")
                raise ValueError("Failed to calculate predictions or returns")

            # Calculate performance metrics
            try:
                performance_metrics = self._calculate_performance_metrics(
                    predictions, y_new, returns
                )
                self.performance_history.append(performance_metrics)
            except Exception as e:
                logger.error(f"Error calculating update metrics: {str(e)}")
                raise ValueError("Failed to calculate performance metrics")

            # Update the model incrementally if supported
            if hasattr(self.model, 'partial_fit'):
                try:
                    classes = np.unique(y_new)
                    self.model.partial_fit(X_transformed, y_new, classes=classes)
                    logger.info("Model successfully updated with new data using partial_fit")
                except Exception as e:
                    logger.error(f"Error updating model with partial_fit: {str(e)}")
                    raise ValueError("Failed to update model with partial_fit")
            else:
                # Retrain the model with combined old and new data
                try:
                    # Ensure self.X_train and self.y_train exist
                    if self.X_train is None or self.y_train is None:
                        raise ValueError("Training data not available for full retrain")

                    # Combine data
                    X_combined = np.vstack([self.X_train, X_transformed])
                    y_combined = np.concatenate([self.y_train, y_new])

                    # Retrain the model
                    self.model.fit(X_combined, y_combined)

                    # Update stored training data
                    self.X_train = X_combined
                    self.y_train = y_combined

                    logger.info("Model successfully retrained with combined data")
                except Exception as e:
                    logger.error(f"Error retraining model: {str(e)}")
                    raise ValueError("Failed to retrain model with combined data")

        except Exception as e:
            logger.error(f"Error updating model: {str(e)}")
            raise
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores with error handling.

        Returns:
            Dictionary of feature importances
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")

            if not hasattr(self.model, 'feature_importances_'):
                raise ValueError("Model does not support feature importances")

            importances = self.model.feature_importances_
            if len(importances) != len(self.feature_columns):
                raise ValueError("Feature importance length mismatch")

            return dict(zip(self.feature_columns, importances))

        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get current model parameters with error handling.

        Returns:
            Dictionary of model parameters
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")

            params = self.model.get_params()
            return params

        except Exception as e:
            logger.error(f"Error getting parameters: {str(e)}")
            raise

    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Set model parameters with validation.

        Args:
            parameters: Dictionary of parameters to set
        """
        try:
            if not isinstance(parameters, dict):
                raise ValueError("Parameters must be provided as a dictionary")

            if self.model is None:
                raise ValueError("Model not initialized")

            # Validate parameters before setting
            valid_params = self.model.get_params()
            for param in parameters:
                if param not in valid_params:
                    raise ValueError(f"Invalid parameter: {param}")

            self.model.set_params(**parameters)
            logger.info("Model parameters updated successfully")

        except Exception as e:
            logger.error(f"Error setting parameters: {str(e)}")
            raise
    def _save_training_report(self, report: Dict[str, Any]) -> None:
        """
        Save training report to file with error handling.

        Args:
            report: Dictionary containing training report data
        """
        try:
            if not isinstance(report, dict):
                raise ValueError("Report must be a dictionary")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_path = f"training_reports/model_report_{timestamp}.json"

            # Create directory if it doesn't exist
            try:
                os.makedirs("training_reports", exist_ok=True)
            except Exception as e:
                logger.error(f"Error creating report directory: {str(e)}")
                raise ValueError("Failed to create report directory")

            # Add metadata to report
            report['timestamp'] = timestamp
            report['model_parameters'] = self.get_parameters()
            report['feature_importance'] = self.get_feature_importance()

            # Save report
            try:
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=4)
                logger.info(f"Training report saved to {report_path}")
            except Exception as e:
                logger.error(f"Error writing report to file: {str(e)}")
                raise ValueError("Failed to write report to file")

        except Exception as e:
            logger.error(f"Error saving training report: {str(e)}")
            raise
    def save_model(self, path: str) -> None:
        """
        Save model state and components with error handling.

        Args:
            path: Path to save the model
        """
        try:
            if not isinstance(path, str):
                raise ValueError("Path must be a string")

            if self.model is None:
                raise ValueError("No trained model to save")

            # Prepare model state
            model_state = {
                'model': self.model,
                'pipeline': self.pipeline,
                'feature_columns': self.feature_columns,
                'lookback_period': self.lookback_period,
                'current_market_regime': self.current_market_regime,
                'performance_history': self.performance_history,
                'indicator_thresholds': self.indicator_thresholds,
                'market_params': self.market_params,
                'signal_thresholds': self.signal_thresholds,
                'X_train': self.X_train,
                'y_train': self.y_train
            }

            # Save model state
            try:
                joblib.dump(model_state, f"{path}_model.joblib")
            except Exception as e:
                logger.error(f"Error saving model state: {str(e)}")
                raise ValueError("Failed to save model state")

            # Save overfitting controller state
            try:
                self.overfitting_controller.save_state(f"{path}_overfitting_controller.json")
            except Exception as e:
                logger.error(f"Error saving overfitting controller: {str(e)}")
                raise ValueError("Failed to save overfitting controller")

            logger.info(f"Model saved successfully to {path}")

        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    @classmethod
    def load_model(cls, path: str) -> 'MachineLearningModel':
        """
        Load model state and components with error handling, and return a new instance.

        Args:
            path: Path to load the model from

        Returns:
            Instance of MachineLearningModel with loaded state
        """
        try:
            if not isinstance(path, str):
                raise ValueError("Path must be a string")

            # Load model state
            try:
                model_state = joblib.load(f"{path}_model.joblib")
            except Exception as e:
                logger.error(f"Error loading model state: {str(e)}")
                raise ValueError("Failed to load model state")

            # Validate model state
            required_keys = {
                'model', 'pipeline', 'feature_columns', 'lookback_period',
                'current_market_regime', 'performance_history',
                'indicator_thresholds', 'market_params', 'signal_thresholds',
                'X_train', 'y_train'
            }
            missing_keys = required_keys - set(model_state.keys())
            if missing_keys:
                raise ValueError(f"Incomplete model state, missing: {missing_keys}")

            # Create a new instance with loaded model and pipeline
            instance = cls(
                lookback_period=model_state['lookback_period'],
                model=model_state['model'],
                pipeline=model_state['pipeline']
            )

            # Restore other attributes
            instance.feature_columns = model_state['feature_columns']
            instance.current_market_regime = model_state['current_market_regime']
            instance.performance_history = model_state['performance_history']
            instance.indicator_thresholds = model_state['indicator_thresholds']
            instance.market_params = model_state['market_params']
            instance.signal_thresholds = model_state['signal_thresholds']
            instance.X_train = model_state['X_train']
            instance.y_train = model_state['y_train']

            # Load overfitting controller state
            try:
                instance.overfitting_controller.load_state(f"{path}_overfitting_controller.json")
            except Exception as e:
                logger.error(f"Error loading overfitting controller: {str(e)}")
                raise ValueError("Failed to load overfitting controller")

            logger.info(f"Model loaded successfully from {path}")
            return instance

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    def _calculate_performance_metrics(self, predictions: np.ndarray, y: np.ndarray, returns: np.ndarray) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for the model.

        Args:
            predictions: Model predictions
            y: Actual target values
            returns: Actual returns

        Returns:
            PerformanceMetrics object
        """
        try:
            # Input validation
            if not isinstance(predictions, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(returns, np.ndarray):
                raise ValueError("Inputs must be numpy arrays")

            # Ensure arrays are the same length
            min_length = min(len(returns), len(predictions), len(y))
            if min_length == 0:
                raise ValueError("Empty arrays provided")

            returns = returns[-min_length:]
            predictions = predictions[-min_length:]
            y = y[-min_length:]

            # Calculate strategy returns
            try:
                strategy_returns = returns * np.where(predictions > 0.5, 1, -1)
            except Exception as e:
                logger.error(f"Error calculating strategy returns: {str(e)}")
                raise ValueError("Error calculating strategy returns")

            # Calculate metrics
            winning_trades = np.sum(strategy_returns > 0)
            total_trades = len(strategy_returns)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            # Calculate profit factor
            gross_profits = np.sum(strategy_returns[strategy_returns > 0])
            gross_losses = abs(np.sum(strategy_returns[strategy_returns < 0]))
            profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')

            # Calculate volatility and Sharpe ratio
            try:
                volatility = np.std(strategy_returns) * np.sqrt(252)
                sharpe_ratio = (np.mean(strategy_returns) - self.risk_free_rate/252) / volatility * np.sqrt(252) if volatility != 0 else 0
            except Exception as e:
                logger.error(f"Error calculating volatility metrics: {str(e)}")
                raise ValueError("Error calculating volatility metrics")

            # Calculate max drawdown
            try:
                cumulative_returns = np.cumprod(1 + strategy_returns)
                running_max = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - running_max) / running_max
                max_drawdown = abs(np.min(drawdowns)) if len(drawdowns) > 0 else 0
            except Exception as e:
                logger.error(f"Error calculating drawdown: {str(e)}")
                raise ValueError("Error calculating drawdown")

            # Create PerformanceMetrics object
            return PerformanceMetrics(
                sharpe_ratio=float(sharpe_ratio),
                win_rate=float(win_rate),
                profit_factor=float(profit_factor),
                max_drawdown=float(max_drawdown),
                volatility=float(volatility)
            )

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals based on the model's predictions.

        Args:
            data: DataFrame with market data.

        Returns:
            Series with signals (1 for buy, -1 for sell, 0 for hold)
        """
        try:
            # Validate and prepare data
            self._validate_data(data)
            if not self._is_pipeline_fitted():
                raise ValueError("Pipeline not fitted. Please train the model first.")

            data = self.indicator_calculator.calculate_indicators(data)
            X, _ = self.prepare_features(data)
            if len(X) == 0:
                raise ValueError("No valid features after preparation")

            # Transform features
            X_transformed = self.pipeline.transform(X)

            # Generate predictions
            prediction_probabilities = self.model.predict_proba(X_transformed)[:, 1]

            # Generate signals based on thresholds
            signals = pd.Series(
                np.where(
                    prediction_probabilities > self.signal_thresholds['base_bull'], 1,
                    np.where(
                        prediction_probabilities < self.signal_thresholds['base_bear'], -1,
                        0
                    )
                ),
                index=data.index[-len(prediction_probabilities):]
            )

            return signals

        except Exception as e:
            logger.error(f"Error generating signals: {str(e)}")
            raise
