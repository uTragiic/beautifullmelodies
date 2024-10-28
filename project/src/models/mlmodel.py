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
    
    def __init__(self, lookback_period: int = 500):
        """
        Initialize the machine learning model.
        
        Args:
            lookback_period: Number of periods to look back for training
        """
        self.lookback_period = lookback_period
        
        # Create preprocessing pipeline
        self.pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=20,
            random_state=42
        )
        
        # Define feature columns
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
        self.performance_history: List[PerformanceMetrics] = []

    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and target for model training.
        
        Args:
            data: Market data with indicators
            
        Returns:
            Tuple of (features, target)
        """
        try:
            if any(col not in data.columns for col in self.feature_columns):
                missing_cols = [col for col in self.feature_columns if col not in data.columns]
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            X = data[self.feature_columns].iloc[-self.lookback_period:]
            y = np.where(data['close'].pct_change().shift(-1).iloc[-self.lookback_period:] > 0, 1, 0)
            
            # Handle NaN values in target
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
            # Drop any remaining rows with NaN in features
            mask = ~X.isna().any(axis=1)
            X = X[mask]
            y = y[mask]
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise

    def calculate_performance_metrics(self, predictions: np.ndarray,
                                actuals: np.ndarray,
                                returns: np.ndarray) -> PerformanceMetrics:
        """
        Calculate performance metrics for the model.
        
        Args:
            predictions: Model predictions
            actuals: Actual values
            returns: Asset returns
            
        Returns:
            PerformanceMetrics object
        """
        try:
            # Ensure all arrays have the same length by using the minimum length
            min_length = min(len(predictions), len(actuals), len(returns))
            predictions = predictions[-min_length:]
            actuals = actuals[-min_length:]
            returns = returns[-min_length:].values  # Convert to numpy array
            
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
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            raise

    def train(self, data: pd.DataFrame) -> None:
        """
        Train the model with overfitting protection.
        
        Args:
            data: Training data
        """
        try:
            # Split data into training and validation sets
            train_size = int(len(data) * 0.7)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:]

            # Prepare features
            X_train, y_train = self.prepare_features(train_data)
            X_val, y_val = self.prepare_features(val_data)

            if len(X_train) == 0 or len(X_val) == 0:
                raise ValueError("No valid training/validation data after preparation")

            # Transform features using pipeline
            X_train_transformed = self.pipeline.fit_transform(X_train)
            X_val_transformed = self.pipeline.transform(X_val)

            # Train model
            self.model.fit(X_train_transformed, y_train)

            # Get predictions
            train_predictions = self.model.predict(X_train_transformed)
            val_predictions = self.model.predict(X_val_transformed)

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
                self.model.fit(X_train_transformed, y_train)

                logger.info("Model adjusted due to detected overfitting")

            # Generate and save report
            report = self.overfitting_controller.generate_report(
                in_sample_metrics=in_sample_metrics,
                out_sample_metrics=out_sample_metrics,
                market_regime=self.current_market_regime,
                overfitting_scores=overfitting_scores
            )

            # Save report
            self._save_training_report(report)

        except Exception as e:
            logger.error(f"Error in model training: {e}")
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