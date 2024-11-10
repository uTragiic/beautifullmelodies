"""
Model validation system integrating existing overfitting control and backtesting components.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from datetime import datetime

from ..models.overfittingcontrol import OverfittingController
from ..backtesting.enhancedbacktest import Backtest
from ..core.performance_metrics import PerformanceMetrics
from ..analysis.marketcondition import MarketConditionAnalyzer
from ..core.types import MarketRegime

logger = logging.getLogger(__name__)

@dataclass
class ValidationMetrics:
    """Validation metrics container."""
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    alpha: float
    beta: float
    information_ratio: float
    calmar_ratio: float
    recovery_factor: float
    stability_score: float
    regime_consistency: float
    overfitting_score: float

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return vars(self)

import logging
import pandas as pd
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

class ModelValidator:
    """
    Comprehensive model validation system leveraging existing components.
    """

    def __init__(self, 
                 min_validation_window: int = 252,
                 max_validation_window: int = 756,
                 n_monte_carlo: int = 1000,
                 confidence_level: float = 0.95):
        """
        Initialize ModelValidator.
        
        Args:
            min_validation_window: Minimum validation period in days
            max_validation_window: Maximum validation period in days
            n_monte_carlo: Number of Monte Carlo simulations
            confidence_level: Confidence level for metrics
        """
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        
        # Initialize existing components
        self.overfitting_controller = OverfittingController(
            base_lookback_period=min_validation_window,
            min_samples=min_validation_window,
            max_complexity_score=0.8,
            parameter_stability_threshold=0.3
        )
        
        self.market_analyzer = MarketConditionAnalyzer()
        
        # Store configuration
        self.min_validation_window = min_validation_window
        self.max_validation_window = max_validation_window
        self.n_monte_carlo = n_monte_carlo
        self.confidence_level = confidence_level
        
        # Initialize containers
        self.validation_history: Dict[str, List[ValidationMetrics]] = {}
        

    def validate_model(self, 
                      model: Any,
                      training_data: pd.DataFrame,
                      validation_data: pd.DataFrame,
                      market_regime: str,
                      model_parameters: Dict[str, Any] = None,
                      performance_threshold: float = 0.0) -> Tuple[Dict[str, float], Dict[str, Any]]:
        """
        Comprehensive model validation including performance, overfitting, and regime analysis.
        
        Args:
            model: The trained model to validate
            training_data: DataFrame containing training dataset
            validation_data: DataFrame containing validation dataset
            market_regime: Current market regime
            model_parameters: Optional dictionary of model parameters
            performance_threshold: Minimum performance threshold
            
        Returns:
            Tuple of (metrics_dict, validation_details)
        """
        try:
            # Get model parameters if not provided
            if model_parameters is None:
                model_parameters = model.get_parameters()
                
            # Calculate performance metrics for both datasets
            train_metrics = self._calculate_performance_metrics(model, training_data)
            val_metrics = self._calculate_performance_metrics(model, validation_data)
            
            # Detect overfitting
            is_overfitting, overfitting_scores = self.overfitting_controller.detect_overfitting(
                train_metrics,
                val_metrics,
                market_regime,
                model_parameters
            )
            
            # Calculate validation scores
            validation_scores = {
                'sharpe_ratio': val_metrics.sharpe_ratio,
                'sortino_ratio': val_metrics.sortino_ratio,
                'max_drawdown': val_metrics.max_drawdown,
                'win_rate': val_metrics.win_rate,
                'profit_factor': val_metrics.profit_factor,
                'recovery_factor': val_metrics.recovery_factor,
                'volatility': val_metrics.volatility
            }
            
            # Check performance threshold
            meets_threshold = validation_scores['sharpe_ratio'] >= performance_threshold
            
            # Get regime-specific performance
            regime_performance = self._analyze_regime_performance(model, validation_data, market_regime)
            
            # Prepare response
            metrics_dict = {
                **validation_scores,
                'is_overfitting': is_overfitting,
                'meets_threshold': meets_threshold,
                'overfitting_score': overfitting_scores['final_score']
            }
            
            validation_details = {
                'performance_metrics': {
                    'training': train_metrics.to_dict(),
                    'validation': val_metrics.to_dict()
                },
                'overfitting_analysis': overfitting_scores,
                'regime_analysis': regime_performance,
                'model_parameters': model_parameters
            }
            
            # If overfitting detected, get adjustment recommendations
            if is_overfitting:
                adjusted_params = self.overfitting_controller.adjust_model(
                    model,
                    overfitting_scores,
                    market_regime
                )
                validation_details['recommended_adjustments'] = adjusted_params
                
            # Generate detailed report
            validation_details['report'] = self.overfitting_controller.generate_report(
                train_metrics,
                val_metrics,
                market_regime,
                overfitting_scores
            )
            
            # Update validation history
            self._update_validation_history(metrics_dict)
            
            return metrics_dict, validation_details
            
        except Exception as e:
            self.logger.error(f"Error in model validation: {str(e)}", exc_info=True)
            raise ValueError(f"Model validation failed: {str(e)}")    
        
    def _calculate_validation_metrics(self,
                                performance_metrics: dict,
                                monte_carlo_results: pd.DataFrame,
                                overfitting_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate comprehensive validation metrics."""
        try:
            # Calculate stability and regime consistency scores
            stability_score = 1 - overfitting_scores.get('stability_score', 0)
            regime_consistency = 1 - overfitting_scores.get('regime_score', 0)
            
            return {
                'sharpe_ratio': performance_metrics['sharpe_ratio'],
                'sortino_ratio': monte_carlo_results['sortino_ratio'].mean(),
                'win_rate': performance_metrics['win_rate'],
                'profit_factor': performance_metrics['profit_factor'],
                'max_drawdown': performance_metrics['max_drawdown'],
                'alpha': monte_carlo_results['alpha'].mean(),
                'beta': monte_carlo_results['beta'].mean(),
                'information_ratio': monte_carlo_results['information_ratio'].mean(),
                'calmar_ratio': monte_carlo_results['calmar_ratio'].mean(),
                'recovery_factor': monte_carlo_results['recovery_factor'].mean(),
                'stability_score': stability_score,
                'regime_consistency': regime_consistency,
                'overfitting_score': overfitting_scores.get('final_score', 0)
            }
        except Exception as e:
            self.logger.error(f"Error calculating validation metrics: {str(e)}")
            return {}
                
    def _analyze_regime_performance(self,
                                    data: pd.DataFrame,
                                    model: Any) -> Dict[str, Dict[str, float]]:
        """Analyze model performance across different market regimes."""
        try:
            regime_performance = {}
            
            # Get market conditions for validation period
            market_conditions = self.market_analyzer.determine_market_condition(data)
            
            # Add market conditions to data
            data = data.copy()
            data['Market_Regime'] = market_conditions
            
            # Calculate performance metrics for each regime
            for regime in MarketRegime:
                regime_data = data[data['Market_Regime'] == regime.value]
                if regime_data.empty:
                    continue
                    
                # Initialize backtester for regime data
                regime_backtester = Backtest(regime_data)
                results = regime_backtester.run_backtest(model)
                regime_metrics = regime_backtester.calculate_performance_metrics(results)
                
                regime_performance[regime.value] = {
                    'sharpe_ratio': regime_metrics.sharpe_ratio,
                    'win_rate': regime_metrics.win_rate,
                    'profit_factor': regime_metrics.profit_factor,
                    'max_drawdown': regime_metrics.max_drawdown,
                    'volatility': regime_metrics.volatility
                }
                    
            return regime_performance
        except Exception as e:
            self.logger.error(f"Error analyzing regime performance: {e}")
            raise
                
    def _update_validation_history(self, metrics: ValidationMetrics) -> None:
        """Update validation history with new metrics."""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if timestamp not in self.validation_history:
                self.validation_history[timestamp] = []
                
            self.validation_history[timestamp].append(metrics)
            
            # Maintain history length
            if len(self.validation_history) > 100:  # Keep last 100 validations
                oldest_key = min(self.validation_history.keys())
                del self.validation_history[oldest_key]
        except Exception as e:
            self.logger.error(f"Error updating validation history: {e}")
            raise
                
    def get_validation_history(self) -> Dict[str, List[ValidationMetrics]]:
        """Get validation history."""
        return self.validation_history
            
    def get_validation_summary(self) -> pd.DataFrame:
        """Get summary of validation history."""
        try:
            metrics_list = []
            
            for timestamp, metrics in self.validation_history.items():
                for metric in metrics:
                    metric_dict = metric.to_dict()
                    metric_dict['timestamp'] = timestamp
                    metrics_list.append(metric_dict)
                        
            return pd.DataFrame(metrics_list)
        except Exception as e:
            self.logger.error(f"Error generating validation summary: {e}")
            raise
