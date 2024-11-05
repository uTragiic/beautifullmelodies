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
                       validation_data: pd.DataFrame,
                       model_parameters: Dict[str, Any],
                       market_regime: Optional[str] = None) -> Tuple[ValidationMetrics, Dict[str, Any]]:
        """
        Perform comprehensive model validation.
        
        Args:
            model: Model to validate
            validation_data: Data for validation
            model_parameters: Current model parameters
            market_regime: Current market regime
            
        Returns:
            Tuple of (validation metrics, detailed results)
        """
        try:
            # Initialize backtester with validation data
            backtester = Backtest(validation_data)
            
            # Get in-sample and out-of-sample metrics
            in_sample_metrics, out_sample_metrics = self._get_performance_metrics(
                backtester, model
            )
            
            # Check for overfitting
            is_overfitting, overfitting_scores = self.overfitting_controller.detect_overfitting(
                in_sample_metrics=in_sample_metrics,
                out_sample_metrics=out_sample_metrics,
                market_regime=market_regime,
                model_parameters=model_parameters
            )
            
            adjusted_params = model_parameters.copy()
            if is_overfitting:
                # Get adjusted parameters from overfitting controller
                adjusted_params = self.overfitting_controller.adjust_model(
                    model=model,
                    overfitting_scores=overfitting_scores,
                    market_regime=market_regime
                )
                logger.warning(f"Overfitting detected, parameters adjusted: {adjusted_params}")
            
            # Generate overfitting analysis report
            report = self.overfitting_controller.generate_report(
                in_sample_metrics=in_sample_metrics,
                out_sample_metrics=out_sample_metrics,
                market_regime=market_regime,
                overfitting_scores=overfitting_scores
            )
            
            # Run Monte Carlo simulation using EnhancedBacktest
            monte_carlo_results = backtester.run_monte_carlo(
                strategy=model,
                num_simulations=self.n_monte_carlo,
                simulation_length=self.min_validation_window
            )
            
            # Calculate comprehensive validation metrics
            metrics = self._calculate_validation_metrics(
                out_sample_metrics,
                monte_carlo_results,
                overfitting_scores
            )
            
            # Compile detailed results
            detailed_results = {
                'is_overfitting': is_overfitting,
                'overfitting_scores': overfitting_scores,
                'adjusted_parameters': adjusted_params,
                'overfitting_report': report,
                'monte_carlo_results': monte_carlo_results,
                'regime_performance': self._analyze_regime_performance(
                    validation_data, model
                )
            }
            
            # Update validation history
            self._update_validation_history(metrics)
            
            return metrics, detailed_results
                
        except Exception as e:
            logger.error(f"Error in model validation: {e}")
            raise
                
    def _get_performance_metrics(self,
                                 backtester: Backtest,
                                 model: Any) -> Tuple[PerformanceMetrics, PerformanceMetrics]:
        """Get in-sample and out-of-sample performance metrics."""
        # Run backtest with walk-forward optimization
        results = backtester.run_backtest(model)
        
        # Split results into in-sample and out-of-sample periods
        split_idx = int(len(results) * 0.7)
        in_sample = results.iloc[:split_idx]
        out_sample = results.iloc[split_idx:]
        
        # Calculate metrics
        in_sample_metrics = backtester.calculate_performance_metrics(in_sample)
        out_sample_metrics = backtester.calculate_performance_metrics(out_sample)
        
        return in_sample_metrics, out_sample_metrics
                
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
            
    def _update_validation_history(self, metrics: ValidationMetrics) -> None:
        """Update validation history with new metrics."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if timestamp not in self.validation_history:
            self.validation_history[timestamp] = []
            
        self.validation_history[timestamp].append(metrics)
        
        # Maintain history length
        if len(self.validation_history) > 100:  # Keep last 100 validations
            oldest_key = min(self.validation_history.keys())
            del self.validation_history[oldest_key]
                
    def get_validation_history(self) -> Dict[str, List[ValidationMetrics]]:
        """Get validation history."""
        return self.validation_history
            
    def get_validation_summary(self) -> pd.DataFrame:
        """Get summary of validation history."""
        metrics_list = []
        
        for timestamp, metrics in self.validation_history.items():
            for metric in metrics:
                metric_dict = metric.to_dict()
                metric_dict['timestamp'] = timestamp
                metrics_list.append(metric_dict)
                    
        return pd.DataFrame(metrics_list)
            

