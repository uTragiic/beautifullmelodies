# Standard Library Imports
import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime

# Third-Party Imports
import numpy as np
import pandas as pd

# Local Imports
from ..core.types import MarketRegime
from ..core.performance_metrics import PerformanceMetrics

# Set up logging
logger = logging.getLogger(__name__)

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
        Initialize the overfitting controller.
        
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
        self.key_metrics = ['sharpe_ratio', 'win_rate', 'profit_factor', 
                           'max_drawdown', 'volatility']
        
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
        Detect overfitting using multiple indicators.
        
        Args:
            in_sample_metrics: Performance metrics from training data
            out_sample_metrics: Performance metrics from test data
            market_regime: Current market regime
            model_parameters: Current model parameters
            
        Returns:
            Tuple of (is_overfitting, detailed_scores)
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
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary of thresholds
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
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary of weights
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
        
        Args:
            in_sample: In-sample performance metrics
            out_sample: Out-of-sample performance metrics
            thresholds: Current thresholds
            
        Returns:
            Performance degradation score
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
        
        Args:
            current_parameters: Current model parameters
            thresholds: Current thresholds
            
        Returns:
            Parameter stability score
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
        
        Args:
            metrics: Current performance metrics
            current_regime: Current market regime
            
        Returns:
            Regime consistency score
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
        
        Args:
            model_parameters: Model parameters
            
        Returns:
            Complexity score
        """
        complexity_factors = []

        if 'n_features' in model_parameters:
            n_features = model_parameters['n_features']
            complexity_factors.append(min(n_features / 100, 1.0))

        if 'max_depth' in model_parameters:
            depth = model_parameters['max_depth']
            complexity_factors.append(min(depth / 10, 1.0))

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
            Dictionary of adjusted parameters
        """
        try:
            current_params = model.get_params()
            adjustment_strength = min(overfitting_scores['final_score'], 0.5)
            
            # Adjust for complexity if needed
            if overfitting_scores['complexity_score'] > self.max_complexity_score:
                current_params = self._reduce_complexity(current_params, adjustment_strength)
            
            # Adjust for stability if needed
            if overfitting_scores['stability_score'] > self.parameter_stability_threshold:
                current_params = self._increase_regularization(current_params, adjustment_strength)
            
            # Apply regime-specific adjustments
            current_params = self._apply_regime_adjustments(current_params, market_regime)
            
            # Update adjustment history
            self.adjustment_history.append(current_params)
            
            return current_params
            
        except Exception as e:
            logger.error(f"Error adjusting model: {e}")
            return current_params

    def _reduce_complexity(self, parameters: Dict[str, Any], 
                          adjustment_strength: float) -> Dict[str, Any]:
        """
        Reduce model complexity by adjusting relevant parameters.
        
        Args:
            parameters: Current parameters
            adjustment_strength: Strength of adjustment
            
        Returns:
            Dictionary of adjusted parameters
        """
        adjusted = parameters.copy()
        
        if 'max_depth' in adjusted:
            adjusted['max_depth'] = max(
                3,  # Minimum depth
                int(adjusted['max_depth'] * (1 - adjustment_strength))
            )
        
        if 'n_estimators' in adjusted:
            adjusted['n_estimators'] = max(
                50,  # Minimum estimators
                int(adjusted['n_estimators'] * (1 - adjustment_strength * 0.5))
            )
        
        if 'min_samples_leaf' in adjusted:
            adjusted['min_samples_leaf'] = max(
                1,
                int(adjusted['min_samples_leaf'] * (1 + adjustment_strength))
            )
        
        return adjusted

    def _increase_regularization(self, parameters: Dict[str, Any], 
                               adjustment_strength: float) -> Dict[str, Any]:
        """
        Increase model regularization to combat instability.
        
        Args:
            parameters: Current parameters
            adjustment_strength: Strength of adjustment
            
        Returns:
            Dictionary of adjusted parameters
        """
        adjusted = parameters.copy()
        
        if 'l1_ratio' in adjusted:
            adjusted['l1_ratio'] = min(
                1.0,
                adjusted['l1_ratio'] * (1 + adjustment_strength)
            )
        
        if 'l2_ratio' in adjusted:
            adjusted['l2_ratio'] = min(
                1.0,
                adjusted['l2_ratio'] * (1 + adjustment_strength)
            )
        
        if 'dropout_rate' in adjusted:
            adjusted['dropout_rate'] = min(
                0.5,
                adjusted['dropout_rate'] * (1 + adjustment_strength)
            )
        
        return adjusted

    def _apply_regime_adjustments(self, parameters: Dict[str, Any],
                                market_regime: str) -> Dict[str, Any]:
        """
        Apply market regime-specific parameter adjustments.
        
        Args:
            parameters: Current parameters
            market_regime: Current market regime
            
        Returns:
            Dictionary of adjusted parameters
        """
        adjusted = parameters.copy()
        
        if market_regime == MarketRegime.TRENDING.value:
            # Favor longer-term patterns in trending markets
            if 'lookback_period' in adjusted:
                adjusted['lookback_period'] = int(adjusted['lookback_period'] * 1.2)
            if 'momentum_period' in adjusted:
                adjusted['momentum_period'] = int(adjusted['momentum_period'] * 1.3)
            if 'learning_rate' in adjusted:
                adjusted['learning_rate'] *= 0.8
                
        elif market_regime == MarketRegime.RANGING.value:
            # Favor faster adaptation in ranging markets
            if 'lookback_period' in adjusted:
                adjusted['lookback_period'] = int(adjusted['lookback_period'] * 0.8)
            if 'learning_rate' in adjusted:
                adjusted['learning_rate'] *= 1.2
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
        
        Args:
            metrics: Performance metrics
            market_regime: Current market regime
            overfitting_score: Current overfitting score
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
            Dictionary containing detailed analysis and recommendations
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
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

            # Add historical context if available
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

    def _analyze_performance_trend(self) -> Dict[str, float]:
        """
        Analyze historical performance trends.
        
        Returns:
            Dictionary of trend analysis results
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
        
        Returns:
            Dictionary of regime stability metrics
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
        
        Returns:
            Dictionary of parameter stability metrics
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
        
        Args:
            filepath: Path to save the state
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
        
        Args:
            filepath: Path to load the state from
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

    def _generate_recommendations(self,
                            overfitting_scores: Dict[str, float],
                            market_regime: str) -> List[str]:
        """
        Generate specific recommendations based on overfitting analysis.
        
        Args:
            overfitting_scores: Dictionary of overfitting detection scores
            market_regime: Current market regime
            
        Returns:
            List of recommendation strings
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
        elif market_regime == MarketRegime.RANGING.value:
            recommendations.append(
                "In ranging regime: Consider implementing mean reversion strategies "
                "and tighter stop losses."
            )
        elif market_regime == MarketRegime.LOW_VOLATILITY.value:
            recommendations.append(
                "In low volatility regime: Consider more aggressive position sizing "
                "and wider stop losses."
            )

        # Add recommendations based on specific scores
        if 'regime_score' in overfitting_scores and overfitting_scores['regime_score'] > 0.7:
            recommendations.append(
                "High regime sensitivity detected. Consider implementing "
                "regime-specific parameter sets."
            )

        if 'final_score' in overfitting_scores and overfitting_scores['final_score'] > 0.8:
            recommendations.append(
                "Critical overfitting detected. Consider complete model retraining "
                "with different architecture."
            )

        # If no specific issues found
        if not recommendations:
            recommendations.append(
                "Model performance is within acceptable parameters. Continue monitoring "
                "for changes in market conditions."
            )

        return recommendations