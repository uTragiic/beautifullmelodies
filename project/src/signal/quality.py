import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SignalQualityMonitor:
    """Monitors signal quality and detects degradation."""
    
    def __init__(self, 
                 lookback_period: int = 50,
                 degradation_threshold: float = 0.3,
                 min_confidence: float = 0.6):
        self.lookback_period = lookback_period
        self.degradation_threshold = degradation_threshold
        self.min_confidence = min_confidence
        self.signal_history: List[Dict] = []
        
    def evaluate_signal_quality(self,
                              current_signal: int,
                              confidence_score: float,
                              market_condition: str,
                              indicator_values: Dict[str, float]) -> Tuple[bool, Dict[str, float]]:
        """
        Evaluate signal quality and check for degradation.
        
        Args:
            current_signal: Current trading signal (-1, 0, 1)
            confidence_score: Signal confidence score
            market_condition: Current market condition
            indicator_values: Current indicator values
            
        Returns:
            Tuple of (signal_valid, quality_metrics)
        """
        try:
            # Record signal
            self.signal_history.append({
                'signal': current_signal,
                'confidence': confidence_score,
                'market_condition': market_condition,
                'indicators': indicator_values
            })
            
            # Maintain history length
            if len(self.signal_history) > self.lookback_period:
                self.signal_history.pop(0)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_quality_metrics()
            
            # Check for degradation
            signal_valid = not self._detect_degradation(quality_metrics)
            
            # Additional confidence check
            if confidence_score < self.min_confidence:
                signal_valid = False
                
            return signal_valid, quality_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating signal quality: {e}")
            return False, {}
    
    def _calculate_quality_metrics(self) -> Dict[str, float]:
        """Calculate signal quality metrics."""
        if not self.signal_history:
            return {}
            
        try:
            recent_signals = self.signal_history[-20:]  # Last 20 signals
            
            # Calculate metrics
            metrics = {
                'avg_confidence': np.mean([s['confidence'] for s in recent_signals]),
                'confidence_trend': self._calculate_trend([s['confidence'] for s in recent_signals]),
                'signal_consistency': self._calculate_consistency([s['signal'] for s in recent_signals]),
                'market_condition_stability': self._calculate_condition_stability(
                    [s['market_condition'] for s in recent_signals]
                )
            }
            
            # Calculate indicator stability
            indicator_stability = self._calculate_indicator_stability(
                [s['indicators'] for s in recent_signals]
            )
            metrics.update(indicator_stability)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating quality metrics: {e}")
            return {}
    
    def _detect_degradation(self, metrics: Dict[str, float]) -> bool:
        """
        Detect if signal quality has degraded.
        
        Returns:
            bool: True if signal quality has degraded
        """
        if not metrics:
            return True
            
        try:
            # Check confidence trend
            if metrics['confidence_trend'] < -self.degradation_threshold:
                logger.warning("Signal degradation detected: Declining confidence")
                return True
            
            # Check signal consistency
            if metrics['signal_consistency'] < self.min_confidence:
                logger.warning("Signal degradation detected: Inconsistent signals")
                return True
            
            # Check market condition stability
            if metrics['market_condition_stability'] < self.min_confidence:
                logger.warning("Signal degradation detected: Unstable market conditions")
                return True
            
            # Check average confidence
            if metrics['avg_confidence'] < self.min_confidence:
                logger.warning("Signal degradation detected: Low average confidence")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error detecting degradation: {e}")
            return True
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate linear trend in values."""
        if len(values) < 2:
            return 0.0
        return np.polyfit(range(len(values)), values, 1)[0]
    
    def _calculate_consistency(self, signals: List[int]) -> float:
        """Calculate signal consistency score."""
        if not signals:
            return 0.0
        # Check if signals maintain same direction
        changes = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1])
        return 1 - (changes / len(signals))
    
    def _calculate_condition_stability(self, conditions: List[str]) -> float:
        """Calculate market condition stability score."""
        if not conditions:
            return 0.0
        # Check how often market condition changes
        changes = sum(1 for i in range(1, len(conditions)) if conditions[i] != conditions[i-1])
        return 1 - (changes / len(conditions))
    
    def _calculate_indicator_stability(self, indicator_histories: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate indicator stability metrics."""
        stability_metrics = {}
        
        try:
            if not indicator_histories:
                return {}
                
            # Calculate for each indicator
            first_indicators = indicator_histories[0]
            for indicator in first_indicators:
                values = [h[indicator] for h in indicator_histories]
                stability_metrics[f'{indicator}_stability'] = 1 - np.std(values) / (np.mean(values) + 1e-6)
                
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Error calculating indicator stability: {e}")
            return {}