"""
Performance metrics tracking and calculation for trading strategies.
Provides comprehensive metrics calculation, tracking, and conversion capabilities.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import json

@dataclass
class PerformanceMetrics:
    """
    Comprehensive performance metrics for trading strategy evaluation.
    Tracks standard and risk-adjusted return metrics.
    
    Required metrics:
    - sharpe_ratio: Risk-adjusted return measure
    - win_rate: Proportion of winning trades
    - profit_factor: Ratio of gross profits to gross losses
    - max_drawdown: Maximum peak to trough decline
    - volatility: Annualized return standard deviation
    
    Optional metrics with defaults:
    - total_return: Total return over period
    - annual_return: Annualized return
    - sortino_ratio: Downside risk-adjusted return
    - calmar_ratio: Return to max drawdown ratio
    - total_trades: Number of trades executed
    - alpha: Strategy excess return
    - beta: Market sensitivity
    - information_ratio: Risk-adjusted excess return
    - recovery_factor: Return to drawdown ratio
    
    Optional metrics that may be None:
    - avg_return: Average per-period return
    - winning_trades: Number of profitable trades
    - losing_trades: Number of unprofitable trades
    """
    # Required metrics (no defaults)
    sharpe_ratio: float
    win_rate: float  
    profit_factor: float
    max_drawdown: float
    volatility: float
    
    # Optional metrics with defaults
    total_return: float = 0.0
    annual_return: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    total_trades: int = 0
    alpha: float = 0.0
    beta: float = 0.0
    information_ratio: float = 0.0
    recovery_factor: float = 0.0
    
    # Optional metrics that may be None
    avg_return: Optional[float] = None
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format"""
        return {k: v for k, v in self.__dict__.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> 'PerformanceMetrics':
        """Create PerformanceMetrics instance from dictionary"""
        return cls(**data)

    def to_json(self) -> str:
        """Convert metrics to JSON string"""
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> 'PerformanceMetrics':
        """Create PerformanceMetrics instance from JSON string"""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def calculate_from_returns(cls, 
                             returns: np.ndarray, 
                             risk_free_rate: float = 0.02,
                             benchmark_returns: Optional[np.ndarray] = None) -> 'PerformanceMetrics':
        """
        Calculate performance metrics from returns series.
        
        Args:
            returns: Array of period returns
            risk_free_rate: Annual risk-free rate
            benchmark_returns: Optional benchmark returns for relative metrics
            
        Returns:
            PerformanceMetrics object containing calculated metrics
        """
        try:
            # Basic return metrics
            avg_return = np.mean(returns)
            volatility = np.std(returns) * np.sqrt(252)
            
            # Calculate excess returns
            excess_returns = returns - risk_free_rate/252
            
            # Sharpe ratio
            sharpe_ratio = (np.mean(excess_returns) / np.std(returns) * 
                          np.sqrt(252)) if np.std(returns) != 0 else 0
            
            # Win rate statistics
            winning_trades = np.sum(returns > 0)
            total_trades = len(returns[returns != 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            losing_trades = total_trades - winning_trades
            
            # Profit factor
            gross_profits = np.sum(returns[returns > 0])
            gross_losses = abs(np.sum(returns[returns < 0]))
            profit_factor = (gross_profits / gross_losses 
                           if gross_losses != 0 else float('inf'))
            
            # Drawdown calculations
            cum_returns = np.cumprod(1 + returns)
            running_max = np.maximum.accumulate(cum_returns)
            drawdowns = (cum_returns - running_max) / running_max
            max_drawdown = abs(np.min(drawdowns))
            
            # Calculate total and annual returns
            total_return = cum_returns[-1] - 1
            annual_return = (1 + total_return) ** (252/len(returns)) - 1
            
            # Downside volatility (for Sortino)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) * np.sqrt(252)
            sortino_ratio = (np.mean(excess_returns) / downside_std * np.sqrt(252)
                           if downside_std != 0 else 0)
            
            # Calmar ratio
            calmar_ratio = (annual_return / max_drawdown 
                          if max_drawdown != 0 else float('inf'))
            
            # Recovery factor
            recovery_factor = (abs(total_return / max_drawdown) 
                            if max_drawdown != 0 else float('inf'))
            
            # Calculate relative metrics if benchmark provided
            if benchmark_returns is not None:
                # Ensure lengths match
                min_length = min(len(returns), len(benchmark_returns))
                returns = returns[-min_length:]
                benchmark_returns = benchmark_returns[-min_length:]
                
                # Calculate beta
                covariance = np.cov(returns, benchmark_returns)[0,1]
                variance = np.var(benchmark_returns)
                beta = covariance / variance if variance != 0 else 1.0
                
                # Calculate alpha
                alpha = (annual_return - 
                        (risk_free_rate + beta * 
                         (np.mean(benchmark_returns) * 252 - risk_free_rate)))
                
                # Information ratio
                active_returns = returns - benchmark_returns
                information_ratio = (np.mean(active_returns) / 
                                  np.std(active_returns) * np.sqrt(252)
                                  if np.std(active_returns) != 0 else 0)
            else:
                beta = 1.0
                alpha = annual_return - risk_free_rate
                information_ratio = sharpe_ratio
            
            return cls(
                sharpe_ratio=float(sharpe_ratio),
                win_rate=float(win_rate),
                profit_factor=float(profit_factor),
                max_drawdown=float(max_drawdown),
                volatility=float(volatility),
                total_return=float(total_return),
                annual_return=float(annual_return),
                sortino_ratio=float(sortino_ratio),
                calmar_ratio=float(calmar_ratio),
                total_trades=int(total_trades),
                alpha=float(alpha),
                beta=float(beta),
                information_ratio=float(information_ratio),
                recovery_factor=float(recovery_factor),
                avg_return=float(avg_return),
                winning_trades=int(winning_trades),
                losing_trades=int(losing_trades)
            )
            
        except Exception as e:
            raise ValueError(f"Error calculating performance metrics: {str(e)}")

    def combine_metrics(self, other: 'PerformanceMetrics', 
                       weight: float = 0.5) -> 'PerformanceMetrics':
        """
        Combine metrics with another PerformanceMetrics object using weighted average.
        
        Args:
            other: Another PerformanceMetrics object
            weight: Weight for this object's metrics (1-weight used for other)
            
        Returns:
            New PerformanceMetrics object with combined metrics
        """
        if not 0 <= weight <= 1:
            raise ValueError("Weight must be between 0 and 1")
            
        other_weight = 1 - weight
        
        def weighted_combine(v1: float, v2: float) -> float:
            return v1 * weight + v2 * other_weight
        
        return PerformanceMetrics(
            sharpe_ratio=weighted_combine(self.sharpe_ratio, other.sharpe_ratio),
            win_rate=weighted_combine(self.win_rate, other.win_rate),
            profit_factor=weighted_combine(self.profit_factor, other.profit_factor),
            max_drawdown=weighted_combine(self.max_drawdown, other.max_drawdown),
            volatility=weighted_combine(self.volatility, other.volatility),
            total_return=weighted_combine(self.total_return, other.total_return),
            annual_return=weighted_combine(self.annual_return, other.annual_return),
            sortino_ratio=weighted_combine(self.sortino_ratio, other.sortino_ratio),
            calmar_ratio=weighted_combine(self.calmar_ratio, other.calmar_ratio),
            total_trades=int(weighted_combine(self.total_trades, other.total_trades)),
            alpha=weighted_combine(self.alpha, other.alpha),
            beta=weighted_combine(self.beta, other.beta),
            information_ratio=weighted_combine(self.information_ratio, 
                                            other.information_ratio),
            recovery_factor=weighted_combine(self.recovery_factor, 
                                          other.recovery_factor)
        )

    def evaluate_strategy(self) -> Dict[str, str]:
        """
        Evaluate strategy performance using standard benchmarks.
        
        Returns:
            Dictionary of performance assessments
        """
        assessments = {}
        
        # Sharpe Ratio assessment
        if self.sharpe_ratio >= 2.0:
            assessments['sharpe_ratio'] = 'Excellent'
        elif self.sharpe_ratio >= 1.0:
            assessments['sharpe_ratio'] = 'Good'
        else:
            assessments['sharpe_ratio'] = 'Poor'
            
        # Win Rate assessment  
        if self.win_rate >= 0.6:
            assessments['win_rate'] = 'Excellent'
        elif self.win_rate >= 0.5:
            assessments['win_rate'] = 'Good'
        else:
            assessments['win_rate'] = 'Poor'
            
        # Profit Factor assessment
        if self.profit_factor >= 2.0:
            assessments['profit_factor'] = 'Excellent'
        elif self.profit_factor >= 1.5:
            assessments['profit_factor'] = 'Good'
        else:
            assessments['profit_factor'] = 'Poor'
            
        # Max Drawdown assessment
        if self.max_drawdown <= 0.1:
            assessments['max_drawdown'] = 'Excellent'
        elif self.max_drawdown <= 0.2:
            assessments['max_drawdown'] = 'Good'
        else:
            assessments['max_drawdown'] = 'Poor'
            
        return assessments

    def get_summary_stats(self) -> pd.Series:
        """
        Get summary statistics as a pandas Series.
        
        Returns:
            Series containing key performance metrics
        """
        return pd.Series({
            'Total Return': f"{self.total_return:.2%}",
            'Annual Return': f"{self.annual_return:.2%}",
            'Sharpe Ratio': f"{self.sharpe_ratio:.2f}",
            'Sortino Ratio': f"{self.sortino_ratio:.2f}",
            'Max Drawdown': f"{self.max_drawdown:.2%}",
            'Win Rate': f"{self.win_rate:.2%}",
            'Profit Factor': f"{self.profit_factor:.2f}",
            'Recovery Factor': f"{self.recovery_factor:.2f}",
            'Total Trades': f"{self.total_trades}",
            'Alpha': f"{self.alpha:.2%}",
            'Beta': f"{self.beta:.2f}",
            'Information Ratio': f"{self.information_ratio:.2f}",
            'Volatility': f"{self.volatility:.2%}"
        })

    def __str__(self) -> str:
        """String representation of performance metrics"""
        metrics = self.get_summary_stats()
        return "\n".join([f"{k}: {v}" for k, v in metrics.items()])