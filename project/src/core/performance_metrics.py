# Standard Library Imports
from dataclasses import dataclass
from typing import Dict, Optional
import json

# Third-Party Imports
import numpy as np
import pandas as pd

@dataclass
class PerformanceMetrics:
    """
    Data class for storing and calculating trading performance metrics.
    """
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    volatility: float
    avg_return: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    total_trades: Optional[int] = None
    winning_trades: Optional[int] = None
    losing_trades: Optional[int] = None
    
    @classmethod
    def calculate_from_returns(cls, returns: pd.Series, risk_free_rate: float = 0.02) -> 'PerformanceMetrics':
        """
        Calculate performance metrics from a series of returns.
        
        Args:
            returns (pd.Series): Series of period returns
            risk_free_rate (float): Annual risk-free rate
            
        Returns:
            PerformanceMetrics: Calculated performance metrics
        """
        # Calculate basic metrics
        avg_return = returns.mean()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate Sharpe Ratio
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() != 0 else 0
        
        # Calculate Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252)
        sortino_ratio = np.sqrt(252) * excess_returns.mean() / downside_std if downside_std != 0 else 0
        
        # Calculate win rate and trade counts
        winning_trades = (returns > 0).sum()
        losing_trades = (returns < 0).sum()
        total_trades = len(returns)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        gross_profits = returns[returns > 0].sum()
        gross_losses = abs(returns[returns < 0].sum())
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        rolling_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Calculate Calmar Ratio
        calmar_ratio = avg_return * 252 / max_drawdown if max_drawdown != 0 else 0
        
        return cls(
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            volatility=volatility,
            avg_return=avg_return,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades
        )

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary format"""
        return {
            'sharpe_ratio': self.sharpe_ratio,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'max_drawdown': self.max_drawdown,
            'volatility': self.volatility,
            'avg_return': self.avg_return,
            'sortino_ratio': self.sortino_ratio,
            'calmar_ratio': self.calmar_ratio,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades
        }

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
        return cls.from_dict(data)

    def __str__(self) -> str:
        """String representation of performance metrics"""
        metrics_str = [
            f"Sharpe Ratio: {self.sharpe_ratio:.2f}",
            f"Win Rate: {self.win_rate:.2%}",
            f"Profit Factor: {self.profit_factor:.2f}",
            f"Max Drawdown: {self.max_drawdown:.2%}",
            f"Volatility: {self.volatility:.2%}"
        ]
        
        if self.avg_return is not None:
            metrics_str.append(f"Average Return: {self.avg_return:.2%}")
        if self.sortino_ratio is not None:
            metrics_str.append(f"Sortino Ratio: {self.sortino_ratio:.2f}")
        if self.calmar_ratio is not None:
            metrics_str.append(f"Calmar Ratio: {self.calmar_ratio:.2f}")
        if self.total_trades is not None:
            metrics_str.append(f"Total Trades: {self.total_trades}")
            metrics_str.append(f"Winning Trades: {self.winning_trades}")
            metrics_str.append(f"Losing Trades: {self.losing_trades}")
            
        return "\n".join(metrics_str)

    def compare_with(self, other: 'PerformanceMetrics') -> Dict[str, float]:
        """
        Compare this instance with another PerformanceMetrics instance.
        
        Args:
            other (PerformanceMetrics): Instance to compare with
            
        Returns:
            Dict[str, float]: Dictionary of metric differences
        """
        return {
            'sharpe_ratio_diff': self.sharpe_ratio - other.sharpe_ratio,
            'win_rate_diff': self.win_rate - other.win_rate,
            'profit_factor_diff': self.profit_factor - other.profit_factor,
            'max_drawdown_diff': self.max_drawdown - other.max_drawdown,
            'volatility_diff': self.volatility - other.volatility
        }