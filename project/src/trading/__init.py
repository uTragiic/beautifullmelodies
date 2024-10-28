"""
Trading module providing core trading system functionality.

This module implements the main trading engine that coordinates signal generation,
risk management, position sizing, and order execution.
"""

from .system import TradingSystem
from .order import Order, OrderStatus, OrderType, Position, Portfolio

__all__ = [
    'TradingSystem',
    'Order',
    'OrderStatus',
    'OrderType',
    'Position',
    'Portfolio'
]