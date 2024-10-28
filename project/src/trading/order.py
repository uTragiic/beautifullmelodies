"""
Order management and position tracking components for the trading system.
"""

# Standard Library Imports
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Union
import logging

# Third-Party Imports
import numpy as np

# Set up logging
logger = logging.getLogger(__name__)

class OrderStatus(Enum):
    """Order status enumeration"""
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    PARTIALLY_FILLED = "partially_filled"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

@dataclass
class Order:
    """
    Represents a trading order with all relevant information.
    
    Attributes:
        symbol: Trading symbol/ticker
        order_type: Type of order (market, limit, etc.)
        direction: 1 for buy, -1 for sell
        quantity: Number of units to trade
        entry_price: Desired entry price (None for market orders)
        stop_loss: Stop loss price level
        take_profit: Take profit price level
        status: Current order status
        fill_price: Actual fill price once executed
        timestamp: Order creation time
        id: Unique order identifier
    """
    symbol: str
    order_type: OrderType
    direction: int
    quantity: float
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    fill_price: Optional[float] = None
    timestamp: datetime = datetime.now()
    id: Optional[str] = None
    
    def __post_init__(self):
        """Validate order parameters after initialization"""
        if self.direction not in [-1, 1]:
            raise ValueError("Direction must be 1 (buy) or -1 (sell)")
        if self.quantity <= 0:
            raise ValueError("Quantity must be positive")
        if self.order_type != OrderType.MARKET and self.entry_price is None:
            raise ValueError("Entry price required for non-market orders")

@dataclass
class Position:
    """
    Represents an open trading position.
    
    Attributes:
        symbol: Trading symbol/ticker
        quantity: Position size (positive for long, negative for short)
        entry_price: Average entry price
        current_price: Current market price
        stop_loss: Current stop loss level
        take_profit: Current take profit level
        unrealized_pnl: Current unrealized profit/loss
        realized_pnl: Realized profit/loss from partial closes
        timestamp: Position open time
    """
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    timestamp: datetime = datetime.now()

    def update_price(self, price: float) -> None:
        """Update position with new market price"""
        self.current_price = price
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity

    def update_stops(self, stop_loss: Optional[float] = None, take_profit: Optional[float] = None) -> None:
        """Update stop loss and take profit levels"""
        if stop_loss is not None:
            self.stop_loss = stop_loss
        if take_profit is not None:
            self.take_profit = take_profit

class Portfolio:
    """
    Manages a collection of trading positions and overall portfolio metrics.
    """
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.history: List[Dict] = []
        
    def add_position(self, position: Position) -> None:
        """Add a new position to the portfolio"""
        try:
            if position.symbol in self.positions:
                # Average in with existing position
                existing = self.positions[position.symbol]
                total_quantity = existing.quantity + position.quantity
                new_entry = (existing.entry_price * existing.quantity + 
                           position.entry_price * position.quantity) / total_quantity
                existing.quantity = total_quantity
                existing.entry_price = new_entry
                existing.update_price(position.current_price)
            else:
                self.positions[position.symbol] = position
            
            # Record to history
            self.history.append({
                'timestamp': datetime.now(),
                'type': 'position_added',
                'position': position
            })
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            raise

    def remove_position(self, symbol: str) -> Optional[Position]:
        """Remove a position from the portfolio"""
        try:
            if symbol in self.positions:
                position = self.positions.pop(symbol)
                self.closed_positions.append(position)
                self.current_capital += position.unrealized_pnl
                
                # Record to history
                self.history.append({
                    'timestamp': datetime.now(),
                    'type': 'position_closed',
                    'position': position
                })
                return position
            return None
        except Exception as e:
            logger.error(f"Error removing position: {e}")
            raise

    def update_prices(self, prices: Dict[str, float]) -> None:
        """Update all positions with new market prices"""
        try:
            for symbol, price in prices.items():
                if symbol in self.positions:
                    self.positions[symbol].update_price(price)
        except Exception as e:
            logger.error(f"Error updating prices: {e}")
            raise

    def get_total_value(self) -> float:
        """Calculate total portfolio value including cash and positions"""
        return self.current_capital + sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_metrics(self) -> Dict[str, float]:
        """Calculate current portfolio metrics"""
        try:
            total_value = self.get_total_value()
            return {
                'total_value': total_value,
                'unrealized_pnl': sum(pos.unrealized_pnl for pos in self.positions.values()),
                'realized_pnl': sum(pos.realized_pnl for pos in self.closed_positions),
                'return_pct': (total_value - self.initial_capital) / self.initial_capital * 100,
                'num_positions': len(self.positions)
            }
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            raise