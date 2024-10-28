"""
Validation utilities for the trading system.
Provides comprehensive validation for data structures, parameters,
and trading objects to ensure system integrity and safety.
"""

# Standard Library Imports
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import re
from decimal import Decimal

# Third-Party Imports
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

# Local Imports
from ..trading.order import Order, OrderType, OrderStatus
from ..core.types import MarketRegime

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

def validate_dataframe(
    df: pd.DataFrame,
    required_columns: List[str],
    min_rows: int = 100,
    check_timestamps: bool = True
) -> None:
    """
    Validate a market data DataFrame.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        check_timestamps: Whether to validate timestamp index
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check if DataFrame is empty
        if df.empty:
            raise ValidationError("DataFrame is empty")
            
        # Check minimum rows
        if len(df) < min_rows:
            raise ValidationError(f"DataFrame has insufficient rows: {len(df)} < {min_rows}")
            
        # Check required columns
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValidationError(f"Missing required columns: {missing_columns}")
            
        # Check for numeric data in price and volume columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in df.columns and not is_numeric_dtype(df[col]):
                raise ValidationError(f"Column {col} must be numeric")
                
        # Validate timestamps if required
        if check_timestamps:
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValidationError("DataFrame index must be DatetimeIndex")
            
            # Check for duplicate timestamps
            if df.index.duplicated().any():
                raise ValidationError("DataFrame contains duplicate timestamps")
            
            # Check for chronological order
            if not df.index.is_monotonic_increasing:
                raise ValidationError("DataFrame timestamps are not in chronological order")
                
        # Check for NaN values
        nan_counts = df[required_columns].isna().sum()
        if nan_counts.any():
            raise ValidationError(f"Found NaN values in columns: {nan_counts[nan_counts > 0]}")
            
        # Validate price integrity
        if all(col in df.columns for col in ['high', 'low', 'open', 'close']):
            if not (df['high'] >= df['low']).all():
                raise ValidationError("High prices must be greater than or equal to low prices")
            if not ((df['open'] >= df['low']) & (df['open'] <= df['high'])).all():
                raise ValidationError("Open prices must be between high and low prices")
            if not ((df['close'] >= df['low']) & (df['close'] <= df['high'])).all():
                raise ValidationError("Close prices must be between high and low prices")
                
    except ValidationError as e:
        raise e
    except Exception as e:
        raise ValidationError(f"Unexpected error validating DataFrame: {str(e)}")

def validate_parameters(
    parameters: Dict[str, Any],
    required_params: Dict[str, type],
    bounds: Optional[Dict[str, tuple]] = None
) -> None:
    """
    Validate configuration parameters.
    
    Args:
        parameters: Dictionary of parameters to validate
        required_params: Dictionary of required parameters and their types
        bounds: Optional dictionary of parameter bounds (min, max)
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Check for missing parameters
        missing_params = set(required_params.keys()) - set(parameters.keys())
        if missing_params:
            raise ValidationError(f"Missing required parameters: {missing_params}")
            
        # Validate parameter types
        for param_name, param_type in required_params.items():
            value = parameters[param_name]
            if not isinstance(value, param_type):
                raise ValidationError(
                    f"Parameter {param_name} must be of type {param_type.__name__}, "
                    f"got {type(value).__name__}"
                )
                
        # Validate bounds if provided
        if bounds:
            for param_name, (min_val, max_val) in bounds.items():
                if param_name in parameters:
                    value = parameters[param_name]
                    if not min_val <= value <= max_val:
                        raise ValidationError(
                            f"Parameter {param_name} must be between {min_val} and {max_val}, "
                            f"got {value}"
                        )
                        
    except ValidationError as e:
        raise e
    except Exception as e:
        raise ValidationError(f"Unexpected error validating parameters: {str(e)}")

def validate_order(order: Order) -> None:
    """
    Validate a trading order.
    
    Args:
        order: Order to validate
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Validate symbol format
        if not re.match(r'^[A-Z]{1,5}$', order.symbol):
            raise ValidationError(f"Invalid symbol format: {order.symbol}")
            
        # Validate order type
        if not isinstance(order.order_type, OrderType):
            raise ValidationError(f"Invalid order type: {order.order_type}")
            
        # Validate direction
        if order.direction not in [-1, 1]:
            raise ValidationError(f"Invalid order direction: {order.direction}")
            
        # Validate quantity
        if not isinstance(order.quantity, (int, float, Decimal)) or order.quantity <= 0:
            raise ValidationError(f"Invalid order quantity: {order.quantity}")
            
        # Validate prices for limit and stop orders
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.entry_price is None or order.entry_price <= 0:
                raise ValidationError(f"Invalid entry price for {order.order_type}: {order.entry_price}")
                
        # Validate stop loss and take profit if provided
        if order.stop_loss is not None:
            if order.stop_loss <= 0:
                raise ValidationError(f"Invalid stop loss: {order.stop_loss}")
            if order.direction == 1 and order.stop_loss >= order.entry_price:
                raise ValidationError("Stop loss must be below entry price for long orders")
            if order.direction == -1 and order.stop_loss <= order.entry_price:
                raise ValidationError("Stop loss must be above entry price for short orders")
                
        if order.take_profit is not None:
            if order.take_profit <= 0:
                raise ValidationError(f"Invalid take profit: {order.take_profit}")
            if order.direction == 1 and order.take_profit <= order.entry_price:
                raise ValidationError("Take profit must be above entry price for long orders")
            if order.direction == -1 and order.take_profit >= order.entry_price:
                raise ValidationError("Take profit must be below entry price for short orders")
                
        # Validate status
        if not isinstance(order.status, OrderStatus):
            raise ValidationError(f"Invalid order status: {order.status}")
            
        # Validate timestamp
        if not isinstance(order.timestamp, datetime):
            raise ValidationError(f"Invalid timestamp type: {type(order.timestamp)}")
        if order.timestamp > datetime.now():
            raise ValidationError("Order timestamp cannot be in the future")
            
    except ValidationError as e:
        raise e
    except Exception as e:
        raise ValidationError(f"Unexpected error validating order: {str(e)}")

def validate_position_size(
    size: float,
    account_value: float,
    max_position_pct: float = 0.2
) -> bool:
    """
    Validate position size against account limits.
    
    Args:
        size: Position size in currency units
        account_value: Total account value
        max_position_pct: Maximum position size as percentage of account
        
    Returns:
        bool: True if position size is valid
    """
    try:
        if not isinstance(size, (int, float, Decimal)) or size <= 0:
            return False
        if size > account_value * max_position_pct:
            return False
        return True
    except Exception:
        return False

def validate_risk_reward_ratio(
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    min_ratio: float = 2.0
) -> bool:
    """
    Validate risk-reward ratio for a trade.
    
    Args:
        entry_price: Entry price
        stop_loss: Stop loss price
        take_profit: Take profit price
        min_ratio: Minimum required risk-reward ratio
        
    Returns:
        bool: True if risk-reward ratio is valid
    """
    try:
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        if risk == 0:
            return False
        return (reward / risk) >= min_ratio
    except Exception:
        return False

def validate_market_regime(regime: MarketRegime) -> bool:
    """
    Validate market regime enumeration.
    
    Args:
        regime: Market regime to validate
        
    Returns:
        bool: True if regime is valid
    """
    try:
        return isinstance(regime, MarketRegime)
    except Exception:
        return False

def validate_indicator_parameters(
    parameters: Dict[str, Any],
    indicator_type: str
) -> None:
    """
    Validate technical indicator parameters.
    
    Args:
        parameters: Dictionary of indicator parameters
        indicator_type: Type of indicator
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Common validations for all indicators
        if 'window' in parameters:
            window = parameters['window']
            if not isinstance(window, int) or window <= 0:
                raise ValidationError(f"Invalid window parameter: {window}")
                
        # Indicator-specific validations
        if indicator_type == 'RSI':
            if not 1 <= parameters.get('window', 14) <= 100:
                raise ValidationError("RSI window must be between 1 and 100")
                
        elif indicator_type == 'MACD':
            fast = parameters.get('fast_period', 12)
            slow = parameters.get('slow_period', 26)
            if not (1 <= fast < slow):
                raise ValidationError("MACD fast period must be less than slow period")
                
        elif indicator_type == 'ATR':
            if parameters.get('window', 14) < 1:
                raise ValidationError("ATR window must be positive")
                
    except ValidationError as e:
        raise e
    except Exception as e:
        raise ValidationError(f"Unexpected error validating indicator parameters: {str(e)}")

def validate_backtest_parameters(
    start_date: datetime,
    end_date: datetime,
    initial_capital: float,
    params: Dict[str, Any]
) -> None:
    """
    Validate backtesting parameters.
    
    Args:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Initial capital
        params: Additional backtest parameters
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        # Validate dates
        if start_date >= end_date:
            raise ValidationError("Start date must be before end date")
        if end_date > datetime.now():
            raise ValidationError("End date cannot be in the future")
            
        # Validate initial capital
        if not isinstance(initial_capital, (int, float, Decimal)) or initial_capital <= 0:
            raise ValidationError(f"Invalid initial capital: {initial_capital}")
            
        # Validate additional parameters
        required_params = {
            'transaction_costs': float,
            'slippage': float,
            'position_size_limit': float
        }
        
        validate_parameters(params, required_params)
        
        # Validate specific parameter bounds
        if not 0 <= params['transaction_costs'] <= 0.01:
            raise ValidationError("Transaction costs must be between 0 and 1%")
        if not 0 <= params['slippage'] <= 0.01:
            raise ValidationError("Slippage must be between 0 and 1%")
        if not 0 < params['position_size_limit'] <= 1:
            raise ValidationError("Position size limit must be between 0 and 100%")
            
    except ValidationError as e:
        raise e
    except Exception as e:
        raise ValidationError(f"Unexpected error validating backtest parameters: {str(e)}")