"""
Core trading system implementation coordinating all trading activities.
"""

# Standard Library Imports
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json
import os
from concurrent.futures import ThreadPoolExecutor
import queue

# Third-Party Imports
import numpy as np
import pandas as pd

# Local Imports
from ..core.database_handler import DatabaseHandler
from ..core.performance_metrics import PerformanceMetrics
from ..indicators.calculator import IndicatorCalculator
from ..analysis.marketcondition import MarketConditionAnalyzer
from ..analysis.confidencescore import ConfidenceScoreCalculator
from ..models.mlmodel import MachineLearningModel
from ..risk.manager import RiskManagement
from ..risk.positionsizing import PositionSizing
from ..risk.tpsl import TakeProfitStopLoss
from ..signal.generator import SignalGenerator
from ..backtesting.enhancedbacktest import Backtest
from .order import Order, OrderStatus, OrderType, Position, Portfolio
from ..risk.exposure import ExposureTracker
from ..signal.quality import SignalQualityMonitor

# Set up logging
logger = logging.getLogger(__name__)

class TradingSystem:
    """
    Main trading system coordinating all trading activities including signal generation,
    risk management, position sizing, and order execution.
    """
    
    def __init__(self, 
                 db_path: str,
                 initial_capital: float,
                 max_risk_per_trade: float,
                 config: Dict[str, Any]):
        """
        Initialize the trading system.
        
        Args:
            db_path: Path to market data database
            initial_capital: Initial trading capital
            max_risk_per_trade: Maximum risk per trade as decimal
            config: Configuration dictionary
        """
        try:
            # Initialize components
            self.db_handler = DatabaseHandler(db_path)
            self.portfolio = Portfolio(initial_capital)
            self.signal_generator = SignalGenerator(db_path, config['model_path'])
            self.risk_manager = RiskManagement(
                backtest_results=pd.DataFrame(),  # Will be updated after first backtest
                market_conditions_file=config['market_conditions_file'],
                db_path=db_path
            )
            self.position_sizer = PositionSizing(
                initial_capital,
                max_risk_per_trade,
                config['min_risk_reward_ratio'],
                config['volatility_adjustment_factor']
            )
            self.tp_sl_manager = TakeProfitStopLoss(
                config['atr_multiplier'],
                config['confidence_factor'],
                config['volatility_factor']
            )
            
            # Initialize state
            self.active_orders: Dict[str, Order] = {}
            self.order_history: List[Order] = []
            self.market_data: Dict[str, pd.DataFrame] = {}
            self.is_running: bool = False
            self.config = config
            
            # Initialize queues for real-time processing
            self.market_data_queue = queue.Queue()
            self.signal_queue = queue.Queue()
            self.order_queue = queue.Queue()
            
            # Initialize thread pool
            self.executor = ThreadPoolExecutor(max_workers=config.get('max_workers', 4))
            self.exposure_tracker = ExposureTracker()
            self.signal_monitor = SignalQualityMonitor(
            lookback_period=config.get('signal_lookback_period', 50),
            degradation_threshold=config.get('degradation_threshold', 0.3),
            min_confidence=config.get('min_signal_confidence', 0.6)
    )
            
            logger.info("Trading system initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {e}")
            raise

    def start(self) -> None:
        """Start the trading system and begin processing"""
        try:
            logger.info("Starting trading system...")
            self.is_running = True
            
            # Start processing threads
            self.executor.submit(self._process_market_data)
            self.executor.submit(self._process_signals)
            self.executor.submit(self._process_orders)
            
            logger.info("Trading system started successfully")
            
        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
            self.is_running = False
            raise

    def stop(self) -> None:
        """Stop the trading system and cleanup"""
        try:
            logger.info("Stopping trading system...")
            self.is_running = False
            
            # Close all positions
            self._close_all_positions()
            
            # Cancel all pending orders
            self._cancel_all_orders()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            logger.info("Trading system stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping trading system: {e}")
            raise

    def add_market_data(self, symbol: str, data: pd.DataFrame) -> None:
        """
        Add new market data for processing.
        
        Args:
            symbol: Trading symbol
            data: Market data DataFrame
        """
        try:
            self.market_data_queue.put((symbol, data))
        except Exception as e:
            logger.error(f"Error adding market data: {e}")
            raise

    def _process_market_data(self) -> None:
        """Process incoming market data"""
        while self.is_running:
            try:
                symbol, data = self.market_data_queue.get(timeout=1)
                
                # Update stored market data
                self.market_data[symbol] = data
                
                # Update portfolio positions
                if symbol in self.portfolio.positions:
                    current_price = data['close'].iloc[-1]
                    self.portfolio.positions[symbol].update_price(current_price)
                
                # Generate new signals
                self._generate_signals(symbol, data)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing market data: {e}")

    def _generate_signals(self, symbol: str, data: pd.DataFrame) -> None:
        """Generate trading signals from market data"""
        try:
            # Get market condition and confidence score
            market_condition = self.signal_generator.get_market_condition(data)
            confidence_score = self.signal_generator.calculate_confidence_score(
                data, market_condition
            )
            
            # Generate signal
            signal = self.signal_generator.check_signal(data)
        
            # Add this new section:
            indicator_values = self.signal_generator.get_indicator_values(data)
            signal_valid, quality_metrics = self.signal_monitor.evaluate_signal_quality(
                current_signal=signal,
                confidence_score=confidence_score,
                market_condition=market_condition,
                indicator_values=indicator_values
            )
            
            if signal != 0 and signal_valid:  # Changed condition to check signal validity
                self.signal_queue.put({
                'symbol': symbol,
                'signal': signal,
                'market_condition': market_condition,
                'confidence_score': confidence_score,
                'data': data,
                'quality_metrics': quality_metrics  # Added quality metrics
            })
        except Exception as e:
            logger.error(f"Error generating signals: {e}")

    def _process_signals(self) -> None:
        """
        Process generated trading signals and create orders.
        
        Continuously monitors the signal queue and creates appropriate orders
        based on trading signals, risk management, and position sizing rules.
        """
        while self.is_running:
            try:
                # Get signal data with timeout
                signal_data = self.signal_queue.get(timeout=1)
                
                symbol = signal_data['symbol']
                signal = signal_data['signal']
                confidence_score = signal_data['confidence_score']
                market_data = signal_data['data']
                
                # Skip if we already have a position in this symbol
                if symbol in self.portfolio.positions:
                    logger.info(f"Skipping signal for {symbol} - position already exists")
                    continue
                
                # Get current market data
                current_price = market_data['close'].iloc[-1]
                
                # Calculate position size based on risk parameters
                position_size = self.position_sizer.calculate_position_size(
                    entry_price=current_price,
                    confidence_score=confidence_score,
                    current_volatility=market_data['ATR'].iloc[-1],
                    average_volatility=market_data['ATR'].mean(),
                    market_data={
                        'high_volatility': market_data['ATR'].iloc[-1] > market_data['ATR'].mean(),
                        'strong_trend': abs(market_data['RSI'].iloc[-1] - 50) > 20
                    }
                )
                
                if position_size <= 0:
                    logger.info(f"Invalid position size calculated for {symbol}")
                    continue
                
                # Calculate stop loss and take profit levels
                stop_loss, take_profit = self.tp_sl_manager.calculate_tp_sl(
                    entry_price=current_price,
                    atr=market_data['ATR'].iloc[-1],
                    risk_parameters=self.config['risk_parameters'],
                    trade_direction='long' if signal > 0 else 'short',
                    confidence_score=confidence_score,
                    current_volatility=market_data['ATR'].iloc[-1],
                    average_volatility=market_data['ATR'].mean()
                )
                
                # Create and submit order
                order = Order(
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    direction=signal,
                    quantity=position_size,
                    entry_price=current_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    timestamp=datetime.now()
                )
                
                self.order_queue.put(order)
                logger.info(f"Created order for {symbol}: {order}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing signals: {e}")

    def _process_orders(self) -> None:
        """Process the order queue and execute trades"""
        while self.is_running:
            try:
                # Get order with timeout
                order = self.order_queue.get(timeout=1)
                
                # Final risk check before execution
                if not self._validate_risk_limits(order):
                    order.status = OrderStatus.REJECTED
                    self.order_history.append(order)
                    logger.warning(f"Order rejected due to risk limits: {order}")
                    continue
                
                # Execute the order
                filled_order = self._execute_order(order)
                
                if filled_order.status == OrderStatus.FILLED:
                    # Create and add position to portfolio
                    position = Position(
                        symbol=filled_order.symbol,
                        quantity=filled_order.quantity * filled_order.direction,
                        entry_price=filled_order.fill_price,
                        current_price=filled_order.fill_price,
                        stop_loss=filled_order.stop_loss,
                        take_profit=filled_order.take_profit
                    )
                    self.portfolio.add_position(position)
                    
                    # Update exposure tracking with all current positions
                    current_positions = {
                        s: p.quantity * p.current_price 
                        for s, p in self.portfolio.positions.items()
                    }
                    self.exposure_tracker.update_exposures(current_positions)
                
                # Record order
                self.order_history.append(filled_order)
                logger.info(f"Processed order: {filled_order}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing orders: {e}")

    def _validate_risk_limits(self, order: Order) -> bool:
        """
        Validate order against risk management rules.
        
        Args:
            order: Order to validate

        Returns:
            bool: True if order passes risk checks, False otherwise
        """
        try:
            # Calculate position value
            position_value = abs(order.quantity * order.entry_price)
            
            # Check maximum position size
            if position_value > self.portfolio.current_capital * self.config['max_position_size']:
                logger.warning(f"Order exceeds maximum position size: {order}")
                return False
            
            # Check portfolio heat
            current_heat = self.portfolio.get_metrics()['total_value'] / self.portfolio.initial_capital
            if current_heat > self.config['max_portfolio_heat']:
                logger.warning(f"Order would exceed maximum portfolio heat: {order}")
                return False
            
            # Check correlation with existing positions
            if not self._check_correlation_limits(order):
                logger.warning(f"Order would exceed correlation limits: {order}")
                return False
                
            # Check sector and asset class exposure limits
            if not self.exposure_tracker.check_exposure_limits(
                new_position=position_value,
                symbol=order.symbol,
                max_sector_exposure=self.config.get('max_sector_exposure', 0.3),
                max_asset_class_exposure=self.config.get('max_asset_class_exposure', 0.6)
            ):
                logger.warning(f"Order would exceed exposure limits: {order}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating risk limits: {e}")
            return False

    def _check_correlation_limits(self, order: Order) -> bool:
        """
        Check if adding the position would exceed correlation limits.
        
        Args:
            order: Order to check

        Returns:
            bool: True if correlation limits are satisfied, False otherwise
        """
        try:
            if not self.portfolio.positions:
                return True
                
            # Get historical data for correlation calculation
            order_data = self.market_data[order.symbol]['close']
            
            correlations = []
            for pos in self.portfolio.positions.values():
                pos_data = self.market_data[pos.symbol]['close']
                correlation = order_data.corr(pos_data)
                correlations.append(abs(correlation))
            
            # Check average correlation
            avg_correlation = np.mean(correlations)
            return avg_correlation <= self.config['max_correlation']
            
        except Exception as e:
            logger.error(f"Error checking correlation limits: {e}")
            return False

    def _execute_order(self, order: Order) -> Order:
        """
        Execute a trading order.
        
        Args:
            order: Order to execute

        Returns:
            Order: Executed order with updated status and fill information
        """
        try:
            # In real implementation, this would interact with broker/exchange API
            # For now, simulate immediate fill at requested price
            order.status = OrderStatus.FILLED
            order.fill_price = order.entry_price
            logger.info(f"Executed order: {order}")
            return order
            
        except Exception as e:
            logger.error(f"Error executing order: {e}")
            order.status = OrderStatus.REJECTED
            return order

    def _close_all_positions(self) -> None:
        """Close all open positions in the portfolio."""
        try:
            for symbol in list(self.portfolio.positions.keys()):
                position = self.portfolio.positions[symbol]
                
                # Create closing order
                order = Order(
                    symbol=symbol,
                    order_type=OrderType.MARKET,
                    direction=-np.sign(position.quantity),  # Opposite direction to close
                    quantity=abs(position.quantity),
                    entry_price=position.current_price
                )
                
                # Execute closing order
                filled_order = self._execute_order(order)
                
                if filled_order.status == OrderStatus.FILLED:
                    self.portfolio.remove_position(symbol)
                    
            logger.info("Closed all positions")
            
        except Exception as e:
            logger.error(f"Error closing positions: {e}")

    def _cancel_all_orders(self) -> None:
        """Cancel all pending orders."""
        try:
            while not self.order_queue.empty():
                order = self.order_queue.get()
                order.status = OrderStatus.CANCELLED
                self.order_history.append(order)
                
            logger.info("Cancelled all pending orders")
            
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current trading system metrics.
        
        Returns:
            Dict containing various performance and risk metrics
        """
        try:
            metrics = self.portfolio.get_metrics()
            
            # Add additional metrics
            metrics.update({
                'win_rate': self._calculate_win_rate(),
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'max_drawdown': self._calculate_max_drawdown(),
                'active_positions': len(self.portfolio.positions),
                'pending_orders': self.order_queue.qsize()
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}

    def _calculate_win_rate(self) -> float:
        """Calculate win rate from closed positions."""
        if not self.portfolio.closed_positions:
            return 0.0
        winning_trades = sum(1 for pos in self.portfolio.closed_positions 
                           if pos.realized_pnl > 0)
        return winning_trades / len(self.portfolio.closed_positions)

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trading history."""
        if len(self.portfolio.history) < 2:
            return 0.0
            
        returns = pd.Series([h['position'].realized_pnl 
                           for h in self.portfolio.history 
                           if h['type'] == 'position_closed'])
        
        if len(returns) < 2:
            return 0.0
            
        return np.sqrt(252) * returns.mean() / (returns.std() + 1e-6)

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from portfolio history."""
        if not self.portfolio.history:
            return 0.0
            
        equity_curve = [h.get('portfolio_value', 0) for h in self.portfolio.history]
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            max_dd = max(max_dd, dd)
            
        return max_dd
