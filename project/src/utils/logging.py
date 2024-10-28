"""
Advanced logging configuration for the trading system.
Provides centralized logging with different levels of detail,
custom formatting, log rotation, and performance monitoring.
"""

# Standard Library Imports
import logging
import logging.handlers
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json
import os
import sys
import time
from functools import wraps

# Third-Party Imports
from pythonjsonlogger import jsonlogger

# Set up custom log levels
SIGNAL = 25  # Between INFO and WARNING
TRADE = 26
PERFORMANCE = 27

# Add custom levels to logging
logging.addLevelName(SIGNAL, 'SIGNAL')
logging.addLevelName(TRADE, 'TRADE')
logging.addLevelName(PERFORMANCE, 'PERFORMANCE')

class TradeLogger:
    """
    Custom logger class for trading system with additional functionality
    for tracking trades, signals, and performance metrics.
    """
    
    def __init__(self, 
                 name: str,
                 log_dir: str = "logs",
                 log_level: int = logging.INFO,
                 max_bytes: int = 10_000_000,  # 10MB
                 backup_count: int = 5,
                 performance_tracking: bool = True):
        """
        Initialize the trade logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
            log_level: Minimum logging level
            max_bytes: Maximum size of each log file
            backup_count: Number of backup files to keep
            performance_tracking: Whether to track performance metrics
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_level = log_level
        self.performance_tracking = performance_tracking
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Remove existing handlers
        self.logger.handlers.clear()
        
        # Set up handlers
        self._setup_console_handler()
        self._setup_file_handlers(max_bytes, backup_count)
        
        if performance_tracking:
            self.performance_metrics: Dict[str, Any] = {
                'start_time': time.time(),
                'signal_count': 0,
                'trade_count': 0,
                'error_count': 0,
                'warning_count': 0
            }
    
    def _setup_console_handler(self) -> None:
        """Set up console logging handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        # Custom formatter for console output
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, max_bytes: int, backup_count: int) -> None:
        """
        Set up file handlers for different log types with rotation.
        
        Args:
            max_bytes: Maximum size of each log file
            backup_count: Number of backup files to keep
        """
        # Main log file (JSON format)
        main_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}.json",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        main_handler.setLevel(self.log_level)
        json_formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s %(extra)s'
        )
        main_handler.setFormatter(json_formatter)
        self.logger.addHandler(main_handler)
        
        # Error log file
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_error.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s\n'
            'Exception: %(exc_info)s\n'
            'Stack Trace: %(stack_info)s\n'
        )
        error_handler.setFormatter(error_formatter)
        self.logger.addHandler(error_handler)
        
        # Trade log file
        trade_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / f"{self.name}_trades.log",
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        trade_handler.setLevel(TRADE)
        trade_formatter = logging.Formatter(
            '%(asctime)s - %(message)s'
        )
        trade_handler.setFormatter(trade_formatter)
        self.logger.addHandler(trade_handler)
    
    def log_signal(self, signal_data: Dict[str, Any]) -> None:
        """
        Log a trading signal.
        
        Args:
            signal_data: Dictionary containing signal information
        """
        if self.performance_tracking:
            self.performance_metrics['signal_count'] += 1
            
        self.logger.log(
            SIGNAL,
            f"Signal generated: {json.dumps(signal_data)}",
            extra={'signal_data': signal_data}
        )
    
    def log_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Log a trade execution.
        
        Args:
            trade_data: Dictionary containing trade information
        """
        if self.performance_tracking:
            self.performance_metrics['trade_count'] += 1
            
        self.logger.log(
            TRADE,
            f"Trade executed: {json.dumps(trade_data)}",
            extra={'trade_data': trade_data}
        )
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """
        Log performance metrics.
        
        Args:
            metrics: Dictionary containing performance metrics
        """
        self.logger.log(
            PERFORMANCE,
            f"Performance metrics: {json.dumps(metrics)}",
            extra={'performance_data': metrics}
        )
    
    def log_error(self, error: Exception, context: str = "") -> None:
        """
        Log an error with full stack trace and context.
        
        Args:
            error: Exception object
            context: Additional context information
        """
        if self.performance_tracking:
            self.performance_metrics['error_count'] += 1
            
        self.logger.exception(
            f"Error in {context}: {str(error)}",
            exc_info=error
        )
    
    def log_warning(self, message: str, context: Dict[str, Any] = None) -> None:
        """
        Log a warning message with context.
        
        Args:
            message: Warning message
            context: Additional context information
        """
        if self.performance_tracking:
            self.performance_metrics['warning_count'] += 1
            
        self.logger.warning(
            message,
            extra={'context': context or {}}
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get current logger performance metrics.
        
        Returns:
            Dictionary containing logger performance metrics
        """
        if not self.performance_tracking:
            return {}
            
        current_time = time.time()
        run_time = current_time - self.performance_metrics['start_time']
        
        metrics = self.performance_metrics.copy()
        metrics.update({
            'run_time': run_time,
            'signals_per_hour': (metrics['signal_count'] / run_time) * 3600,
            'trades_per_hour': (metrics['trade_count'] / run_time) * 3600,
            'error_rate': (metrics['error_count'] / max(1, metrics['signal_count'] + metrics['trade_count'])) * 100
        })
        
        return metrics

def log_execution_time(logger: TradeLogger):
    """
    Decorator to log function execution time.
    
    Args:
        logger: TradeLogger instance to use for logging
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                logger.logger.debug(
                    f"Function {func.__name__} executed in {execution_time:.4f} seconds",
                    extra={
                        'function': func.__name__,
                        'execution_time': execution_time
                    }
                )
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                logger.log_error(
                    e,
                    f"Error in {func.__name__} after {execution_time:.4f} seconds"
                )
                raise
        return wrapper
    return decorator

def setup_logging(
    name: str,
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    max_bytes: int = 10_000_000,
    backup_count: int = 5,
    performance_tracking: bool = True
) -> TradeLogger:
    """
    Set up logging configuration for the trading system.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        log_level: Minimum logging level
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
        performance_tracking: Whether to track performance metrics
        
    Returns:
        Configured TradeLogger instance
    """
    return TradeLogger(
        name=name,
        log_dir=log_dir,
        log_level=log_level,
        max_bytes=max_bytes,
        backup_count=backup_count,
        performance_tracking=performance_tracking
    )