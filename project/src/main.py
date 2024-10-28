"""
Main entry point for the trading system.
"""

# Standard Library Imports
import argparse
import logging
import sys
from pathlib import Path

# Local Imports
from core.config import ConfigurationManager
from trading.system import TradingSystem
from utils.logging import setup_logging

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument(
        '--config', 
        type=str, 
        default='config',
        help='Path to configuration directory'
    )
    parser.add_argument(
        '--env', 
        type=str, 
        default='development',
        choices=['development', 'production', 'test'],
        help='Environment to run in'
    )
    parser.add_argument(
        '--log-level', 
        type=str, 
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level'
    )
    return parser.parse_args()

def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Set up logging
        logger = setup_logging(
            name="trading_system",
            log_dir="logs",
            log_level=getattr(logging, args.log_level)
        )
        
        # Load configuration
        config_manager = ConfigurationManager(
            config_path=args.config,
            env=args.env
        )
        
        # Initialize trading system
        trading_system = TradingSystem(
            db_path=config_manager.get('database.path'),
            initial_capital=config_manager.get('trading.initial_capital'),
            max_risk_per_trade=config_manager.get('risk.max_risk_per_trade'),
            config=config_manager.to_dict()
        )
        
        # Start trading system
        trading_system.start()
        
        # Wait for shutdown signal
        try:
            while True:
                pass
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            trading_system.stop()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
        
    sys.exit(0)

if __name__ == "__main__":
    main()