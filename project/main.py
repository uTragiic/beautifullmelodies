import logging
import argparse
from pathlib import Path
from datetime import datetime

from src.core.config import ConfigurationManager
from src.trading.system import TradingSystem
from src.utils.logging import setup_logging

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading System')
    parser.add_argument(
        '--config-dir',
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
        '--symbols',
        type=str,
        nargs='+',
        help='Specific symbols to trade (optional)'
    )
    return parser.parse_args()

def main():
    """Main entry point for trading system"""
    # Parse arguments
    args = parse_arguments()
    
    try:
        # Set up logging
        logger = setup_logging(
            name="trading_system",
            log_dir="logs",
            log_level=logging.INFO
        )
        
        # Load configuration
        config_manager = ConfigurationManager(
            config_path=args.config_dir,
            env=args.env
        )
        
        # Initialize trading system
        trading_system = TradingSystem(
            db_path=config_manager.get('database.path'),
            initial_capital=1000000.0,  # $1M initial capital
            max_risk_per_trade=config_manager.get('risk.max_risk_per_trade'),
            config=config_manager.to_dict()
        )
        
        # Set specific symbols if provided
        if args.symbols:
            trading_system.set_symbols(args.symbols)
            
        # Start trading system
        trading_system.start()
        
        logger.info("Trading system started successfully")
        
        # Keep running until interrupted
        try:
            while True:
                trading_system.update()  # Regular updates/monitoring
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
            trading_system.stop()
            
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()