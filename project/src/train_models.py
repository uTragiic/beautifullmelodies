"""
Main training script for stock trading models.
Handles model training across different market segments.
"""

import sys
from pathlib import Path

# Add the project root to PYTHONPATH
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import logging
import json
import argparse
from datetime import datetime, timedelta
import pandas as pd

from src.universe import UniverseManager, UniverseClusterer, UniverseFilter, FilterConfig
from src.training import ModelTrainer, TrainingScheduler, ModelValidator
from src.models.mlmodel import MachineLearningModel
from src.core.database_handler import DatabaseHandler
from src.utils.logging import setup_logging

def train_models(config_path: str):
    """
    Main training function coordinating all components.
    
    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    with open(config_path) as f:
        config = json.load(f)
        
    # Setup directories
    base_dir = Path(config['base_dir'])
    model_dir = base_dir / "models"
    cache_dir = base_dir / "cache"
    log_dir = base_dir / "logs"
    results_dir = base_dir / "results"
    
    for directory in [model_dir, cache_dir, log_dir, results_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        
    # Setup logging
    logger = logging.getLogger('model_training')
    logger.setLevel(logging.INFO)
    
    # Create file handler which logs info messages
    fh = logging.FileHandler(log_dir / 'model_training.log')
    fh.setLevel(logging.INFO)
    
    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    try:
        logger.info("Starting model training process...")
        
        # Initialize components
        db_handler = DatabaseHandler(config['database']['path'])
        
        # Initialize Universe Management
        universe_manager = UniverseManager(
            db_path=config['database']['path'],
            config_path=config_path,
            cache_dir=config['paths']['cache_dir']
        )
        
        logger.info("Building trading universe...")
        
        # Build universe first - this will initialize the clusters
        universe_manager.build_universe()
        
        # Get initial universe
        all_symbols = universe_manager.db_setup.get_nyse_symbols()
        logger.info(f"Total NYSE symbols: {len(all_symbols)}")

        # Filter for available symbols
        available_symbols = set(db_handler.get_available_symbols())
        filtered_symbols = [s for s in all_symbols if s in available_symbols]
        logger.info(f"Available symbols in database: {len(filtered_symbols)}")
            
        # Initialize filters
        filter_config = FilterConfig(
            min_price=config['filters'].get('min_price', 5.0),
            max_price=config['filters'].get('max_price', 10000.0),
            min_volume=config['filters'].get('min_volume', 100000),
            min_market_cap=config['filters'].get('min_market_cap', 500000000),
            min_history_days=config['filters'].get('min_history_days', 252),
            min_dollar_volume=config['filters'].get('min_dollar_volume', 1000000),
            max_spread_pct=config['filters'].get('max_spread_pct', 0.02),
            min_trading_days_pct=config['filters'].get('min_trading_days_pct', 0.95),
            max_gap_days=config['filters'].get('max_gap_days', 5),
            min_price_std=config['filters'].get('min_price_std', 0.001)
        )
        universe_filter = UniverseFilter(config=filter_config)
            
        # Apply additional filters to available symbols
        filtered_symbols = universe_filter.apply_filters(
            symbols=filtered_symbols,
            market_data={s: db_handler.load_market_data(s) for s in filtered_symbols},
            features=universe_manager.stock_features
        )
        
        logger.info(f"Filtered to {len(filtered_symbols)} tradable symbols")
        
        # After filtering, update universe clusters
        universe_manager.clusters = universe_manager._group_by_sector(filtered_symbols)
        
        # Initialize clusterer
        clusterer = UniverseClusterer()
        
        # Create clusters using filtered symbols only
        clusters = clusterer.create_clusters(
            stock_features=universe_manager.stock_features,
            sector_groups=universe_manager.clusters
        )
        
        logger.info(f"Created {sum(len(c) for c in clusters.values())} clusters")
        
        # Initialize model trainer
        trainer = ModelTrainer(
            db_path=config['database']['path'],
            universe_manager=universe_manager,
            model_save_dir=config['paths']['model_dir']
        )
        
        # Initialize validator
        validator = ModelValidator(
            min_validation_window=config['validation']['min_window'],
            max_validation_window=config['validation']['max_window'],
            n_monte_carlo=config['validation']['n_monte_carlo']
        )
        
        # Training timeframes
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_dates = {
            'market': (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d'),
            'sector': (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d'),
            'cluster': (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
        }
        
        # Train market model
        logger.info("Training market model...")
        market_results = trainer._train_market_model(start_dates['market'], end_date)
        logger.info(f"Requesting data from {start_dates['market']} to {end_date}")

        # Validate market model
        market_metrics, market_details = validator.validate_model(
            model=trainer.market_model,
            validation_data=db_handler.load_market_data('SPY'),
            model_parameters=trainer.market_model.get_parameters()
        )
        
        logger.info(f"Market model metrics: {market_metrics}")
        
        # Train sector models
        logger.info("Training sector models...")
        sector_results = trainer.train_sector_models(start_dates['sector'], end_date)
        
        # Train and validate cluster models
        logger.info("Training cluster models...")
        cluster_results = {}
        cluster_metrics = {}
        
        for sector, sector_clusters in clusters.items():
            cluster_results[sector] = {}
            cluster_metrics[sector] = {}
            
            for cluster_name, symbols in sector_clusters.items():
                logger.info(f"Training cluster {sector}/{cluster_name}...")
                
                # Filter cluster symbols for availability
                available_cluster_symbols = [s for s in symbols if s in available_symbols]
                if not available_cluster_symbols:
                    logger.warning(f"No available symbols for cluster {sector}/{cluster_name}, skipping...")
                    continue
                
                # Get cluster data
                cluster_data = pd.concat([
                    db_handler.load_market_data(s) for s in available_cluster_symbols
                ])
                
                if cluster_data.empty:
                    logger.warning(f"No data available for cluster {sector}/{cluster_name}, skipping...")
                    continue
                
                # Train cluster model
                cluster_results[sector][cluster_name] = trainer.train_cluster_models(
                    start_dates['cluster'],
                    end_date,
                    sector=sector,
                    cluster=cluster_name
                )
                
                # Validate cluster model
                metrics, details = validator.validate_model(
                    model=trainer.cluster_models[sector][cluster_name],
                    validation_data=cluster_data,
                    model_parameters=trainer.cluster_models[sector][cluster_name].get_parameters(),
                    market_regime=universe_manager.current_market_regime
                )
                
                cluster_metrics[sector][cluster_name] = metrics
                
        # Save results
        results = {
            'timestamp': datetime.now().isoformat(),
            'market_model': {
                'results': market_results,
                'metrics': market_metrics.to_dict()
            },
            'sector_models': sector_results,
            'cluster_models': {
                sector: {
                    cluster: {
                        'results': cluster_results[sector][cluster],
                        'metrics': cluster_metrics[sector][cluster].to_dict()
                    }
                    for cluster in clusters[sector]
                }
                for sector in clusters
            }
        }
        
        results_path = results_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        logger.info(f"Training results saved to {results_path}")
        
        # Setup training scheduler
        scheduler = TrainingScheduler(
            trainer=trainer,
            universe_manager=universe_manager
        )
        
        # Start scheduler if configured
        if config.get('start_scheduler', False):
            scheduler.start()
            logger.info("Training scheduler started")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trading models')
    parser.add_argument('config', type=str, nargs='?', default='project/config/training_config.json',
                        help='Path to the configuration file')
    args = parser.parse_args()
    train_models(args.config)