"""
Main model training implementation.
Handles training of hierarchical models across market clusters.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import joblib
from datetime import timedelta

from ..models.mlmodel import MachineLearningModel
from ..indicators.calculator import IndicatorCalculator
from ..analysis.marketcondition import MarketConditionAnalyzer
from ..core.database_handler import DatabaseHandler
from ..universe import UniverseManager, UniverseClusterer
from ..backtesting.enhancedbacktest import EnhancedBacktest
from ..utils.validation import validate_dataframe

logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration for model training."""
    def __init__(self,
                 base_lookback: int = 300,
                 train_test_split: float = 0.7,
                 n_validation_splits: int = 5,
                 min_train_samples: int = 252,
                 max_train_samples: int = 2520,  # 10 years
                 batch_size: int = 32,
                 learning_rate: float = 0.001,
                 early_stopping_patience: int = 10,
                 min_performance_threshold: float = 0.5,
                 max_correlation_threshold: float = 0.7,
                 parallel_training: bool = True,
                 max_workers: int = 4):
        self.base_lookback = base_lookback
        self.train_test_split = train_test_split
        self.n_validation_splits = n_validation_splits
        self.min_train_samples = min_train_samples
        self.max_train_samples = max_train_samples
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stopping_patience = early_stopping_patience
        self.min_performance_threshold = min_performance_threshold
        self.max_correlation_threshold = max_correlation_threshold
        self.parallel_training = parallel_training
        self.max_workers = max_workers
        self.risk_free_rate = 0.02


class ModelTrainer:
    """
    Handles training of hierarchical models across market clusters.
    """
    
    def __init__(self,
                 db_path: str,
                 universe_manager: UniverseManager,
                 model_save_dir: str = "models",
                 config: Optional[TrainingConfig] = None):
        """
        Initialize ModelTrainer.
        
        Args:
            db_path: Path to market database
            universe_manager: UniverseManager instance
            model_save_dir: Directory to save trained models
            config: Training configuration
        """
        self.db_handler = DatabaseHandler(db_path)
        self.universe_manager = universe_manager
        self.model_save_dir = Path(model_save_dir)
        self.config = config or TrainingConfig()
        
        # Create save directory structure
        self.model_save_dir.mkdir(parents=True, exist_ok=True)
        (self.model_save_dir / "market").mkdir(exist_ok=True)
        (self.model_save_dir / "sector").mkdir(exist_ok=True)
        (self.model_save_dir / "cluster").mkdir(exist_ok=True)
        
        # Initialize components
        self.indicator_calculator = IndicatorCalculator()
        self.market_analyzer = MarketConditionAnalyzer()
        
        # Initialize model containers
        self.market_model: Optional[MachineLearningModel] = None
        self.sector_models: Dict[str, MachineLearningModel] = {}
        self.cluster_models: Dict[str, Dict[str, MachineLearningModel]] = {}
        
        # Training metrics
        self.training_metrics: Dict[str, Any] = {}
        
    def train_all_models(self,
                        start_date: str,
                        end_date: str) -> Dict[str, Any]:
        """
        Train complete model hierarchy.
        
        Args:
            start_date: Training start date
            end_date: Training end date
            
        Returns:
            Dictionary of training results
        """
        try:
            logger.info("Starting full model training...")
            
            # Train market model
            self._train_market_model(start_date, end_date)
            
            # Train sector models
            self._train_sector_models(start_date, end_date)
            
            # Train cluster models
            self._train_cluster_models(start_date, end_date)
            
            # Save training metrics
            self._save_training_metrics()
            
            return self.training_metrics
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise
            
    def _train_market_model(self, start_date: str, end_date: str) -> None:
        """Train market-wide model."""
        try:
            logger.info("Training market model...")
            
            # Get market data (e.g., SPY)
            market_data = self._prepare_market_data(start_date, end_date)
            
            # Calculate technical indicators before training
            logger.info("Calculating technical indicators...")
            market_data = self.indicator_calculator.calculate_indicators(market_data)
            
            # Validate that indicators were calculated
            required_indicators = [
                'RSI', 'MACD_diff', 'ADX', 'ATR', 'Volume_Ratio', 
                'Momentum', 'Stoch_K', 'OBV', 'BB_width', 'VWAP'
            ]
            
            missing_indicators = [ind for ind in required_indicators if ind not in market_data.columns]
            if missing_indicators:
                raise ValueError(f"Failed to calculate indicators: {missing_indicators}")
            
            # Initialize and train market model
            self.market_model = MachineLearningModel(
                lookback_period=self.config.base_lookback
            )
            
            # Train the model
            logger.info("Training model...")
            self.market_model.train(market_data)
            
            # Save model
            model_path = self.model_save_dir / "market" / "market_model"
            self.market_model.save_model(str(model_path))
            
            # Validate and store metrics
            logger.info("Validating model...")
            validation_metrics = self._validate_model(
                self.market_model,
                market_data
            )
            
            # Record metrics
            self.training_metrics['market_model'] = validation_metrics
            
            logger.info("Market model training complete")
            
        except Exception as e:
            logger.error(f"Error training market model: {e}")
            raise

            
    def _train_sector_models(self, start_date: str, end_date: str) -> None:
        """Train sector-specific models."""
        try:
            logger.info("Training sector models...")
            
            # Get sector groupings
            sector_groups = self.universe_manager.clusters
            
            if self.config.parallel_training:
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = []
                    for sector in sector_groups:
                        future = executor.submit(
                            self._train_single_sector_model,
                            sector,
                            sector_groups[sector],
                            start_date,
                            end_date
                        )
                        futures.append((sector, future))
                        
                    # Collect results
                    for sector, future in futures:
                        try:
                            self.sector_models[sector] = future.result()
                        except Exception as e:
                            logger.error(f"Error training sector model {sector}: {e}")
            else:
                for sector, symbols in sector_groups.items():
                    try:
                        self.sector_models[sector] = self._train_single_sector_model(
                            sector, symbols, start_date, end_date
                        )
                    except Exception as e:
                        logger.error(f"Error training sector model {sector}: {e}")
                        
            logger.info("Sector model training complete")
            
        except Exception as e:
            logger.error(f"Error training sector models: {e}")
            raise
            
    def _train_single_sector_model(self,
                                 sector: str,
                                 symbols: List[str],
                                 start_date: str,
                                 end_date: str) -> MachineLearningModel:
        """Train model for a single sector."""
        # Prepare sector data
        sector_data = self._prepare_sector_data(symbols, start_date, end_date)
        
        # Initialize and train model
        model = MachineLearningModel(
            lookback_period=self.config.base_lookback
        )
        model.train(sector_data)
        
        # Save model
        model_path = self.model_save_dir / "sector" / f"{sector}_model"
        model.save_model(str(model_path))
        
        # Record metrics
        self.training_metrics[f'sector_model_{sector}'] = self._validate_model(
            model,
            sector_data
        )
        
        return model
        
    def _train_cluster_models(self, start_date: str, end_date: str) -> None:
        """Train cluster-specific models."""
        try:
            logger.info("Training cluster models...")
            
            for sector, clusters in self.universe_manager.clusters.items():
                self.cluster_models[sector] = {}
                
                for cluster_name, symbols in clusters.items():
                    try:
                        # Prepare cluster data
                        cluster_data = self._prepare_cluster_data(
                            symbols, start_date, end_date
                        )
                        
                        # Initialize and train model
                        model = MachineLearningModel(
                            lookback_period=self.config.base_lookback
                        )
                        model.train(cluster_data)
                        
                        # Save model
                        model_path = self.model_save_dir / "cluster" / f"{sector}_{cluster_name}_model"
                        model.save_model(str(model_path))
                        
                        # Record metrics
                        self.training_metrics[f'cluster_model_{sector}_{cluster_name}'] = self._validate_model(
                            model,
                            cluster_data
                        )
                        
                        self.cluster_models[sector][cluster_name] = model
                        
                    except Exception as e:
                        logger.error(f"Error training cluster model {sector}/{cluster_name}: {e}")
                        continue
                        
            logger.info("Cluster model training complete")
            
        except Exception as e:
            logger.error(f"Error training cluster models: {e}")
            raise
            
    def _prepare_market_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Prepare market-wide training data."""
        try:
            logger.info(f"Loading market data from {start_date} to {end_date}")
            
            # Get market data (using SPY as market proxy)
            market_data = self.db_handler.load_market_data('SPY', start_date, end_date)
            
            if market_data.empty:
                raise ValueError("No market data loaded")
                
            # Add check for minimum data points
            min_required_points = 300  # Based on lookback period
            if len(market_data) < min_required_points:
                logger.warning(f"Insufficient data points ({len(market_data)}), attempting to load more historical data")
                # Try loading more historical data
                extended_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365*5)).strftime('%Y-%m-%d')
                market_data = self.db_handler.load_market_data('SPY', extended_start, end_date)
                
                if len(market_data) < min_required_points:
                    raise ValueError(f"Insufficient data points even with extended history: {len(market_data)} < {min_required_points}")
            
            logger.info(f"Loaded {len(market_data)} data points")
            
            # Rest of the data preparation...
            return market_data
            
        except Exception as e:
            logger.error(f"Error preparing market data: {e}")
            raise
    def _prepare_sector_data(self,
                           symbols: List[str],
                           start_date: str,
                           end_date: str) -> pd.DataFrame:
        """Prepare sector-level training data."""
        sector_data = []
        
        for symbol in symbols:
            try:
                data = self.db_handler.load_market_data(symbol)
                data = data[
                    (data.index >= start_date) &
                    (data.index <= end_date)
                ]
                
                # Calculate indicators
                data = self.indicator_calculator.calculate_indicators(data)
                sector_data.append(data)
                
            except Exception as e:
                logger.warning(f"Error preparing data for {symbol}: {e}")
                continue
                
        return pd.concat(sector_data)
        
    def _prepare_cluster_data(self,
                            symbols: List[str],
                            start_date: str,
                            end_date: str) -> pd.DataFrame:
        """Prepare cluster-level training data."""
        return self._prepare_sector_data(symbols, start_date, end_date)
        
    def _validate_model(self, model: MachineLearningModel, data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate model performance with adjusted data requirements.
        
        Args:
            model: Model to validate
            data: Validation data
            
        Returns:
            Dictionary of performance metrics
        """
        try:
            # Initialize backtester
            backtest = EnhancedBacktest(data)
            
            # Run backtest with adjusted parameters
            backtest_results = backtest.run_backtest(
                model,
                n_splits=3,  # Reduced number of splits
                n_jobs=-1
            )
            
            # Calculate performance metrics
            metrics = backtest.calculate_performance_metrics()
            
            # Calculate metrics for market condition performance
            market_condition_stats = backtest.calculate_market_condition_statistics(backtest_results)
            
            # Calculate trading statistics
            trading_stats = backtest.calculate_trading_statistics()
            
            # Return comprehensive metrics
            return {
                'sharpe_ratio': metrics.sharpe_ratio,
                'win_rate': metrics.win_rate,
                'profit_factor': metrics.profit_factor,
                'max_drawdown': metrics.max_drawdown,
                'volatility': metrics.volatility,
                'market_condition_stats': market_condition_stats,
                'trading_stats': trading_stats,
                'total_trades': len(backtest_results),
                'average_trade_duration': trading_stats.get('avg_trade_duration', 0),
                'max_consecutive_losses': trading_stats.get('max_consecutive_losses', 0),
                'recovery_factor': trading_stats.get('recovery_factor', 0)
            }
            
        except Exception as e:
            logger.error(f"Error validating model: {e}")
            raise
            
    def _save_training_metrics(self) -> None:
        """Save training metrics to file."""
        metrics_path = self.model_save_dir / "training_metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(self.training_metrics, f, indent=4)
            
    def load_models(self) -> None:
        """Load all trained models."""
        try:
            # Load market model
            market_path = self.model_save_dir / "market" / "market_model"
            if market_path.exists():
                self.market_model = MachineLearningModel()
                self.market_model.load_model(str(market_path))
                
            # Load sector models
            for sector_path in (self.model_save_dir / "sector").glob("*_model"):
                sector = sector_path.name.replace("_model", "")
                model = MachineLearningModel()
                model.load_model(str(sector_path))
                self.sector_models[sector] = model
                
            # Load cluster models
            for cluster_path in (self.model_save_dir / "cluster").glob("*_model"):
                parts = cluster_path.name.replace("_model", "").split("_")
                sector = parts[0]
                cluster_name = "_".join(parts[1:])
                
                if sector not in self.cluster_models:
                    self.cluster_models[sector] = {}
                    
                model = MachineLearningModel()
                model.load_model(str(cluster_path))
                self.cluster_models[sector][cluster_name] = model
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise