import logging
import json
import psutil
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
import sys
import os


# Add the 'project' directory to sys.path, considering the script location
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from .models.overfittingcontrol import OverfittingController
from .universe import UniverseManager, UniverseClusterer, UniverseFilter, FilterConfig
from .training import ModelTrainer, TrainingScheduler, ModelValidator
from .models.mlmodel import MachineLearningModel
from .core.database_handler import DatabaseHandler
from .utils.logging import setup_logging



logger = logging.getLogger(__name__)

import logging
import json
import psutil
import multiprocessing as mp
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import pickle
import sys
import os
from functools import lru_cache

# Existing imports remain the same...

class DataManager:
    """Handles efficient data loading and caching with memory management."""
    
    def __init__(self, db_handler, cache_dir: Path, memory_threshold: float = 0.75):
        self.db_handler = db_handler
        self.cache_dir = cache_dir
        self.memory_threshold = memory_threshold
        self._data_cache = {}
        self._memory_mapped_files = {}
        
        # Pre-load SPY data on initialization
        self._preload_spy_data()
        
    def _preload_spy_data(self):
        """Ensure SPY data is loaded and cached."""
        try:
            spy_data = self.db_handler.load_market_data('SPY')
            if spy_data is None or spy_data.empty:
                raise ValueError("SPY data not available in database")
            self._data_cache['SPY'] = spy_data
            logger.info(f"Preloaded SPY data with {len(spy_data)} rows")
        except Exception as e:
            logger.error(f"Error preloading SPY data: {e}")
            raise ValueError("Failed to preload SPY data") from e
        
    def get_market_data(self, symbols: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Get market data efficiently using caching and memory mapping."""
        # Ensure SPY is always included
        if 'SPY' not in symbols:
            symbols = ['SPY'] + list(symbols)
            
        missing_symbols = [s for s in symbols if s not in self._data_cache]
        
        if missing_symbols:
            self._load_symbols(missing_symbols, start_date, end_date)
            
        return {symbol: self._get_cached_data(symbol) for symbol in symbols}
    
    def _load_symbols(self, symbols: List[str], start_date: str, end_date: str):
        """Load symbols data with memory management."""
        memory_usage = psutil.virtual_memory().percent / 100
        
        if memory_usage > self.memory_threshold:
            self._move_to_disk()
            
        # Don't move SPY to disk during batch loading
        symbols_to_load = [s for s in symbols if s != 'SPY']
        if symbols_to_load:
            batch_data = self.db_handler.load_all_market_data(
                symbols=symbols_to_load,
                start_date=start_date,
                end_date=end_date
            )
            self._data_cache.update(batch_data)
    
    def _move_to_disk(self):
        """Move least recently used data to disk using memory mapping."""
        if not self._data_cache:
            return
            
        # Move 25% of cached items to disk, but never move SPY
        items_to_move = len(self._data_cache) // 4
        symbols_to_move = [s for s in list(self._data_cache.keys())[:items_to_move] if s != 'SPY']
        
        for symbol in symbols_to_move:
            data = self._data_cache.pop(symbol)
            mmap_path = self.cache_dir / f"{symbol}_mmap.npy"
            
            # Save data to memory-mapped file
            mmap_array = np.memmap(mmap_path, dtype='float64', mode='w+',
                                 shape=data.values.shape)
            mmap_array[:] = data.values[:]
            mmap_array.flush()
            
            # Store metadata for reconstruction
            metadata_path = self.cache_dir / f"{symbol}_metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'index': data.index,
                    'columns': data.columns,
                    'shape': data.shape
                }, f)
                
            self._memory_mapped_files[symbol] = (mmap_path, metadata_path)
    
    def _get_cached_data(self, symbol: str) -> pd.DataFrame:
        """Retrieve data from cache or memory-mapped file."""
        if symbol in self._data_cache:
            return self._data_cache[symbol]
            
        if symbol in self._memory_mapped_files:
            mmap_path, metadata_path = self._memory_mapped_files[symbol]
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
                
            # Load memory-mapped array
            mmap_array = np.memmap(mmap_path, dtype='float64', mode='r',
                                 shape=metadata['shape'])
            
            # Reconstruct DataFrame
            return pd.DataFrame(
                mmap_array,
                index=metadata['index'],
                columns=metadata['columns']
            )
            
        raise KeyError(f"Data for symbol {symbol} not found in cache or disk")
    
    def clear_cache(self):
        """Clear all cached data and memory-mapped files."""
        # Save SPY data before clearing
        spy_data = self._data_cache.get('SPY')
        
        self._data_cache.clear()
        
        # Restore SPY data
        if spy_data is not None:
            self._data_cache['SPY'] = spy_data
        
        for mmap_path, metadata_path in self._memory_mapped_files.values():
            if mmap_path.exists():
                mmap_path.unlink()
            if metadata_path.exists():
                metadata_path.unlink()
                
        self._memory_mapped_files.clear()
class TrainingManager:
    """
    Enhanced training manager that handles all aspects of the model training process.
    Includes memory management, parallel training, checkpointing, and performance monitoring.
    """
    
    def __init__(self, config_path: str):
        """Initialize training manager with configuration."""
        with open(config_path) as f:
            self.config = json.load(f)
            
        # Add new configuration sections with defaults if not present
        self._init_extended_config()
        
        # Initialize paths
        self.base_dir = Path(self.config['base_dir'])
        self.model_dir = self.base_dir / "models"
        self.cache_dir = self.base_dir / "cache"
        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.log_dir = self.base_dir / "logs"
        self.results_dir = self.base_dir / "results"
        
        # Create directories
        for directory in [self.model_dir, self.cache_dir, self.checkpoint_dir, 
                         self.log_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize components with proper UniverseManager setup
        self._init_components()
        
        # Initialize memory management
        self.memory_threshold = self.config['memory_management']['memory_threshold']
        self.batch_size = self.config['memory_management']['batch_size']
        
        # Initialize checkpointing
        self.checkpoint_interval = self.config['checkpointing']['interval_minutes']
        self.last_checkpoint = datetime.now()
        
        # Initialize data manager
        self.data_manager = DataManager(
            self.db_handler,
            self.cache_dir,
            self.memory_threshold
        )
        
        # Performance tracking
        self.training_metrics = {
            'start_time': None,
            'duration': None,
            'memory_usage': [],
            'training_progress': {},
            'validation_metrics': {}
        }

    def _init_extended_config(self):
        """Initialize extended configuration with defaults."""
        default_extensions = {
            'memory_management': {
                'memory_threshold': 0.75,  # Use max 75% of available RAM
                'batch_size': 1000,  # Process 1000 samples at a time
                'cleanup_threshold': 0.85  # Trigger cleanup at 85% memory usage
            },
            'parallel_processing': {
                'max_workers': max(1, mp.cpu_count() - 1),
                'chunk_size': 10000,
                'use_threading': False  # Use multiprocessing by default
            },
            'checkpointing': {
                'enabled': True,
                'interval_minutes': 30,
                'keep_last_n': 3
            },
            'performance_monitoring': {
                'track_memory': True,
                'track_time': True,
                'alert_thresholds': {
                    'memory_usage': 0.9,
                    'training_time': 3600  # Alert if single model training exceeds 1 hour
                }
            }
        }
        
        # Update config with defaults for missing sections
        for section, defaults in default_extensions.items():
            if section not in self.config:
                self.config[section] = {}
            for key, value in defaults.items():
                if key not in self.config[section]:
                    self.config[section][key] = value

    def _init_components(self):
        """Initialize all required components with proper UniverseManager configuration."""
        self.db_handler = DatabaseHandler(self.config['database']['path'])
        
        # Initialize UniverseManager with proper configuration
        self.universe_manager = UniverseManager(
            db_path=self.config['database']['path'],
            config_path=self.config.get('universe_config_path', 'project/config/universe_config.json'),
            cache_dir=str(self.cache_dir)
        )
        
        # Initialize model trainer with UniverseManager integration
        self.model_trainer = ModelTrainer(
            db_path=self.config['database']['path'],
            universe_manager=self.universe_manager,
            model_save_dir=str(self.model_dir)
        )
        
        self.model_validator = ModelValidator(
            min_validation_window=self.config['validation']['min_window'],
            max_validation_window=self.config['validation']['max_window'],
            n_monte_carlo=self.config['validation']['n_monte_carlo']
        )
        
        self.overfitting_controller = OverfittingController(
            base_lookback_period=252,
            min_samples=100,
            max_complexity_score=0.8,
            parameter_stability_threshold=0.3
        )


    def _monitor_memory(self) -> float:
        """Monitor memory usage and trigger cleanup if needed."""
        memory = psutil.virtual_memory()
        usage = memory.percent / 100.0
        
        self.training_metrics['memory_usage'].append(usage)
        
        if usage > self.config['memory_management']['cleanup_threshold']:
            self._cleanup_memory()
            
        return usage

    def _cleanup_memory(self):
        """Clean up memory when usage is high."""
        import gc
        gc.collect()
        
        # Clear any cached data in components
        self.universe_manager.clear_cache()
        
        # Write memory-mapped files if needed
        if hasattr(self, '_temp_data'):
            for key, df in self._temp_data.items():
                mmap_path = self.cache_dir / f"{key}_mmap.npy"
                np.save(mmap_path, df.values)
                self._temp_data[key] = None
                
        self.training_metrics['memory_usage'].append(
            psutil.virtual_memory().percent / 100.0
        )

    def _create_checkpoint(self, stage: str, data: Dict[str, Any]):
        """Create a training checkpoint."""
        if not self.config['checkpointing']['enabled']:
            return
            
        now = datetime.now()
        if (now - self.last_checkpoint).seconds < self.checkpoint_interval * 60:
            return
            
        checkpoint_path = self.checkpoint_dir / f"checkpoint_{stage}_{now.strftime('%Y%m%d_%H%M%S')}.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(data, f)
            
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        self.last_checkpoint = now

    def _cleanup_and_save_results(self, results: Dict, report: Dict):
        """Clean up and save training results and report."""
        # Save results
        results_path = self.results_dir / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.results_dir / f"training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
            
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
            
        logger.info(f"Training results saved to {results_path}")
        logger.info(f"Training report saved to {report_path}")
        
        # Clean up caches
        self.data_manager.clear_cache()
        self.universe_manager.clear_cache()

    def _cleanup_old_checkpoints(self):
        """Clean up old checkpoints keeping only the last N."""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pkl"))
        keep_last = self.config['checkpointing']['keep_last_n']
        
        for checkpoint in checkpoints[:-keep_last]:
            checkpoint.unlink()

    def _train_model_parallel(self, model_info: Tuple[str, str, str, Dict]) -> Dict[str, Any]:
        """Train a single model in parallel."""
        model_type, identifier, start_date, data = model_info
        
        try:
            if model_type == 'sector':
                results = self.model_trainer.train_sector_models(
                    start_date, self.end_date, sector=identifier
                )
            else:  # cluster
                sector, cluster = identifier.split('/')
                results = self.model_trainer.train_cluster_models(
                    start_date, self.end_date, sector=sector, 
                    cluster=cluster, base_model=self.model_trainer.sector_models[sector]
                )
                
            # Validate the model
            model = (self.model_trainer.sector_models[identifier] if model_type == 'sector'
                    else self.model_trainer.cluster_models[sector][cluster])
            
            metrics, details = self.model_validator.validate_model(
                model=model,
                validation_data=data,
                model_parameters=model.get_parameters(),
                market_regime=self.universe_manager.current_market_regime
            )
            
            # Check for overfitting
            is_overfit, overfitting_scores = self.overfitting_controller.detect_overfitting(
                in_sample_metrics=results['in_sample_metrics'],
                out_sample_metrics=metrics,
                market_regime=self.universe_manager.current_market_regime,
                model_parameters=model.get_parameters()
            )
            
            if is_overfit:
                # Adjust model parameters
                adjusted_params = self.overfitting_controller.adjust_model(
                    model=model,
                    overfitting_scores=overfitting_scores,
                    market_regime=self.universe_manager.current_market_regime
                )
                
                # Retrain with adjusted parameters
                if model_type == 'sector':
                    results = self.model_trainer.train_sector_models(
                        start_date, self.end_date, sector=identifier,
                        parameters=adjusted_params
                    )
                else:
                    results = self.model_trainer.train_cluster_models(
                        start_date, self.end_date, sector=sector,
                        cluster=cluster, parameters=adjusted_params,
                        base_model=self.model_trainer.sector_models[sector]
                    )
                
                # Revalidate
                metrics, details = self.model_validator.validate_model(
                    model=model,
                    validation_data=data,
                    model_parameters=adjusted_params,
                    market_regime=self.universe_manager.current_market_regime
                )
            
            return {
                'type': model_type,
                'identifier': identifier,
                'results': results,
                'metrics': metrics.to_dict(),
                'overfitting_scores': overfitting_scores if is_overfit else None,
                'parameters': model.get_parameters()
            }
            
        except Exception as e:
            logger.error(f"Error training {model_type} model {identifier}: {e}")
            return {
                'type': model_type,
                'identifier': identifier,
                'error': str(e)
            }

    def train_models(self):
        """Main training function with optimized UniverseManager integration."""
        try:
            self.training_metrics['start_time'] = datetime.now()
            logger.info("Starting model training process...")
            
            # Build and validate universe
            logger.info("Building and validating trading universe...")
            if not self.universe_manager.is_cache_valid():
                self.universe_manager.build_universe()
            else:
                self.universe_manager.load_cached_universe()
            
            # Get universe statistics for logging
            universe_stats = self.universe_manager.get_universe_statistics()
            logger.info(f"Universe Statistics: {json.dumps(universe_stats, indent=2)}")
            
            # Load market data for filtered symbols
            start_date = min(self.config['training']['start_dates'].values())
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Explicitly check for SPY data first
            spy_data = self.data_manager.get_market_data(['SPY'], start_date, end_date).get('SPY')
            if spy_data is None or spy_data.empty:
                raise ValueError("SPY data not available or empty")
            logger.info(f"Successfully loaded SPY data with {len(spy_data)} rows")
            
            # Get tradable symbols from universe manager
            tradable_symbols = []
            for group_symbols in self.universe_manager.clusters.values():
                tradable_symbols.extend(group_symbols)
            
            logger.info(f"Loading market data for {len(tradable_symbols)} tradable symbols")
            
            # Load data efficiently using DataManager
            all_market_data = self.data_manager.get_market_data(
                symbols=tradable_symbols,
                start_date=start_date,
                end_date=end_date
            )
            
            # Ensure SPY data is in the market data
            if 'SPY' not in all_market_data:
                all_market_data['SPY'] = spy_data
                
            # Prepare consolidated market data for validation
            consolidated_market_data = self._prepare_consolidated_data(all_market_data)
            
            # Train market model using SPY data
            logger.info("Training market model...")
            if spy_data is None:
                raise ValueError("SPY data not found in database")
                
            market_results = self.model_trainer._train_market_model(
                self.config['training']['start_dates']['market'],
                end_date
            )
            
            # Validate market model with consolidated data
            market_metrics, market_details = self.model_validator.validate_model(
                model=self.model_trainer.market_model,
                validation_data=consolidated_market_data,
                model_parameters=self.model_trainer.market_model.get_parameters(),
                market_regime=self.universe_manager.current_market_regime
            )

            # Train sector models
            logger.info("Training sector models...")
            sector_results = {}
            sector_metrics = {}

            for sector, symbols in self.universe_manager.clusters.items():
                logger.info(f"Training model for sector: {sector}")
                sector_data = {sym: all_market_data[sym] for sym in symbols if sym in all_market_data}
                
                if not sector_data:
                    logger.warning(f"No data available for sector {sector}, skipping...")
                    continue
                
                # Train sector model
                sector_results[sector] = self.model_trainer.train_sector_models(
                    start_dates=self.config['training']['start_dates']['sector'],
                    end_date=end_date,
                    sector_data=sector_data
                )
                
                # Validate sector model
                sector_metrics[sector], _ = self.model_validator.validate_model(
                    model=self.model_trainer.sector_models[sector],
                    validation_data=pd.concat(sector_data.values()),
                    model_parameters=self.model_trainer.sector_models[sector].get_parameters(),
                    market_regime=self.universe_manager.current_market_regime
                )

            # Train cluster models
            logger.info("Training cluster models...")
            cluster_results = {}
            cluster_metrics = {}

            for sector, sector_clusters in self.universe_manager.clusters.items():
                cluster_results[sector] = {}
                cluster_metrics[sector] = {}
                
                for cluster_name, symbols in sector_clusters.items():
                    logger.info(f"Training model for cluster {sector}/{cluster_name}")
                    
                    # Get cluster data
                    cluster_data = {sym: all_market_data[sym] for sym in symbols if sym in all_market_data}
                    
                    if not cluster_data:
                        logger.warning(f"No data available for cluster {sector}/{cluster_name}, skipping...")
                        continue
                    
                    # Train cluster model
                    cluster_results[sector][cluster_name] = self.model_trainer.train_cluster_models(
                        start_dates=self.config['training']['start_dates']['cluster'],
                        end_date=end_date,
                        sector=sector,
                        cluster=cluster_name,
                        cluster_data=cluster_data
                    )
                    
                    # Validate cluster model
                    cluster_metrics[sector][cluster_name], _ = self.model_validator.validate_model(
                        model=self.model_trainer.cluster_models[sector][cluster_name],
                        validation_data=pd.concat(cluster_data.values()),
                        model_parameters=self.model_trainer.cluster_models[sector][cluster_name].get_parameters(),
                        market_regime=self.universe_manager.current_market_regime
                    )

            # Save all training results
            training_results = {
                'timestamp': datetime.now().isoformat(),
                'market_model': {
                    'results': market_results,
                    'metrics': market_metrics
                },
                'sector_models': {
                    sector: {
                        'results': sector_results[sector],
                        'metrics': sector_metrics[sector]
                    } for sector in sector_results
                },
                'cluster_models': {
                    sector: {
                        cluster: {
                            'results': cluster_results[sector][cluster],
                            'metrics': cluster_metrics[sector][cluster]
                        } for cluster in cluster_results[sector]
                    } for sector in cluster_results
                }
            }

            # Save results to file
            results_path = Path(self.config['paths']['results_dir']) / f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_path, 'w') as f:
                json.dump(training_results, f, indent=4)

            # Update training metrics
            self.training_metrics['end_time'] = datetime.now()
            self.training_metrics['duration'] = (self.training_metrics['end_time'] - self.training_metrics['start_time']).total_seconds()
            self.training_metrics['market_metrics'] = market_metrics
            self.training_metrics['sector_metrics'] = sector_metrics
            self.training_metrics['cluster_metrics'] = cluster_metrics

            logger.info("Model training completed successfully")
            return training_results

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise

    def _prepare_consolidated_data(self, market_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Prepare consolidated market data for validation.
        
        Args:
            market_data: Dictionary of DataFrames with market data per symbol
            
        Returns:
            pd.DataFrame: Consolidated market data
        """
        try:
            logger.info("Preparing consolidated market data...")
            
            # Start with SPY as the base
            if 'SPY' not in market_data:
                raise ValueError("SPY data required for consolidation")
            
            base_df = market_data['SPY'].copy()
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Create a dictionary to store all symbol DataFrames
            symbol_dfs = {}
            
            # Add SPY data first
            spy_cols = {col: col for col in required_columns}
            symbol_dfs['SPY'] = base_df[required_columns]
            
            # Process other symbols
            for symbol, data in market_data.items():
                if symbol == 'SPY':
                    continue
                    
                # Ensure data alignment
                aligned_data = data.reindex(base_df.index)
                
                # Rename columns with symbol prefix
                renamed_cols = {col: f"{symbol}_{col}" for col in required_columns}
                symbol_df = aligned_data[required_columns].rename(columns=renamed_cols)
                symbol_dfs[symbol] = symbol_df
            
            # Concatenate all DataFrames at once
            consolidated = pd.concat(symbol_dfs.values(), axis=1)
            
            # Verify the consolidated data meets Backtest requirements
            if not isinstance(consolidated.index, pd.DatetimeIndex):
                consolidated.index = pd.to_datetime(consolidated.index)
                
            # Ensure required columns exist
            missing_columns = [col for col in required_columns 
                            if col not in consolidated.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in consolidated data: {missing_columns}")
                
            logger.info(f"Successfully prepared consolidated data with {len(consolidated)} rows")
            return consolidated
            
        except Exception as e:
            logger.error(f"Error preparing consolidated market data: {e}")
            raise

    def _organize_training_results(self, market_results: Dict, market_metrics: pd.DataFrame,
                                 training_results: List[Dict], universe_stats: Dict) -> Dict:
        """Organize all training results into a structured format."""
        sector_results = {}
        sector_metrics = {}
        cluster_results = {}
        cluster_metrics = {}
        
        for result in training_results:
            if result.get('error'):
                logger.error(f"Error in {result['type']} model {result['identifier']}: {result['error']}")
                continue
                
            if result['type'] == 'sector':
                sector_results[result['identifier']] = result['results']
                sector_metrics[result['identifier']] = result['metrics']
            else:
                sector, cluster = result['identifier'].split('/')
                if sector not in cluster_results:
                    cluster_results[sector] = {}
                    cluster_metrics[sector] = {}
                cluster_results[sector][cluster] = result['results']
                cluster_metrics[sector][cluster] = result['metrics']
        
        return {
            'timestamp': datetime.now().isoformat(),
            'universe_statistics': universe_stats,
            'market_model': {
                'results': market_results,
                'metrics': market_metrics.to_dict(),
            },
            'sector_models': {
                sector: {
                    'results': sector_results[sector],
                    'metrics': sector_metrics[sector]
                }
                for sector in sector_results
            },
            'cluster_models': {
                sector: {
                    cluster: {
                        'results': cluster_results[sector][cluster],
                        'metrics': cluster_metrics[sector][cluster]
                    }
                    for cluster in cluster_results.get(sector, {})
                }
                for sector in cluster_results
            },
            'training_metrics': {
                'duration': str(datetime.now() - self.training_metrics['start_time']),
                'max_memory_usage': max(self.training_metrics['memory_usage']),
                'avg_memory_usage': sum(self.training_metrics['memory_usage']) / len(self.training_metrics['memory_usage']),
                'training_progress': self.training_metrics['training_progress']
            }
        }

    def _update_training_metrics(self, result: Dict[str, Any]):
        """Update training metrics with new result."""
        if 'type' in result and 'identifier' in result:
            self.training_metrics['training_progress'][f"{result['type']}_{result['identifier']}"] = 'completed'
            if 'metrics' in result:
                self.training_metrics['validation_metrics'][f"{result['type']}_{result['identifier']}"] = result['metrics']
                
    def _generate_training_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive training report with analysis and recommendations."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'overview': {
                'total_models_trained': len(results['sector_models']) + 
                    sum(len(clusters) for clusters in results['cluster_models'].values()) + 1,
                'training_duration': results['training_metrics']['duration'],
                'memory_usage': {
                    'max': results['training_metrics']['max_memory_usage'],
                    'average': results['training_metrics']['avg_memory_usage']
                }
            },
            'model_performance': {
                'market_model': self._analyze_model_performance(
                    results['market_model']['metrics'],
                    'market'
                ),
                'sector_models': {
                    sector: self._analyze_model_performance(metrics['metrics'], 'sector')
                    for sector, metrics in results['sector_models'].items()
                },
                'cluster_models': {
                    sector: {
                        cluster: self._analyze_model_performance(metrics['metrics'], 'cluster')
                        for cluster, metrics in clusters.items()
                    }
                    for sector, clusters in results['cluster_models'].items()
                }
            },
            'overfitting_analysis': {
                'market_model': results['market_model'].get('overfitting_scores'),
                'problematic_models': self._identify_problematic_models(results)
            },
            'recommendations': self._generate_recommendations(results)
        }
        
        return report

    def _analyze_model_performance(self, metrics: Dict[str, float], 
                                 model_type: str) -> Dict[str, Any]:
        """Analyze performance metrics for a specific model."""
        analysis = {
            'overall_score': self._calculate_model_score(metrics),
            'key_metrics': {
                metric: metrics[metric]
                for metric in ['sharpe_ratio', 'win_rate', 'profit_factor', 
                             'max_drawdown', 'volatility']
                if metric in metrics
            },
            'concerns': []
        }
        
        # Add performance concerns
        if metrics.get('sharpe_ratio', 0) < 0.5:
            analysis['concerns'].append('Low risk-adjusted returns')
        if metrics.get('win_rate', 0) < 0.45:
            analysis['concerns'].append('Low win rate')
        if metrics.get('max_drawdown', 0) < -0.2:
            analysis['concerns'].append('High maximum drawdown')
        
        return analysis

    def _calculate_model_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall model score based on key metrics."""
        weights = {
            'sharpe_ratio': 0.3,
            'win_rate': 0.2,
            'profit_factor': 0.2,
            'max_drawdown': 0.15,
            'volatility': 0.15
        }
        
        score = 0
        for metric, weight in weights.items():
            if metric in metrics:
                normalized_value = self._normalize_metric(metric, metrics[metric])
                score += normalized_value * weight
                
        return score

    def _normalize_metric(self, metric: str, value: float) -> float:
        """Normalize metric value to [0,1] range."""
        metric_ranges = {
            'sharpe_ratio': (-1, 3),
            'win_rate': (0, 1),
            'profit_factor': (0, 3),
            'max_drawdown': (-1, 0),
            'volatility': (0, 0.5)
        }
        
        min_val, max_val = metric_ranges[metric]
        normalized = (value - min_val) / (max_val - min_val)
        return max(0, min(1, normalized))

    def _identify_problematic_models(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify models with potential issues."""
        problematic_models = []
        
        def check_model(metrics, model_type, identifier):
            score = self._calculate_model_score(metrics)
            if score < 0.5:
                problematic_models.append({
                    'type': model_type,
                    'identifier': identifier,
                    'score': score,
                    'metrics': metrics
                })
        
        # Check market model
        check_model(
            results['market_model']['metrics'],
            'market',
            'market'
        )
        
        # Check sector models
        for sector, data in results['sector_models'].items():
            check_model(data['metrics'], 'sector', sector)
        
        # Check cluster models
        for sector, clusters in results['cluster_models'].items():
            for cluster, data in clusters.items():
                check_model(
                    data['metrics'],
                    'cluster',
                    f"{sector}/{cluster}"
                )
        
        return problematic_models

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on training results."""
        recommendations = []
        
        # Performance recommendations
        problematic_models = self._identify_problematic_models(results)
        if problematic_models:
            recommendations.append(
                f"Consider retraining {len(problematic_models)} underperforming models "
                f"with adjusted parameters or different architectures."
            )
        
        # Memory usage recommendations
        avg_memory = results['training_metrics']['avg_memory_usage']
        if avg_memory > 0.8:
            recommendations.append(
                "High memory usage detected. Consider increasing batch size or "
                "implementing data streaming for large datasets."
            )
        
        # Training duration recommendations
        duration = pd.Timedelta(results['training_metrics']['duration'])
        if duration > pd.Timedelta(hours=4):
            recommendations.append(
                "Long training duration. Consider optimizing parallel processing "
                "settings or reducing validation complexity."
            )
        
        # Model-specific recommendations
        market_metrics = results['market_model']['metrics']
        if market_metrics['sharpe_ratio'] < 1.0:
            recommendations.append(
                "Market model showing suboptimal risk-adjusted returns. "
                "Consider feature engineering or alternative architectures."
            )
        
        # Overfitting recommendations
        if results['market_model'].get('overfitting_scores'):
            recommendations.append(
                "Market model shows signs of overfitting. Review regularization "
                "parameters and feature selection."
            )
        
        return recommendations

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train trading models')
    parser.add_argument('config', type=str, nargs='?', 
                       default='project/config/training_config.json',
                       help='Path to the configuration file')
    args = parser.parse_args()
    
    training_manager = TrainingManager(args.config)
    results, report = training_manager.train_models()