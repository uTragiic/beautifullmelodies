"""
Universe management implementation for NYSE trading.
Handles stock filtering, universe construction, and updates.
"""

import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

from src.core.database_handler import DatabaseHandler
from src.indicators.calculator import IndicatorCalculator
from data.dbsetup import DatabaseSetup

logger = logging.getLogger(__name__)

class UniverseManager:
    """Manages stock universe selection and clustering."""
    
    def __init__(self, 
                 db_path: str,
                 config_path: str,
                 cache_dir: str = "cache"):
        """
        Initialize UniverseManager.
        
        Args:
            db_path: Path to market database
            config_path: Path to configuration file
            cache_dir: Directory for caching data
        """
        self.db_path = db_path  # Store db_path for later use
        self.db_handler = DatabaseHandler(db_path)
        self.db_setup = DatabaseSetup(db_path)
        self.indicator_calculator = IndicatorCalculator()
        
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
            
        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize containers
        self.universe: Dict[str, pd.DataFrame] = {}
        self.stock_features: Dict[str, Dict] = {}
        self.clusters: Dict[str, List[str]] = {}
        self.current_market_regime = "normal"  # Default market regime
        
    def build_universe(self) -> Dict[str, List[str]]:
        """
        Build and filter trading universe.
        
        Returns:
            Dictionary of filtered symbols by category
        """
        try:
            logger.info("Building trading universe...")
            
            # Get all NYSE symbols
            all_symbols = self.db_setup.get_nyse_symbols()
            
            # Get list of symbols (tables) available in the database
            available_symbols = self._get_available_symbols_from_db()
            logger.info(f"Total symbols available in database: {len(available_symbols)}")
            
            # Filter all_symbols to only include those present in the database
            symbols_in_db = [symbol for symbol in all_symbols if symbol in available_symbols]
            logger.info(f"Symbols after filtering for database availability: {len(symbols_in_db)}")
            
            # Apply initial filters
            tradable_symbols = self._apply_filters(symbols_in_db)
            
            # Calculate features for clustering
            self._calculate_stock_features(tradable_symbols)
            
            # Group stocks by sector
            self.clusters = self._group_by_sector(tradable_symbols)
            
            # Cache results
            self._cache_universe(self.clusters)
            
            logger.info(f"Universe built with {len(tradable_symbols)} symbols")
            return self.clusters
            
        except Exception as e:
            logger.error(f"Error building universe: {e}")
            raise

    def _get_available_symbols_from_db(self) -> List[str]:
        """Retrieve the list of symbols (tables) available in the database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            conn.close()
            # Extract table names from the tuples returned by fetchall()
            available_symbols = [table[0] for table in tables]
            return available_symbols
        except Exception as e:
            logger.error(f"Error retrieving symbols from database: {e}")
            return []
        
    def _apply_filters(self, symbols: List[str]) -> List[str]:
        """Apply filtering criteria to symbol list."""
        filtered_symbols = []
        
        filters = self.config.get('universe_filters', {})
        min_price = filters.get('min_price', 5.0)
        min_volume = filters.get('min_volume', 100000)
        min_market_cap = filters.get('min_market_cap', 500000000)
        
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(
                lambda s: self._check_symbol_criteria(
                    s, min_price, min_volume, min_market_cap
                ),
                symbols
            ))
                
        filtered_symbols = [
            symbol for symbol, passes in zip(symbols, results) if passes
        ]
        
        logger.info(f"Symbols after applying filters: {len(filtered_symbols)}")
        return filtered_symbols

    def _check_symbol_criteria(self,
                         symbol: str,
                         min_price: float,
                         min_volume: float,
                         min_market_cap: float) -> bool:
        """Check if symbol meets filtering criteria."""
        try:
            # Load recent data
            data = self.db_handler.load_market_data(symbol)
            if data.empty:
                return False
                
            # Check for minimum data points (e.g., 100 days instead of 200)
            if len(data) < 100:  # Reduced from 200
                return False
                
            # Get latest data point and recent data
            latest = data.iloc[-1]
            recent_data = data.tail(20)  # Last month of trading
            
            # More lenient price check (e.g., $1 instead of $5)
            if latest['close'] < 1.0:  # Reduced from min_price
                return False
                
            # More lenient volume check
            avg_volume = recent_data['volume'].mean()
            if avg_volume < 10000:  # Reduced from min_volume
                return False
                
            # More lenient market cap check
            market_cap = latest['close'] * avg_volume
            if market_cap < 10000000:  # Reduced from min_market_cap
                return False
                
            # Add price stability check
            price_std = recent_data['close'].std() / recent_data['close'].mean()
            if price_std > 0.5:  # 50% standard deviation threshold
                return False
                
            # Store features for later use
            self.stock_features[symbol] = {
                'price': latest['close'],
                'volume': avg_volume,
                'market_cap': market_cap,
                'volatility': data['close'].pct_change().std() * np.sqrt(252),
                'beta': self._calculate_beta(data['close'].pct_change())
            }
                
            return True
            
        except Exception as e:
            logger.warning(f"Error checking criteria for {symbol}: {e}")
            return False


    def _calculate_stock_features(self, symbols: List[str]) -> None:
        """Calculate features for stock classification."""
        for symbol in symbols:
            if symbol not in self.stock_features:  # Only calculate if not already done
                try:
                    data = self.db_handler.load_market_data(symbol)
                    if data.empty:
                        continue
                        
                    # Calculate basic features
                    returns = data['close'].pct_change()
                    volume = data['volume'].mean()  # Store as 'volume' instead of 'avg_volume'
                    
                    self.stock_features[symbol] = {
                        'volatility': returns.std() * np.sqrt(252),
                        'volume': volume,  # Changed from 'avg_volume' to 'volume'
                        'price': data['close'].iloc[-1],
                        'market_cap': data['close'].iloc[-1] * volume,
                        'beta': self._calculate_beta(returns)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error calculating features for {symbol}: {e}")
                    continue
                
    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate beta relative to SPY."""
        try:
            spy_data = self.db_handler.load_market_data('SPY')
            spy_returns = spy_data['close'].pct_change()
            
            # Align dates
            common_dates = returns.index.intersection(spy_returns.index)
            if len(common_dates) < 252:  # Require at least 1 year of data
                return 1.0
                
            returns = returns.loc[common_dates]
            spy_returns = spy_returns.loc[common_dates]
            
            covariance = returns.cov(spy_returns)
            variance = spy_returns.var()
            
            return covariance / variance if variance != 0 else 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating beta: {e}")
            return 1.0
                

    def _group_by_sector(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Group symbols based on their characteristics using existing database data.
        Creates pseudo-sectors based on price, volume, and volatility patterns.
        """
        try:
            logger.info("Grouping symbols based on trading characteristics...")
            
            # Initialize characteristic-based groups
            groups = {
                'Large_Cap_High_Vol': [],    # Large cap stocks with high volume
                'Large_Cap_Low_Vol': [],     # Large cap stocks with low volume
                'Mid_Cap_High_Vol': [],      # Mid cap stocks with high volume
                'Mid_Cap_Low_Vol': [],       # Mid cap stocks with low volume
                'Small_Cap_High_Vol': [],    # Small cap stocks with high volume
                'Small_Cap_Low_Vol': [],     # Small cap stocks with low volume
                'High_Volatility': [],       # High volatility stocks
                'Low_Volatility': [],        # Low volatility stocks
                'High_Volume': [],           # High trading volume
                'Other': []                  # Everything else
            }
            
            # Calculate metrics for each symbol
            symbol_metrics = {}
            
            for symbol in symbols:
                try:
                    # Load recent data (last 3 months)
                    data = self.db_handler.load_market_data(symbol)
                    if data.empty:
                        logger.debug(f"No data found for {symbol}")
                        groups['Other'].append(symbol)
                        continue
                        
                    # Calculate key metrics
                    recent_data = data.tail(60)  # Last ~3 months
                    
                    metrics = {
                        'avg_price': recent_data['close'].mean(),
                        'avg_volume': recent_data['volume'].mean(),
                        'volatility': recent_data['close'].pct_change().std() * np.sqrt(252),
                        'market_cap': recent_data['close'].iloc[-1] * recent_data['volume'].iloc[-1],
                        'avg_dollar_volume': (recent_data['close'] * recent_data['volume']).mean()
                    }
                    
                    symbol_metrics[symbol] = metrics
                    
                except Exception as e:
                    logger.debug(f"Error processing {symbol}: {e}")
                    groups['Other'].append(symbol)
                    continue
            
            if not symbol_metrics:
                logger.warning("No valid metrics calculated, returning all symbols as Other")
                return {'Other': symbols}
            
            # Calculate distribution thresholds
            metrics_df = pd.DataFrame(symbol_metrics).T
            
            market_cap_thresholds = metrics_df['market_cap'].quantile([0.33, 0.67])
            volume_threshold = metrics_df['avg_volume'].median()
            volatility_threshold = metrics_df['volatility'].quantile(0.75)
            
            # Classify symbols
            for symbol, metrics in symbol_metrics.items():
                try:
                    # High volatility stocks get their own category
                    if metrics['volatility'] > volatility_threshold:
                        groups['High_Volatility'].append(symbol)
                        continue
                    
                    # Classify based on market cap and volume
                    if metrics['market_cap'] > market_cap_thresholds[0.67]:
                        if metrics['avg_volume'] > volume_threshold:
                            groups['Large_Cap_High_Vol'].append(symbol)
                        else:
                            groups['Large_Cap_Low_Vol'].append(symbol)
                            
                    elif metrics['market_cap'] > market_cap_thresholds[0.33]:
                        if metrics['avg_volume'] > volume_threshold:
                            groups['Mid_Cap_High_Vol'].append(symbol)
                        else:
                            groups['Mid_Cap_Low_Vol'].append(symbol)
                            
                    else:
                        if metrics['avg_volume'] > volume_threshold:
                            groups['Small_Cap_High_Vol'].append(symbol)
                        else:
                            groups['Small_Cap_Low_Vol'].append(symbol)
                    
                    # Additional high volume category for very liquid stocks
                    if metrics['avg_volume'] > metrics_df['avg_volume'].quantile(0.9):
                        groups['High_Volume'].append(symbol)
                        
                except Exception as e:
                    logger.debug(f"Error classifying {symbol}: {e}")
                    groups['Other'].append(symbol)
            
            # Log group distributions
            for group_name, group_symbols in groups.items():
                if group_symbols:
                    logger.info(f"{group_name}: {len(group_symbols)} symbols")
            
            # Calculate and log some basic statistics
            if symbol_metrics:
                stats_df = pd.DataFrame(symbol_metrics).T
                logger.info("Universe Statistics:")
                logger.info(f"Average Market Cap: ${stats_df['market_cap'].mean():,.2f}")
                logger.info(f"Average Daily Volume: {stats_df['avg_volume'].mean():,.0f}")
                logger.info(f"Average Volatility: {stats_df['volatility'].mean():.2%}")
            
            # Remove empty groups and return
            return {k: v for k, v in groups.items() if v}
            
        except Exception as e:
            logger.error(f"Error in group creation: {e}")
            return {'Other': symbols}
    def _cache_universe(self, sector_groups: Dict[str, List[str]]) -> None:
        """Cache universe data."""
        cache_file = self.cache_dir / 'universe.json'
        with open(cache_file, 'w') as f:
            json.dump({
                'timestamp': pd.Timestamp.now().isoformat(),
                'sectors': sector_groups,
                'features': self.stock_features
            }, f, indent=4)
                
    def load_cached_universe(self) -> Optional[Dict[str, List[str]]]:
        """Load universe from cache if available."""
        cache_file = self.cache_dir / 'universe.json'
        if not cache_file.exists():
            return None
            
        with open(cache_file) as f:
            data = json.load(f)
            
        # Check if cache is recent (less than 1 day old)
        cache_time = pd.Timestamp(data['timestamp'])
        if pd.Timestamp.now() - cache_time > pd.Timedelta(days=1):
            return None
            
        self.stock_features = data['features']
        self.clusters = data['sectors']
        return self.clusters

    def update_universe(self) -> None:
        """Update universe and features."""
        self.build_universe()