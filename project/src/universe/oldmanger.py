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
        """Initialize with same parameters but add SPY data validation"""
        self.db_path = db_path
        self.db_handler = DatabaseHandler(db_path)
        self.db_setup = DatabaseSetup(db_path)
        self.indicator_calculator = IndicatorCalculator()
        
        # Validate SPY data availability early
        if not self._validate_spy_data():
            raise ValueError("SPY data not available - required for beta calculations")
            
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
            
        # Adjust default filter values to be less restrictive
        self._set_default_filters()
            
        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize containers
        self.universe: Dict[str, pd.DataFrame] = {}
        self.stock_features: Dict[str, Dict] = {}
        self.clusters: Dict[str, List[str]] = {}

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
    def _validate_spy_data(self) -> bool:
        """Ensure SPY data is available and valid"""
        try:
            spy_data = self.db_handler.load_market_data('SPY')
            if spy_data is None or spy_data.empty:
                logger.error("SPY data not found in database")
                return False
            
            # Validate minimum data requirements
            if len(spy_data) < 252:  # 1 year minimum
                logger.error("Insufficient SPY history for beta calculations")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating SPY data: {e}")
            return False

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

    def _set_default_filters(self):
        """Set more reasonable default filter values"""
        default_filters = {
            'min_price': 1.0,  # Lower minimum price
            'max_price': 10000.0,
            'min_volume': 5000,  # Lower volume requirement
            'min_market_cap': 1000000,  # Lower market cap requirement
            'min_history_days': 126,  # 6 months instead of 1 year
            'min_dollar_volume': 50000,  # Lower dollar volume requirement
            'max_spread_pct': 0.05,  # More permissive spread
            'min_trading_days_pct': 0.60,  # Lower trading days requirement
            'max_gap_days': 7,  # More permissive gap allowance
            'min_price_std': 0.0005  # Lower price variation requirement
        }
        
        # Update config filters with defaults if not specified
        if 'universe_filters' not in self.config:
            self.config['universe_filters'] = {}
            
        for key, value in default_filters.items():
            if key not in self.config['universe_filters']:
                self.config['universe_filters'][key] = value
                
    def _apply_filters(self, symbols: List[str]) -> List[str]:
        """Apply filtering criteria to symbol list with detailed diagnostics."""
        try:
            logger.info(f"Starting filtering process with {len(symbols)} symbols")
            
            # Always keep SPY in the universe
            if 'SPY' in symbols:
                symbols.remove('SPY')  # Remove temporarily to skip filtering
                keep_spy = True
            else:
                keep_spy = False
                
            # Get filter criteria from config
            filters = self.config.get('universe_filters', {})
            min_price = filters.get('min_price', 5.0)
            max_price = filters.get('max_price', 10000.0)
            min_volume = filters.get('min_volume', 10000)
            min_market_cap = filters.get('min_market_cap', 5000000)
            min_history_days = filters.get('min_history_days', 252)
            min_dollar_volume = filters.get('min_dollar_volume', 100000)
            max_spread_pct = filters.get('max_spread_pct', 0.02)
            min_trading_days_pct = filters.get('min_trading_days_pct', 0.75)
            max_gap_days = filters.get('max_gap_days', 5)
            min_price_std = filters.get('min_price_std', 0.001)
            
            # Apply filters as before...
            remaining_symbols = set(symbols)
            debug_info = {}
            
            # Rest of your existing filtering code...
            
            passed_symbols = list(remaining_symbols)
            
            # Add SPY back if it was present
            if keep_spy:
                passed_symbols.append('SPY')
                
            return passed_symbols
                
        except Exception as e:
            logger.error(f"Error in _apply_filters: {e}")
            return symbols
class UniverseManager:
    def __init__(self, 
                 db_path: str,
                 config_path: str,
                 cache_dir: str = "cache"):
        """Initialize with same parameters but add SPY data validation"""
        self.db_path = db_path
        self.db_handler = DatabaseHandler(db_path)
        self.db_setup = DatabaseSetup(db_path)
        self.indicator_calculator = IndicatorCalculator()
        
        # Validate SPY data availability early
        if not self._validate_spy_data():
            raise ValueError("SPY data not available - required for beta calculations")
            
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
            
        # Adjust default filter values to be less restrictive
        self._set_default_filters()
            
        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize containers
        self.universe: Dict[str, pd.DataFrame] = {}
        self.stock_features: Dict[str, Dict] = {}
        self.clusters: Dict[str, List[str]] = {}
        
    def _validate_spy_data(self) -> bool:
        """Ensure SPY data is available and valid"""
        try:
            spy_data = self.db_handler.load_market_data('SPY')
            if spy_data is None or spy_data.empty:
                logger.error("SPY data not found in database")
                return False
            
            # Validate minimum data requirements
            if len(spy_data) < 252:  # 1 year minimum
                logger.error("Insufficient SPY history for beta calculations")
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error validating SPY data: {e}")
            return False
            
    def _set_default_filters(self):
        """Set more reasonable default filter values"""
        default_filters = {
            'min_price': 1.0,  # Lower minimum price
            'max_price': 10000.0,
            'min_volume': 5000,  # Lower volume requirement
            'min_market_cap': 1000000,  # Lower market cap requirement
            'min_history_days': 126,  # 6 months instead of 1 year
            'min_dollar_volume': 50000,  # Lower dollar volume requirement
            'max_spread_pct': 0.05,  # More permissive spread
            'min_trading_days_pct': 0.60,  # Lower trading days requirement
            'max_gap_days': 7,  # More permissive gap allowance
            'min_price_std': 0.0005  # Lower price variation requirement
        }
        
        # Update config filters with defaults if not specified
        if 'universe_filters' not in self.config:
            self.config['universe_filters'] = {}
            
        for key, value in default_filters.items():
            if key not in self.config['universe_filters']:
                self.config['universe_filters'][key] = value
                
    def _apply_filters(self, symbols: List[str]) -> List[str]:
        """Modified filter application with more lenient criteria"""
        try:
            logger.info(f"Starting filtering process with {len(symbols)} symbols")
            
            # Get filter criteria from config
            filters = self.config.get('universe_filters', {})
            
            remaining_symbols = set(symbols)
            debug_info = {}
            
            # Track both individual and cumulative filtering results
            filter_stats = {criterion: {'count': 0, 'symbols': set()} 
                          for criterion in filters.keys()}
            
            # Process symbols in parallel for better performance
            with ThreadPoolExecutor() as executor:
                future_to_symbol = {
                    executor.submit(self._check_symbol_criteria_debug, symbol, **filters): symbol 
                    for symbol in symbols
                }
                
                for future in future_to_symbol:
                    symbol = future_to_symbol[future]
                    try:
                        failures = future.result()
                        if failures:
                            debug_info[symbol] = failures
                            for failure_reason in failures:
                                filter_stats[failure_reason]['count'] += 1
                                filter_stats[failure_reason]['symbols'].add(symbol)
                            remaining_symbols.remove(symbol)
                    except Exception as e:
                        logger.warning(f"Error processing {symbol}: {e}")
                        
            # Log detailed statistics
            self._log_filter_statistics(symbols, filter_stats, debug_info)
            
            passed_symbols = list(remaining_symbols)
            if len(passed_symbols) < 50:  # If too few symbols pass
                logger.warning("Few symbols passed filters - using fallback criteria")
                return self._get_fallback_symbols(symbols, debug_info)
                
            return passed_symbols
            
        except Exception as e:
            logger.error(f"Error in _apply_filters: {e}")
            return self._get_fallback_symbols(symbols, {})
            
    def _log_filter_statistics(self, symbols: List[str], 
                             filter_stats: Dict, 
                             debug_info: Dict) -> None:
        """Log detailed filter statistics"""
        total_symbols = len(symbols)
        logger.info(f"\n=== Filter Statistics ===")
        
        for reason, stats in filter_stats.items():
            if stats['count'] > 0:
                pct = stats['count']/total_symbols*100
                logger.info(f"{reason}: {stats['count']} symbols ({pct:.1f}%)")
                
        symbols_failing_multiple = sum(1 for s in debug_info if len(debug_info[s]) > 1)
        logger.info(f"\nSymbols failing multiple criteria: {symbols_failing_multiple}")
        
        # Sample of multiple failures
        if symbols_failing_multiple > 0:
            sample_symbols = list(debug_info.keys())[:5]
            logger.info("\nSample symbols failing multiple criteria:")
            for symbol in sample_symbols:
                if len(debug_info[symbol]) > 1:
                    logger.info(f"{symbol}: {list(debug_info[symbol].keys())}")

    def _check_symbol_criteria_debug(self, symbol: str, **filters) -> dict:
        """
        Check symbol criteria and return detailed failure information.
        
        Returns:
            dict: Map of failed criteria with details
        """
        failures = {}
        
        try:
            data = self.db_handler.load_market_data(symbol)
            if data.empty:
                failures['data_quality'] = "No data available"
                return failures
                
            # History length check
            if len(data) < filters['min_history_days']:
                failures['history'] = f"Only {len(data)} days of history"
            
            recent_data = data.tail(20)
            latest_price = recent_data['close'].iloc[-1]
            
            # Price range check
            if not (filters['min_price'] <= latest_price <= filters['max_price']):
                failures['price_range'] = f"Price {latest_price:.2f} outside range [{filters['min_price']}, {filters['max_price']}]"
            
            # Volume check
            avg_volume = recent_data['volume'].mean()
            if avg_volume < filters['min_volume']:
                failures['volume'] = f"Volume {avg_volume:.0f} below minimum {filters['min_volume']}"
            
            # Dollar volume check
            avg_dollar_volume = (recent_data['close'] * recent_data['volume']).mean()
            if avg_dollar_volume < filters['min_dollar_volume']:
                failures['dollar_volume'] = f"Dollar volume {avg_dollar_volume:.0f} below minimum"
            
            # Market cap check
            market_cap = latest_price * avg_volume
            if market_cap < filters['min_market_cap']:
                failures['market_cap'] = f"Market cap {market_cap:.0f} below minimum"
            
            # Spread check
            if 'high' in recent_data.columns and 'low' in recent_data.columns:
                avg_spread = ((recent_data['high'] - recent_data['low']) / recent_data['close']).mean()
                if avg_spread > filters['max_spread_pct']:
                    failures['spread'] = f"Spread {avg_spread:.3f} above maximum"
            
            # Trading days check
            trading_days = len(data)
            calendar_days = (data.index[-1] - data.index[0]).days
            trading_days_pct = trading_days / max(calendar_days, 1)
            if trading_days_pct < filters['min_trading_days_pct']:
                failures['trading_days'] = f"Trading days {trading_days_pct:.2%} below minimum"
            
            if not failures:
                self.stock_features[symbol] = {
                    'price': latest_price,
                    'volume': avg_volume,
                    'market_cap': market_cap,
                    'dollar_volume': avg_dollar_volume,
                    'volatility': data['close'].pct_change().std() * np.sqrt(252),
                    'beta': self._calculate_beta(data['close'].pct_change())
                }
            
            return failures
            
        except Exception as e:
            logger.warning(f"Error checking criteria for {symbol}: {e}")
            failures['data_quality'] = str(e)
            return failures

    def _get_fallback_symbols(self, symbols: List[str], debug_info: Dict) -> List[str]:
        """
        Get a fallback list of symbols when no symbols pass all filters.
        Selects symbols that fail the fewest criteria and have the most important qualities.
        """
        try:
            # Score each symbol based on how many and which criteria it fails
            symbol_scores = {}
            critical_criteria = {'data_quality', 'history', 'price_range', 'volume'}
            
            for symbol in symbols:
                if symbol not in debug_info:  # Symbol passed all criteria
                    symbol_scores[symbol] = 0
                    continue
                    
                failures = debug_info[symbol]
                
                # Higher penalty for failing critical criteria
                score = sum(2 if criterion in critical_criteria else 1 
                          for criterion in failures)
                symbol_scores[symbol] = score
            
            # Take the top 50 symbols with the lowest failure scores
            fallback_symbols = sorted(symbol_scores.keys(), 
                                   key=lambda s: symbol_scores[s])[:50]
            
            return fallback_symbols
            
        except Exception as e:
            logger.error(f"Error getting fallback symbols: {e}")
            return symbols[:50]
        
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
                    volume = data['volume'].mean()
                    
                    self.stock_features[symbol] = {
                        'volatility': returns.std() * np.sqrt(252),
                        'volume': volume,
                        'price': data['close'].iloc[-1],
                        'market_cap': data['close'].iloc[-1] * volume,
                        'beta': self._calculate_beta(returns)
                    }
                    
                except Exception as e:
                    logger.warning(f"Error calculating features for {symbol}: {e}")
                    continue

    def _calculate_beta(self, returns: pd.Series) -> float:
        """Calculate beta relative to SPY or use simplified approach."""
        try:
            # Check if we should use simplified beta
            if self.config.get('use_simplified_beta', False):
                # Use volatility as a proxy for beta
                vol = returns.std() * np.sqrt(252)
                # Scale to typical beta range
                return min(max(vol / 0.20, 0.5), 2.0)
                
            # Original SPY-based calculation
            spy_data = self.db_handler.load_market_data('SPY')
            if spy_data is None or spy_data.empty:
                logger.warning("SPY data not available - using simplified beta")
                return self._calculate_simplified_beta(returns)
                
            spy_returns = spy_data['close'].pct_change()
            
            # Align dates
            common_dates = returns.index.intersection(spy_returns.index)
            if len(common_dates) < 126:  # Reduced from 252 to 126 days minimum
                return self._calculate_simplified_beta(returns)
                
            returns = returns.loc[common_dates]
            spy_returns = spy_returns.loc[common_dates]
            
            covariance = returns.cov(spy_returns)
            variance = spy_returns.var()
            
            return covariance / variance if variance != 0 else 1.0
            
        except Exception as e:
            logger.warning(f"Error calculating beta: {e}")
            return self._calculate_simplified_beta(returns)    
        
                
    def _calculate_simplified_beta(self, returns: pd.Series) -> float:
        """Calculate a simplified beta based on volatility."""
        try:
            vol = returns.std() * np.sqrt(252)
            # Scale volatility to approximate beta
            # Typical market volatility is around 20%
            return min(max(vol / 0.20, 0.5), 2.0)
        except Exception as e:
            logger.warning(f"Error calculating simplified beta: {e}")
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
                'Large_Cap_High_Vol': [],
                'Large_Cap_Low_Vol': [],
                'Mid_Cap_High_Vol': [],
                'Mid_Cap_Low_Vol': [],
                'Small_Cap_High_Vol': [],
                'Small_Cap_Low_Vol': [],
                'High_Volatility': [],
                'Low_Volatility': [],
                'High_Volume': [],
                'Other': []
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

    def clear_cache(self) -> None:
        """Clear all cached data from memory and disk."""
        try:
            logger.info("Clearing universe cache...")
            
            # Clear memory caches
            self.universe.clear()
            self.stock_features.clear()
            self.clusters.clear()
            
            # Clear disk cache
            cache_file = self.cache_dir / 'universe.json'
            if cache_file.exists():
                cache_file.unlink()
                logger.info("Disk cache cleared successfully")
            
            # Remove any other cache files in the cache directory
            for cache_file in self.cache_dir.glob('*.json'):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete cache file {cache_file}: {e}")
            
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise

    def is_cache_valid(self) -> bool:
        """
        Check if the cached universe data is still valid.
        
        Returns:
            bool: True if cache is valid, False otherwise
        """
        try:
            cache_file = self.cache_dir / 'universe.json'
            if not cache_file.exists():
                return False
                
            with open(cache_file) as f:
                data = json.load(f)
                
            # Check if cache is recent (less than 1 day old)
            cache_time = pd.Timestamp(data['timestamp'])
            return (pd.Timestamp.now() - cache_time) <= pd.Timedelta(days=1)
            
        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False

    def refresh_cache(self) -> None:
        """Force refresh of the universe cache."""
        try:
            logger.info("Refreshing universe cache...")
            self.clear_cache()
            self.build_universe()
            logger.info("Cache refresh completed successfully")
            
        except Exception as e:
            logger.error(f"Error refreshing cache: {e}")
            raise

    def _cache_universe(self, sector_groups: Dict[str, List[str]]) -> None:
        """
        Cache universe data with improved error handling and atomic writes.
        
        Args:
            sector_groups: Dictionary of sector groups to cache
        """
        try:
            cache_file = self.cache_dir / 'universe.json'
            temp_file = self.cache_dir / 'universe.json.tmp'
            
            # Prepare cache data
            cache_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'sectors': sector_groups,
                'features': self.stock_features
            }
            
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(cache_data, f, indent=4)
                
            # Atomic rename to final cache file
            temp_file.replace(cache_file)
            
            logger.info(f"Universe cached successfully at {cache_file}")
            
        except Exception as e:
            logger.error(f"Error caching universe: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise
        """
        Cache universe data with improved error handling and atomic writes.
        
        Args:
            sector_groups: Dictionary of sector groups to cache
        """
        try:
            cache_file = self.cache_dir / 'universe.json'
            temp_file = self.cache_dir / 'universe.json.tmp'
            
            # Prepare cache data
            cache_data = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'sectors': sector_groups,
                'features': self.stock_features
            }
            
            # Write to temporary file first
            with open(temp_file, 'w') as f:
                json.dump(cache_data, f, indent=4)
                
            # Atomic rename to final cache file
            temp_file.replace(cache_file)
            
            logger.info(f"Universe cached successfully at {cache_file}")
            
        except Exception as e:
            logger.error(f"Error caching universe: {e}")
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except Exception:
                    pass
            raise

    def load_cached_universe(self) -> Optional[Dict[str, List[str]]]:
        """
        Load universe from cache with improved validation.
        
        Returns:
            Optional[Dict[str, List[str]]]: Cached universe data if valid, None otherwise
        """
        try:
            if not self.is_cache_valid():
                return None
                
            cache_file = self.cache_dir / 'universe.json'
            with open(cache_file) as f:
                data = json.load(f)
                
            # Validate cache data structure
            required_keys = {'timestamp', 'sectors', 'features'}
            if not all(key in data for key in required_keys):
                logger.warning("Cache data is missing required keys")
                return None
                
            self.stock_features = data['features']
            self.clusters = data['sectors']
            
            logger.info(f"Successfully loaded universe from cache dated {data['timestamp']}")
            return self.clusters
            
        except Exception as e:
            logger.error(f"Error loading cached universe: {e}")
            return None

    def update_universe(self) -> None:
        """Update universe and features with cache handling."""
        if not self.is_cache_valid():
            self.refresh_cache()
        else:
            logger.info("Using existing valid cache")

    def get_universe_statistics(self) -> Dict:
        """
        Get statistical summary of the current universe.
        
        Returns:
            Dict containing universe statistics
        """
        try:
            stats = {
                'total_symbols': sum(len(symbols) for symbols in self.clusters.values()),
                'symbols_per_group': {group: len(symbols) for group, symbols in self.clusters.items()},
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Calculate aggregate statistics if features are available
            if self.stock_features:
                features_df = pd.DataFrame.from_dict(self.stock_features, orient='index')
                stats.update({
                    'average_market_cap': features_df['market_cap'].mean(),
                    'median_market_cap': features_df['market_cap'].median(),
                    'average_volume': features_df['volume'].mean(),
                    'median_volume': features_df['volume'].median(),
                    'average_volatility': features_df['volatility'].mean(),
                    'average_beta': features_df['beta'].mean()
                })
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating universe statistics: {e}")
            return {}

    def get_symbols_by_criteria(self, 
                              min_market_cap: Optional[float] = None,
                              min_volume: Optional[float] = None,
                              max_volatility: Optional[float] = None,
                              beta_range: Optional[tuple] = None) -> List[str]:
        """
        Get symbols that meet specified criteria.
        
        Args:
            min_market_cap: Minimum market cap requirement
            min_volume: Minimum trading volume requirement
            max_volatility: Maximum allowed volatility
            beta_range: Tuple of (min_beta, max_beta)
            
        Returns:
            List of symbols meeting all specified criteria
        """
        try:
            filtered_symbols = []
            
            for symbol, features in self.stock_features.items():
                # Check each criterion if specified
                if min_market_cap and features['market_cap'] < min_market_cap:
                    continue
                    
                if min_volume and features['volume'] < min_volume:
                    continue
                    
                if max_volatility and features['volatility'] > max_volatility:
                    continue
                    
                if beta_range:
                    min_beta, max_beta = beta_range
                    if not (min_beta <= features['beta'] <= max_beta):
                        continue
                        
                filtered_symbols.append(symbol)
            
            logger.info(f"Found {len(filtered_symbols)} symbols meeting specified criteria")
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Error filtering symbols by criteria: {e}")
            return []

    def get_group_composition(self, group_name: str) -> Dict:
        """
        Get detailed composition of a specific group.
        
        Args:
            group_name: Name of the group to analyze
            
        Returns:
            Dict containing group statistics and characteristics
        """
        try:
            if group_name not in self.clusters:
                logger.warning(f"Group {group_name} not found in clusters")
                return {}
                
            symbols = self.clusters[group_name]
            if not symbols:
                return {'symbol_count': 0}
                
            # Get features for symbols in this group
            group_features = {
                symbol: self.stock_features[symbol]
                for symbol in symbols
                if symbol in self.stock_features
            }
            
            if not group_features:
                return {'symbol_count': len(symbols)}
                
            features_df = pd.DataFrame.from_dict(group_features, orient='index')
            
            composition = {
                'symbol_count': len(symbols),
                'average_market_cap': features_df['market_cap'].mean(),
                'median_market_cap': features_df['market_cap'].median(),
                'average_volume': features_df['volume'].mean(),
                'average_volatility': features_df['volatility'].mean(),
                'average_beta': features_df['beta'].mean(),
                'volatility_range': (features_df['volatility'].min(), features_df['volatility'].max()),
                'beta_range': (features_df['beta'].min(), features_df['beta'].max())
            }
            
            return composition
            
        except Exception as e:
            logger.error(f"Error analyzing group composition: {e}")
            return {}