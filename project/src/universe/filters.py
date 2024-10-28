"""
Implements filtering criteria for stock universe selection.
Handles data quality, liquidity, and tradability filters.
"""

import logging
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class FilterConfig:
    """Configuration for universe filters."""
    min_price: float = 1.0  # Reduced from 5.0 to include more stocks
    max_price: float = 10000.0
    min_volume: int = 10000  # Reduced from 100000
    min_market_cap: float = 50000000  # Reduced from 500M to 50M
    min_history_days: int = 126  # Reduced from 252 (6 months instead of 1 year)
    min_dollar_volume: float = 100000  # Reduced from 1M to 100K daily
    max_spread_pct: float = 0.05  # Increased from 0.02 to 0.05 (5%)
    min_trading_days_pct: float = 0.80  # Reduced from 0.95 to 0.80 (80%)
    max_gap_days: int = 10  # Increased from 5 to 10
    min_price_std: float = 0.0005  # Reduced from 0.001

class UniverseFilter:
    """
    Implements comprehensive filtering for stock universe selection with more lenient criteria.
    """
    
    def __init__(self, config: FilterConfig = None):
        self.config = config or FilterConfig()
        self.filter_stats: Dict[str, Dict] = {}
        
    def apply_filters(self, 
                     symbols: List[str],
                     market_data: Dict[str, pd.DataFrame],
                     features: Dict[str, Dict]) -> List[str]:
        """Apply filters to symbol list with progress tracking."""
        try:
            logger.info(f"Applying filters to {len(symbols)} symbols...")
            filtered_symbols = []
            total = len(symbols)
            
            for i, symbol in enumerate(symbols, 1):
                try:
                    data = market_data.get(symbol)
                    if data is None or data.empty:
                        self._log_filter_failure(symbol, "no_data")
                        continue
                    
                    # Apply filters with more detailed logging
                    if not self._apply_all_filters(symbol, data, features.get(symbol, {})):
                        continue
                    
                    filtered_symbols.append(symbol)
                    
                    # Log progress every 100 symbols
                    if i % 100 == 0:
                        logger.info(f"Processed {i}/{total} symbols. Currently filtered: {len(filtered_symbols)}")
                    
                except Exception as e:
                    logger.warning(f"Error filtering {symbol}: {e}")
                    self._log_filter_failure(symbol, "error")
                    continue
            
            # Log detailed filter statistics
            self._log_filter_statistics()
            logger.info(f"Filtered to {len(filtered_symbols)} symbols")
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Error applying filters: {e}")
            raise

    def _apply_all_filters(self,
                          symbol: str,
                          data: pd.DataFrame,
                          features: Dict) -> bool:
        """Apply all filters with more lenient criteria."""
        try:
            # Basic data quality check (most lenient)
            if not self._check_data_quality(data):
                self._log_filter_failure(symbol, "data_quality")
                return False
            
            # Price filters
            if not self._check_price_filters(data):
                self._log_filter_failure(symbol, "price")
                return False
            
            # Volume filters (with adaptive thresholds)
            if not self._check_volume_filters(data):
                self._log_filter_failure(symbol, "volume")
                return False
            
            # Market cap filter (with size-based adjustment)
            if not self._check_market_cap(features):
                self._log_filter_failure(symbol, "market_cap")
                return False
            
            # Trading consistency filters (more lenient)
            if not self._check_trading_consistency(data):
                self._log_filter_failure(symbol, "consistency")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error in filter application for {symbol}: {e}")
            return False

    def _check_data_quality(self, data: pd.DataFrame) -> bool:
        """Check basic data quality with more lenient criteria."""
        try:
            # Check minimum history length (reduced requirement)
            if len(data) < self.config.min_history_days:
                return False
            
            # Check for required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in data.columns for col in required_columns):
                return False
            
            # More lenient missing value check (allow up to 20% missing)
            missing_pct = data[required_columns].isnull().mean()
            if any(missing_pct > 0.20):
                return False
            
            # Basic price integrity check
            valid_rows = (
                (data['high'] >= data['low']) &
                (data['close'] >= data['low']) &
                (data['close'] <= data['high'])
            ).mean()
            
            return valid_rows >= 0.8  # Allow up to 20% invalid rows
            
        except Exception:
            return False

    def _check_price_filters(self, data: pd.DataFrame) -> bool:
        """Check price-based filters with adaptive thresholds."""
        try:
            recent_data = data.tail(20)  # Last month of trading
            
            avg_price = recent_data['close'].mean()
            if not (self.config.min_price <= avg_price <= self.config.max_price):
                return False
            
            # Adaptive price volatility check based on price level
            price_std = recent_data['close'].std() / avg_price  # Use relative volatility
            min_std_threshold = self.config.min_price_std * (1 + np.log10(max(1, avg_price)))
            
            if price_std < min_std_threshold:
                return False
            
            return True
            
        except Exception:
            return False

    def _check_volume_filters(self, data: pd.DataFrame) -> bool:
        """Check volume-based filters with price-adjusted thresholds."""
        try:
            recent_data = data.tail(20)
            
            # Calculate average price-adjusted volume metrics
            avg_price = recent_data['close'].mean()
            avg_volume = recent_data['volume'].mean()
            avg_dollar_volume = avg_price * avg_volume
            
            # Adjust minimum volume threshold based on price
            adjusted_min_volume = self.config.min_volume * (1 / max(1, np.log10(avg_price)))
            
            if avg_volume < adjusted_min_volume:
                return False
            
            if avg_dollar_volume < self.config.min_dollar_volume:
                return False
            
            return True
            
        except Exception:
            return False
    def _log_filter_failure(self, symbol: str, reason: str) -> None:
        """
        Log filter failures for analysis.
        
        Args:
            symbol: Stock symbol that failed the filter
            reason: Reason for the filter failure
        """
        try:
            if reason not in self.filter_stats:
                self.filter_stats[reason] = {
                    'count': 0,
                    'symbols': []
                }
            
            self.filter_stats[reason]['count'] += 1
            self.filter_stats[reason]['symbols'].append(symbol)
            
            # Log at debug level to avoid cluttering the log
            logger.debug(f"Filter failure for {symbol}: {reason}")
            
            # Keep symbol list manageable
            max_symbols_per_reason = 100
            if len(self.filter_stats[reason]['symbols']) > max_symbols_per_reason:
                self.filter_stats[reason]['symbols'] = (
                    self.filter_stats[reason]['symbols'][:max_symbols_per_reason]
                )
                
        except Exception as e:
            logger.error(f"Error logging filter failure: {e}")
    def _check_market_cap(self, features: Dict) -> bool:
        """
        Check market cap with sector-based and market condition adjustments.
        
        The method adjusts minimum market cap requirements based on:
        1. Sector averages (technology companies tend to have higher market caps than utilities)
        2. Market conditions (bear markets may have lower caps due to price declines)
        3. Company age (newer companies may have lower caps)
        4. Growth rates (high-growth companies might be allowed lower caps)
        
        Args:
            features: Dictionary containing stock features and metrics
            
        Returns:
            bool: Whether the stock passes market cap requirements
        """
        try:
            market_cap = features.get('market_cap', 0)
            if market_cap == 0:
                return False
                
            # Get sector and base sector adjustments
            sector = features.get('sector', 'Unknown')
            sector_adjustments = {
                'Technology': 1.2,      # Tech companies usually have higher caps
                'Healthcare': 1.1,      # Healthcare companies tend to be larger
                'Utilities': 0.8,       # Utilities can be smaller
                'Financial': 1.0,       # Financials at baseline
                'Energy': 0.9,          # Energy companies can be smaller
                'Consumer Cyclical': 0.85,
                'Consumer Defensive': 0.9,
                'Basic Materials': 0.8,
                'Communication Services': 1.1,
                'Industrial': 0.9,
                'Real Estate': 0.8,
                'Unknown': 1.0
            }
            
            # Get market condition and adjust accordingly
            market_condition = features.get('market_condition', 'normal')
            market_condition_adjustments = {
                'bull': 1.2,            # Higher requirements in bull markets
                'bear': 0.8,            # Lower requirements in bear markets
                'normal': 1.0,
                'high_volatility': 0.9,
                'low_volatility': 1.1
            }
            
            # Get company age and growth metrics if available
            company_age = features.get('years_since_ipo', 10)  # Default to 10 years
            revenue_growth = features.get('revenue_growth', 0.0)
            
            # Calculate age adjustment (newer companies can have lower caps)
            age_adjustment = min(1.0, 0.7 + (company_age / 10))
            
            # Calculate growth adjustment (high growth allows lower caps)
            growth_adjustment = max(0.7, 1.0 - (revenue_growth / 2)) if revenue_growth > 0 else 1.0
            
            # Get sector adjustment factor
            sector_factor = sector_adjustments.get(sector, 1.0)
            
            # Get market condition adjustment factor
            market_factor = market_condition_adjustments.get(market_condition, 1.0)
            
            # Calculate final adjusted minimum market cap
            adjusted_min_cap = (
                self.config.min_market_cap *
                sector_factor *
                market_factor *
                age_adjustment *
                growth_adjustment
            )
            
            # Add some logging for transparency
            if market_cap < adjusted_min_cap:
                logger.debug(
                    f"Market cap filter: {market_cap:,.0f} < {adjusted_min_cap:,.0f} "
                    f"(Sector: {sector}, Condition: {market_condition}, "
                    f"Age: {company_age}, Growth: {revenue_growth:.1%})"
                )
                
            # Allow very high growth companies to bypass market cap requirements
            if revenue_growth > 1.0:  # More than 100% growth
                return True
                
            return market_cap >= adjusted_min_cap
            
        except Exception as e:
            logger.error(f"Error in market cap check: {e}")
            return False

    def _check_trading_consistency(self, data: pd.DataFrame) -> bool:
        """Check trading consistency with more lenient criteria."""
        try:
            # Check trading frequency with rolling window
            rolling_volume = data['volume'].rolling(window=20).mean()
            active_days = (rolling_volume > 0).mean()
            
            if active_days < self.config.min_trading_days_pct:
                return False
            
            # Check for trading gaps with more tolerance
            volume_series = data['volume'] > 0
            gap_mask = ~volume_series
            if gap_mask.any():
                gaps = gap_mask.astype(int).groupby(gap_mask.cumsum()).sum()
                if (gaps > self.config.max_gap_days).any():
                    return False
            
            return True
            
        except Exception:
            return False

    def _log_filter_statistics(self) -> None:
        """Log detailed statistics about filter results."""
        total_filtered = sum(stats['count'] for stats in self.filter_stats.values())
        
        logger.info("Filter Statistics:")
        for reason, stats in self.filter_stats.items():
            percentage = (stats['count'] / total_filtered * 100) if total_filtered > 0 else 0
            logger.info(f"{reason}: {stats['count']} symbols ({percentage:.1f}%)")

    def get_filter_summary(self) -> pd.DataFrame:
        """Get detailed summary of filter results."""
        summary = []
        total_filtered = sum(stats['count'] for stats in self.filter_stats.values())
        
        for reason, stats in self.filter_stats.items():
            percentage = (stats['count'] / total_filtered * 100) if total_filtered > 0 else 0
            summary.append({
                'reason': reason,
                'count': stats['count'],
                'percentage': percentage,
                'examples': stats['symbols'][:5]  # Show first 5 examples
            })
        
        return pd.DataFrame(summary)