"""
Universe management implementation for NYSE trading.
Handles stock filtering, universe construction, and updates.
"""

import json
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from project.data.dbsetup import DatabaseSetup
from project.src.core.database_handler import DatabaseHandler
from project.src.indicators.calculator import IndicatorCalculator

logger = logging.getLogger(__name__)


class UniverseManager:
    """Manages stock universe selection and clustering."""

    def __init__(self, db_path: str, config_path: str, cache_dir: str = "cache"):
        """
        Initialize UniverseManager.

        Args:
            db_path: Path to SQLite database
            config_path: Path to configuration JSON file
            cache_dir: Directory for caching universe data
        """
        # Initialize containers first
        self.universe = {}  # Initialize this first!
        self.stock_features = {}
        self.clusters = {}

        # Then initialize other attributes
        self.db_path = db_path
        self.db_handler = DatabaseHandler(db_path)
        self.db_setup = DatabaseSetup(db_path)
        self.indicator_calculator = IndicatorCalculator()

        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)

        # Set default filters
        self._set_default_filters()

        # Setup cache directory
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Initialize market regime attributes
        self.current_market_regime = "Undefined"  # Default regime
        self.regime_history = []
        self.regime_lookback = self.config.get(
            "regime_lookback", 200
        )  # Minimum needed for SMA200

        # Market regime thresholds from config or defaults
        self.regime_thresholds = self.config.get(
            "regime_thresholds",
            {
                "momentum_threshold": 0.05,
                "volume_high_mult": 1.5,
                "volume_low_mult": 0.5,
                "volatility_high_quantile": 0.75,
                "volatility_low_quantile": 0.25,
            },
        )

        # Initial market regime calculation
        self._calculate_current_market_regime()

        # Validate SPY data availability last
        if not self._validate_spy_data():
            raise ValueError("SPY data not available - required for beta calculations")

    def _calculate_current_market_regime(self) -> str:
        """
        Calculate the current market regime based on market indicators.
        Updates self.current_market_regime and self.regime_history.

        Returns:
            str: Current market regime classification
        """
        try:
            # Get SPY data
            spy_data = self.db_handler.load_market_data("SPY")
            if spy_data is None or spy_data.empty:
                logger.error("SPY data not available for market regime calculation")
                return self.current_market_regime

            # Calculate returns
            spy_data["Returns"] = spy_data["close"].pct_change()

            # Calculate indicators
            spy_data["SMA50"] = spy_data["close"].rolling(window=50).mean()
            spy_data["SMA200"] = spy_data["close"].rolling(window=200).mean()
            spy_data["Volatility"] = spy_data["Returns"].rolling(
                window=20
            ).std() * np.sqrt(252)
            spy_data["volume_MA"] = spy_data["volume"].rolling(window=20).mean()
            spy_data["Momentum"] = spy_data["close"].pct_change(periods=20)

            # Get latest data point for classification
            current = spy_data.iloc[-1]

            # Classify trend
            if current["SMA50"] > current["SMA200"]:
                trend = "Uptrend"
            elif current["SMA50"] < current["SMA200"]:
                trend = "Downtrend"
            else:
                trend = "Sideways"

            # Classify volatility
            vol_high = spy_data["Volatility"].quantile(
                self.regime_thresholds["volatility_high_quantile"]
            )
            vol_low = spy_data["Volatility"].quantile(
                self.regime_thresholds["volatility_low_quantile"]
            )

            if current["Volatility"] > vol_high:
                volatility = "High"
            elif current["Volatility"] < vol_low:
                volatility = "Low"
            else:
                volatility = "Medium"

            # Classify volume
            if (
                current["volume"]
                > current["volume_MA"] * self.regime_thresholds["volume_high_mult"]
            ):
                volume = "High"
            elif (
                current["volume"]
                < current["volume_MA"] * self.regime_thresholds["volume_low_mult"]
            ):
                volume = "Low"
            else:
                volume = "Normal"

            # Classify momentum
            mom_threshold = self.regime_thresholds["momentum_threshold"]
            if current["Momentum"] > mom_threshold:
                momentum = "Strong"
            elif current["Momentum"] < -mom_threshold:
                momentum = "Weak"
            else:
                momentum = "Neutral"

            # Combine classifications
            regime = (
                f"{trend}-{volatility}_Volatility-{volume}_Volume-{momentum}_Momentum"
            )

            # Update regime history and current regime
            self.regime_history.append((pd.Timestamp.now(), regime))
            self.current_market_regime = regime

            logger.info(f"Current market regime calculated: {regime}")
            return regime

        except Exception as e:
            logger.error(f"Error calculating market regime: {e}")
            return self.current_market_regime

    def _validate_spy_data(self) -> bool:
        """
        Ensure SPY data is available and valid.

        Returns:
            bool: True if SPY data meets requirements
        """
        try:
            spy_data = self.db_handler.load_market_data("SPY")
            if spy_data is None or spy_data.empty:
                logger.error("SPY data not found in database")
                return False

            # Store SPY data in universe
            self.universe["SPY"] = spy_data

            # Require at least one year of data
            if len(spy_data) < 252:
                logger.error("Insufficient SPY history for beta calculations")
                return False

            return True
        except Exception as e:
            logger.error(f"Error validating SPY data: {e}")
            return False

    def validate_required_data(self):
        """Validate all required data is available before training."""
        required_symbols = ["SPY"] + list(self.universe_manager.clusters.keys())
        missing_symbols = []

        for symbol in required_symbols:
            try:
                data = self.db_handler.load_market_data(symbol)
                if data is None or data.empty:
                    missing_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")
                missing_symbols.append(symbol)

        if missing_symbols:
            raise ValueError(f"Missing required data for symbols: {missing_symbols}")

    def _set_default_filters(self) -> None:
        """Set reasonable default filter values if not specified in config."""
        default_filters = {
            "min_price": 1.0,
            "max_price": 10000.0,
            "min_volume": 5000,
            "min_market_cap": 1000000,
            "min_history_days": 126,
            "min_dollar_volume": 50000,
            "max_spread_pct": 0.05,
            "min_trading_days_pct": 0.60,
            "max_gap_days": 7,
            "min_price_std": 0.0005,
        }

        if "universe_filters" not in self.config:
            self.config["universe_filters"] = {}

        # Only set defaults for missing values
        for key, value in default_filters.items():
            if key not in self.config["universe_filters"]:
                self.config["universe_filters"][key] = value

    def build_universe(self) -> Dict[str, List[str]]:
        """
        Build and filter trading universe.

        Returns:
            Dictionary of filtered symbols by category
        """
        try:
            logger.info("Building trading universe...")

            # Always include SPY first
            if not self._validate_spy_data():
                raise ValueError("SPY data not available or invalid")

            # Get NYSE symbols and filter for availability
            all_symbols = self.db_setup.get_nyse_symbols()
            available_symbols = self._get_available_symbols_from_db()

            # Ensure SPY is in the available symbols
            if "SPY" not in available_symbols:
                logger.error("SPY not found in available symbols")
                raise ValueError("SPY data not found in database")

            symbols_in_db = [
                symbol for symbol in all_symbols if symbol in available_symbols
            ]
            if "SPY" not in symbols_in_db:
                symbols_in_db.append("SPY")  # Explicitly add SPY if not in NYSE list

            logger.info(f"Found {len(symbols_in_db)} symbols available in database")

            # Apply filters and group symbols, ensuring SPY is preserved
            tradable_symbols = self._apply_filters(symbols_in_db)

            # Double check SPY presence after filtering
            if "SPY" not in tradable_symbols:
                logger.warning("SPY missing after filtering, adding back")
                tradable_symbols.append("SPY")

            self._calculate_stock_features(
                ["SPY"] + [s for s in tradable_symbols if s != "SPY"]
            )
            self.clusters = self._group_by_sector(tradable_symbols)

            # Cache results
            self._cache_universe(self.clusters)

            logger.info(
                f"Universe built with {len(tradable_symbols)} symbols (including SPY)"
            )
            return self.clusters

        except Exception as e:
            logger.error(f"Error building universe: {e}")
            raise

    def _get_available_symbols_from_db(self) -> List[str]:
        """
        Retrieve list of symbols available in database.

        Returns:
            List of available symbol names
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                return [table[0] for table in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error retrieving symbols from database: {e}")
            return []

    def _apply_filters(self, symbols: List[str]) -> List[str]:
        """
        Apply filtering criteria to symbol list.

        Args:
            symbols: List of symbols to filter

        Returns:
            List of symbols that pass all filters
        """
        try:
            logger.info(f"Starting filtering process with {len(symbols)} symbols")

            # Handle SPY separately
            symbols = list(symbols)  # Convert to list to avoid modifying original
            if "SPY" in symbols:
                symbols.remove("SPY")
                logger.info("Temporarily removed SPY for filtering")

            filters = self.config.get("universe_filters", {})
            remaining_symbols = set(symbols)
            filter_stats = {
                criterion: {"count": 0, "symbols": set()}
                for criterion in filters.keys()
            }

            # Process symbols in parallel (excluding SPY)
            with ThreadPoolExecutor() as executor:
                future_to_symbol = {
                    executor.submit(
                        self._check_symbol_criteria_debug, symbol, **filters
                    ): symbol
                    for symbol in symbols
                }

                for future in future_to_symbol:
                    symbol = future_to_symbol[future]
                    try:
                        failures = future.result()
                        if failures:
                            for failure_reason in failures:
                                filter_stats[failure_reason]["count"] += 1
                                filter_stats[failure_reason]["symbols"].add(symbol)
                            remaining_symbols.remove(symbol)
                    except Exception as e:
                        logger.warning(f"Error processing {symbol}: {e}")

            passed_symbols = list(remaining_symbols)

            # Handle small universe case
            if len(passed_symbols) < 50:
                logger.warning("Few symbols passed filters - using fallback criteria")
                passed_symbols = self._get_fallback_symbols(symbols, filter_stats)

            # Always add SPY first in the list
            passed_symbols = ["SPY"] + passed_symbols
            logger.info(
                f"Filtering complete. {len(passed_symbols)} symbols passed (including SPY)"
            )

            return passed_symbols

        except Exception as e:
            logger.error(f"Error in _apply_filters: {e}")
            return ["SPY"] + symbols[:49]  # Ensure SPY is always first

    def _check_symbol_criteria_debug(self, symbol: str, **filters) -> dict:
        """
        Check symbol against filtering criteria.

        Args:
            symbol: Symbol to check
            **filters: Filter criteria from config

        Returns:
            dict: Map of failed criteria with details
        """
        failures = {}

        try:
            data = self.db_handler.load_market_data(symbol)
            if data.empty:
                failures["data_quality"] = "No data available"
                return failures

            recent_data = data.tail(20)  # Last month of trading
            latest_price = recent_data["close"].iloc[-1]
            avg_volume = recent_data["volume"].mean()

            # Basic checks
            checks = {
                "history": (
                    len(data) >= filters["min_history_days"],
                    f"Only {len(data)} days of history",
                ),
                "price_range": (
                    filters["min_price"] <= latest_price <= filters["max_price"],
                    f"Price {latest_price:.2f} outside range",
                ),
                "volume": (
                    avg_volume >= filters["min_volume"],
                    f"Volume {avg_volume:.0f} below minimum",
                ),
            }

            # Add failures for failed checks
            failures.update(
                {key: msg for key, (check, msg) in checks.items() if not check}
            )

            # Additional calculations only if basic checks pass
            if not failures:
                # Calculate and store features
                self.stock_features[symbol] = {
                    "price": latest_price,
                    "volume": avg_volume,
                    "market_cap": latest_price * avg_volume,
                    "dollar_volume": latest_price * avg_volume,
                    "volatility": data["close"].pct_change().std() * np.sqrt(252),
                    "beta": self._calculate_beta(data["close"].pct_change()),
                }

            return failures

        except Exception as e:
            logger.warning(f"Error checking criteria for {symbol}: {e}")
            failures["data_quality"] = str(e)
            return failures

    def _calculate_stock_features(self, symbols: List[str]) -> None:
        """
        Calculate features for stock classification.

        Args:
            symbols: List of symbols to calculate features for
        """
        # Process SPY first if present
        if "SPY" in symbols and "SPY" not in self.stock_features:
            try:
                spy_data = self.db_handler.load_market_data("SPY")
                if not spy_data.empty:
                    returns = spy_data["close"].pct_change()
                    self.stock_features["SPY"] = {
                        "price": spy_data["close"].iloc[-1],
                        "volume": spy_data["volume"].mean(),
                        "market_cap": float("inf"),  # SPY is treated specially
                        "dollar_volume": float("inf"),
                        "volatility": returns.std() * np.sqrt(252),
                        "beta": 1.0,  # SPY is the market benchmark
                    }
                else:
                    raise ValueError("SPY data not available")
            except Exception as e:
                logger.error(f"Error calculating SPY features: {e}")
                raise ValueError("SPY data not found in database")

        # Process rest of symbols
        for symbol in symbols:
            if symbol == "SPY" or symbol in self.stock_features:
                continue

            try:
                data = self.db_handler.load_market_data(symbol)
                if data.empty:
                    continue

                recent_data = data.tail(60)  # Last quarter
                returns = data["close"].pct_change()
                avg_volume = recent_data["volume"].mean()
                latest_price = recent_data["close"].iloc[-1]

                self.stock_features[symbol] = {
                    "price": latest_price,
                    "volume": avg_volume,
                    "market_cap": latest_price * avg_volume,
                    "dollar_volume": latest_price * avg_volume,
                    "volatility": returns.std() * np.sqrt(252),
                    "beta": self._calculate_beta(returns),
                }

            except Exception as e:
                logger.warning(f"Error calculating features for {symbol}: {e}")
                continue

    def _calculate_simplified_beta(self, returns: pd.Series) -> float:
        """
        Calculate simplified beta based on volatility.

        Args:
            returns: Series of asset returns

        Returns:
            float: Approximated beta value
        """
        try:
            vol = returns.std() * np.sqrt(252)
            # Scale to typical beta range using 20% as market volatility baseline
            return min(max(vol / 0.20, 0.5), 2.0)
        except Exception as e:
            logger.warning(f"Error calculating simplified beta: {e}")
            return 1.0

    def _calculate_stock_features(self, symbols: List[str]) -> None:
        """
        Calculate features for stock classification.

        Args:
            symbols: List of symbols to calculate features for
        """
        for symbol in symbols:
            if symbol in self.stock_features:  # Skip if already calculated
                continue

            try:
                data = self.db_handler.load_market_data(symbol)
                if data.empty:
                    continue

                recent_data = data.tail(60)  # Last quarter
                returns = data["close"].pct_change()
                avg_volume = recent_data["volume"].mean()
                latest_price = recent_data["close"].iloc[-1]

                self.stock_features[symbol] = {
                    "price": latest_price,
                    "volume": avg_volume,
                    "market_cap": latest_price * avg_volume,
                    "dollar_volume": latest_price * avg_volume,
                    "volatility": returns.std() * np.sqrt(252),
                    "beta": self._calculate_beta(returns),
                }

            except Exception as e:
                logger.warning(f"Error calculating features for {symbol}: {e}")
                continue

    def _group_by_sector(self, symbols: List[str]) -> Dict[str, List[str]]:
        """
        Group symbols based on their characteristics.

        Args:
            symbols: List of symbols to group

        Returns:
            Dict mapping group names to lists of symbols
        """
        try:
            logger.info("Grouping symbols based on trading characteristics...")

            groups = {
                "Large_Cap_High_Vol": [],
                "Large_Cap_Low_Vol": [],
                "Mid_Cap_High_Vol": [],
                "Mid_Cap_Low_Vol": [],
                "Small_Cap_High_Vol": [],
                "Small_Cap_Low_Vol": [],
                "High_Volatility": [],
                "High_Volume": [],
                "Other": [],
            }

            if not self.stock_features:
                logger.warning("No feature data available for grouping")
                return {"Other": symbols}

            # Convert features to DataFrame for easier analysis
            features_df = pd.DataFrame.from_dict(self.stock_features, orient="index")

            # Calculate thresholds
            market_cap_thresholds = features_df["market_cap"].quantile([0.33, 0.67])
            volume_threshold = (
                features_df["avg_volume"].median()
                if "avg_volume" in features_df
                else features_df["volume"].median()
            )
            volatility_threshold = features_df["volatility"].quantile(0.75)

            # Classify symbols
            for symbol in symbols:
                if symbol not in self.stock_features:
                    groups["Other"].append(symbol)
                    continue

                metrics = self.stock_features[symbol]

                # High volatility stocks get their own category
                if metrics["volatility"] > volatility_threshold:
                    groups["High_Volatility"].append(symbol)
                    continue

                # Classify based on market cap and volume
                is_high_volume = metrics["volume"] > volume_threshold

                if metrics["market_cap"] > market_cap_thresholds[0.67]:
                    target_group = (
                        "Large_Cap_High_Vol" if is_high_volume else "Large_Cap_Low_Vol"
                    )
                elif metrics["market_cap"] > market_cap_thresholds[0.33]:
                    target_group = (
                        "Mid_Cap_High_Vol" if is_high_volume else "Mid_Cap_Low_Vol"
                    )
                else:
                    target_group = (
                        "Small_Cap_High_Vol" if is_high_volume else "Small_Cap_Low_Vol"
                    )

                groups[target_group].append(symbol)

                # Additional high volume category for very liquid stocks
                if metrics["volume"] > features_df["volume"].quantile(0.9):
                    groups["High_Volume"].append(symbol)

            # Log group distributions
            for group_name, group_symbols in groups.items():
                if group_symbols:
                    logger.info(f"{group_name}: {len(group_symbols)} symbols")

            # Remove empty groups
            return {k: v for k, v in groups.items() if v}

        except Exception as e:
            logger.error(f"Error in group creation: {e}")
            return {"Other": symbols}

    def _cache_universe(self, sector_groups: Dict[str, List[str]]) -> None:
        """
        Cache universe data with atomic writes.

        Args:
            sector_groups: Dictionary of sector groups to cache
        """
        try:
            cache_file = self.cache_dir / "universe.json"
            temp_file = self.cache_dir / "universe.json.tmp"

            cache_data = {
                "timestamp": pd.Timestamp.now().isoformat(),
                "sectors": sector_groups,
                "features": self.stock_features,
            }

            # Write to temporary file first
            with open(temp_file, "w") as f:
                json.dump(cache_data, f, indent=4)

            # Atomic rename to final cache file
            temp_file.replace(cache_file)
            logger.info(f"Universe cached successfully at {cache_file}")

        except Exception as e:
            logger.error(f"Error caching universe: {e}")
            if temp_file.exists():
                temp_file.unlink()
            raise

    def load_cached_universe(self) -> Optional[Dict[str, List[str]]]:
        """
        Load universe from cache if valid.

        Returns:
            Optional[Dict[str, List[str]]]: Cached universe data if valid, None otherwise
        """
        try:
            if not self.is_cache_valid():
                return None

            cache_file = self.cache_dir / "universe.json"
            with open(cache_file) as f:
                data = json.load(f)

            # Validate cache data structure
            required_keys = {"timestamp", "sectors", "features"}
            if not all(key in data for key in required_keys):
                logger.warning("Cache data is missing required keys")
                return None

            self.stock_features = data["features"]
            self.clusters = data["sectors"]

            logger.info(f"Loaded universe from cache dated {data['timestamp']}")
            return self.clusters

        except Exception as e:
            logger.error(f"Error loading cached universe: {e}")
            return None

    def is_cache_valid(self) -> bool:
        """
        Check if cached universe data is still valid.

        Returns:
            bool: True if cache is valid and recent, False otherwise
        """
        try:
            cache_file = self.cache_dir / "universe.json"
            if not cache_file.exists():
                return False

            with open(cache_file) as f:
                data = json.load(f)

            # Check if cache is recent (less than 1 day old)
            cache_time = pd.Timestamp(data["timestamp"])
            return (pd.Timestamp.now() - cache_time) <= pd.Timedelta(days=1)

        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False

    def clear_cache(self) -> None:
        """Clear all cached data from memory and disk."""
        try:
            logger.info("Clearing universe cache...")

            # Clear memory caches
            self.universe.clear()
            self.stock_features.clear()
            self.clusters.clear()

            # Clear disk cache
            cache_file = self.cache_dir / "universe.json"
            if cache_file.exists():
                cache_file.unlink()

            logger.info("Cache cleared successfully")

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            raise

    def get_universe_statistics(self) -> Dict:
        """
        Get statistical summary of the current universe.

        Returns:
            Dict containing universe statistics
        """
        try:
            stats = {
                "total_symbols": sum(
                    len(symbols) for symbols in self.clusters.values()
                ),
                "symbols_per_group": {
                    group: len(symbols) for group, symbols in self.clusters.items()
                },
                "timestamp": pd.Timestamp.now().isoformat(),
            }

            if self.stock_features:
                features_df = pd.DataFrame.from_dict(
                    self.stock_features, orient="index"
                )
                stats.update(
                    {
                        "avg_market_cap": features_df["market_cap"].mean(),
                        "median_market_cap": features_df["market_cap"].median(),
                        "avg_volume": features_df["volume"].mean(),
                        "avg_volatility": features_df["volatility"].mean(),
                        "avg_beta": features_df["beta"].mean(),
                    }
                )

            return stats

        except Exception as e:
            logger.error(f"Error calculating universe statistics: {e}")
            return {}

    def get_symbols_by_criteria(
        self,
        min_market_cap: Optional[float] = None,
        min_volume: Optional[float] = None,
        max_volatility: Optional[float] = None,
        beta_range: Optional[tuple] = None,
    ) -> List[str]:
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
        filtered_symbols = []

        try:
            for symbol, features in self.stock_features.items():
                if min_market_cap and features["market_cap"] < min_market_cap:
                    continue
                if min_volume and features["volume"] < min_volume:
                    continue
                if max_volatility and features["volatility"] > max_volatility:
                    continue
                if beta_range:
                    min_beta, max_beta = beta_range
                    if not min_beta <= features["beta"] <= max_beta:
                        continue
                filtered_symbols.append(symbol)

            logger.info(f"Found {len(filtered_symbols)} symbols meeting criteria")

        except Exception as e:
            logger.error(f"Error filtering symbols by criteria: {e}")

        return filtered_symbols

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
                logger.warning(f"Group {group_name} not found")
                return {}

            symbols = self.clusters[group_name]
            if not symbols:
                return {"symbol_count": 0}

            # Get features for symbols in this group
            group_features = {
                symbol: self.stock_features[symbol]
                for symbol in symbols
                if symbol in self.stock_features
            }

            if not group_features:
                return {"symbol_count": len(symbols)}

            features_df = pd.DataFrame.from_dict(group_features, orient="index")

            return {
                "symbol_count": len(symbols),
                "avg_market_cap": features_df["market_cap"].mean(),
                "median_market_cap": features_df["market_cap"].median(),
                "avg_volume": features_df["volume"].mean(),
                "avg_volatility": features_df["volatility"].mean(),
                "avg_beta": features_df["beta"].mean(),
                "volatility_range": (
                    features_df["volatility"].min(),
                    features_df["volatility"].max(),
                ),
                "beta_range": (features_df["beta"].min(), features_df["beta"].max()),
            }

        except Exception as e:
            logger.error(f"Error analyzing group composition: {e}")
            return {}

    def _calculate_beta(self, returns: pd.Series) -> float:
        """
        Calculate beta relative to SPY.

        Args:
            returns: Series of asset returns to calculate beta for

        Returns:
            float: Calculated beta value, defaults to 1.0 if calculation fails

        Raises:
            Exception: If data loading or calculation fails
        """
        try:
            # Load SPY data
            spy_data = self.db_handler.load_market_data("SPY")
            spy_returns = spy_data["close"].pct_change()

            # Align dates
            common_dates = returns.index.intersection(spy_returns.index)
            if len(common_dates) < 252:  # Require at least 1 year of data
                logger.warning(
                    "Insufficient overlapping data points for beta calculation"
                )
                return 1.0

            # Get returns for common dates
            returns = returns.loc[common_dates]
            spy_returns = spy_returns.loc[common_dates]

            # Calculate beta
            covariance = returns.cov(spy_returns)
            variance = spy_returns.var()

            # Handle zero variance case
            beta = covariance / variance if variance != 0 else 1.0

            # Validate the beta is reasonable
            if not -5 < beta < 5:  # Cap extreme values
                logger.warning(
                    f"Beta calculation produced extreme value: {beta}, defaulting to 1.0"
                )
                return 1.0

            return beta

        except Exception as e:
            logger.warning(f"Error calculating beta: {e}")
            return 1.0  # Default to market beta on error
