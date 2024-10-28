import logging
from typing import Dict, List, Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

class ExposureTracker:
    """Tracks portfolio exposure across sectors and asset classes"""
    
    def __init__(self):
        self.sector_exposure: Dict[str, float] = {}
        self.asset_class_exposure: Dict[str, float] = {}
        self._symbol_info: Dict[str, Dict] = {}
        
    def update_exposures(self, positions: Dict[str, float]) -> None:
        """
        Update exposure calculations based on current positions.
        
        Args:
            positions: Dictionary of symbol -> position size
        """
        try:
            # Reset exposures
            self.sector_exposure.clear()
            self.asset_class_exposure.clear()
            
            total_value = sum(abs(pos) for pos in positions.values())
            if total_value == 0:
                return
                
            for symbol, position_size in positions.items():
                # Get symbol info if not cached
                if symbol not in self._symbol_info:
                    self._symbol_info[symbol] = self._fetch_symbol_info(symbol)
                
                info = self._symbol_info[symbol]
                position_value = abs(position_size)
                position_weight = position_value / total_value
                
                # Update sector exposure
                sector = info.get('sector', 'Unknown')
                self.sector_exposure[sector] = self.sector_exposure.get(sector, 0) + position_weight
                
                # Update asset class exposure
                asset_class = info.get('asset_class', 'Unknown')
                self.asset_class_exposure[asset_class] = self.asset_class_exposure.get(asset_class, 0) + position_weight
                
            logger.info(f"Updated portfolio exposures: Sectors={self.sector_exposure}")
            
        except Exception as e:
            logger.error(f"Error updating exposures: {e}")
    
    def get_exposure_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get current exposure metrics."""
        return {
            'sector_exposure': self.sector_exposure,
            'asset_class_exposure': self.asset_class_exposure,
            'sector_concentration': self._calculate_concentration(self.sector_exposure),
            'asset_class_concentration': self._calculate_concentration(self.asset_class_exposure)
        }
    
    def check_exposure_limits(self, 
                            new_position: float,
                            symbol: str,
                            max_sector_exposure: float = 0.3,
                            max_asset_class_exposure: float = 0.6) -> bool:
        """
        Check if new position would exceed exposure limits.
        
        Args:
            new_position: Size of new position
            symbol: Trading symbol
            max_sector_exposure: Maximum exposure per sector
            max_asset_class_exposure: Maximum exposure per asset class
            
        Returns:
            bool: True if position is within limits
        """
        try:
            info = self._symbol_info.get(symbol) or self._fetch_symbol_info(symbol)
            sector = info.get('sector', 'Unknown')
            asset_class = info.get('asset_class', 'Unknown')
            
            # Calculate new exposures
            new_sector_exposure = self.sector_exposure.get(sector, 0) + abs(new_position)
            new_asset_exposure = self.asset_class_exposure.get(asset_class, 0) + abs(new_position)
            
            return (new_sector_exposure <= max_sector_exposure and 
                   new_asset_exposure <= max_asset_class_exposure)
                   
        except Exception as e:
            logger.error(f"Error checking exposure limits: {e}")
            return False
    
    def _fetch_symbol_info(self, symbol: str) -> Dict:
        """Fetch symbol information from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'asset_class': self._determine_asset_class(info)
            }
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return {'sector': 'Unknown', 'asset_class': 'Unknown'}
    
    def _determine_asset_class(self, info: Dict) -> str:
        """Determine asset class from symbol info."""
        quoteType = info.get('quoteType', '').lower()
        
        if quoteType == 'equity':
            return 'Stocks'
        elif quoteType == 'etf':
            return 'ETF'
        elif quoteType == 'cryptocurrency':
            return 'Crypto'
        elif quoteType in ['currency', 'fx']:
            return 'Forex'
        return 'Unknown'
    
    def _calculate_concentration(self, exposures: Dict[str, float]) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration."""
        if not exposures:
            return 0.0
        return sum(e * e for e in exposures.values())