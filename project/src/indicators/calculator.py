# Standard Library Imports
import logging
from typing import Dict, Optional

# Third-Party Imports
import numpy as np
import pandas as pd
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, MACD, SMAIndicator
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
from ta.volatility import AverageTrueRange, BollingerBands

# Local Imports
from ..core.types import IndicatorConfig

# Set up logging
logger = logging.getLogger(__name__)

class IndicatorCalculator:
    """
    Calculates and manages technical indicators for market analysis.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        """
        Initialize IndicatorCalculator with optional configuration.
        
        Args:
            config: Optional configuration settings for indicators
        """
        self.config = config or IndicatorConfig()
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required technical indicators."""
        try:
            df = data.copy()
            
            # Calculate RSI
            df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
            
            # Calculate MACD
            macd = MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            
            # Calculate ADX
            df['ADX'] = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14).adx()
            
            # Calculate ATR
            df['ATR'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
            
            # Calculate Volume Ratio
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
            
            # Calculate Momentum
            df['Momentum'] = df['close'].pct_change(periods=20)
            
            # Calculate Stochastic
            stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'], window=14, smooth_window=3)
            df['Stoch_K'] = stoch.stoch()
            df['Stoch_D'] = stoch.stoch_signal()
            
            # Calculate OBV
            df['OBV'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
            
            # Calculate Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            df['BB_upper'] = bb.bollinger_hband()
            df['BB_lower'] = bb.bollinger_lband()
            df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']
            
            # Calculate VWAP
            df['VWAP'] = VolumeWeightedAveragePrice(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume']
            ).volume_weighted_average_price()
            
            # Forward fill any NaN values
            df = df.ffill()
            
            # Verify all indicators were calculated
            required_indicators = [
                'RSI', 'MACD_diff', 'ADX', 'ATR', 'Volume_Ratio', 
                'Momentum', 'Stoch_K', 'OBV', 'BB_width', 'VWAP'
            ]
            
            missing_indicators = [ind for ind in required_indicators if ind not in df.columns]
            if missing_indicators:
                raise ValueError(f"Failed to calculate indicators: {missing_indicators}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            raise

    def _validate_data(self, data: pd.DataFrame) -> None:
        """Validate that the DataFrame has required columns."""
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        missing_columns = required_columns - set(data.columns.str.lower())
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

    def _calculate_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate various moving averages."""
        df['SMA_50'] = SMAIndicator(
            close=df['close'], 
            window=50
        ).sma_indicator()
        
        df['SMA_200'] = SMAIndicator(
            close=df['close'], 
            window=200
        ).sma_indicator()
        
        return df

    def _calculate_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate MACD indicator and its components."""
        macd = MACD(
            close=df['close'],
            window_slow=self.config.MACD_SLOW,
            window_fast=self.config.MACD_FAST,
            window_sign=self.config.MACD_SIGNAL
        )
        
        df['MACD'] = macd.macd()
        df['MACD_signal'] = macd.macd_signal()
        df['MACD_diff'] = macd.macd_diff()
        
        return df

    def _calculate_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI indicator."""
        df['RSI'] = RSIIndicator(
            close=df['close'],
            window=self.config.RSI_PERIOD
        ).rsi()
        
        return df

    def _calculate_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        stoch = StochasticOscillator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.config.STOCH_K_PERIOD,
            smooth_window=self.config.STOCH_D_PERIOD
        )
        
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()
        
        return df

    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range."""
        df['ATR'] = AverageTrueRange(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.config.ATR_PERIOD
        ).average_true_range()
        
        return df

    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume."""
        df['OBV'] = OnBalanceVolumeIndicator(
            close=df['close'],
            volume=df['volume']
        ).on_balance_volume()
        
        return df

    def _calculate_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average Directional Index."""
        df['ADX'] = ADXIndicator(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            window=self.config.ADX_PERIOD
        ).adx()
        
        return df

    def _calculate_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        bb = BollingerBands(
            close=df['close'],
            window=self.config.BB_PERIOD,
            window_dev=self.config.BB_STD
        )
        
        df['BB_upper'] = bb.bollinger_hband()
        df['BB_lower'] = bb.bollinger_lband()
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['close']
        
        return df

    def _calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price."""
        df['VWAP'] = VolumeWeightedAveragePrice(
            high=df['high'],
            low=df['low'],
            close=df['close'],
            volume=df['volume']
        ).volume_weighted_average_price()
        
        return df

    def _calculate_normalized_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized versions of indicators."""
        # Normalize RSI
        df['RSI_Z'] = (
            df['RSI'] - df['RSI'].rolling(window=50).mean()
        ) / df['RSI'].rolling(window=50).std()
        
        # Normalize MACD
        df['MACD_Z'] = (
            df['MACD_diff'] - df['MACD_diff'].rolling(window=50).mean()
        ) / df['MACD_diff'].rolling(window=50).std()
        
        # Calculate additional derived indicators
        df['Momentum'] = df['close'].pct_change(periods=20)
        df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
        
        return df

    def get_indicator_values(self, data: pd.DataFrame) -> Dict[str, tuple]:
        """
        Get the latest indicator values along with their means and standard deviations.
        
        Args:
            data: DataFrame with calculated indicators
            
        Returns:
            Dictionary of indicator values with their statistics
        """
        indicators = {
            'RSI': (data['RSI'].iloc[-1], data['RSI'].mean(), data['RSI'].std()),
            'MACD': (data['MACD'].iloc[-1], data['MACD'].mean(), data['MACD'].std()),
            'ATR': (data['ATR'].iloc[-1], data['ATR'].mean(), data['ATR'].std()),
            'Stochastic': (data['Stoch_K'].iloc[-1], data['Stoch_K'].mean(), data['Stoch_K'].std()),
            'BB_width': (data['BB_width'].iloc[-1], data['BB_width'].mean(), data['BB_width'].std()),
        }
        return indicators