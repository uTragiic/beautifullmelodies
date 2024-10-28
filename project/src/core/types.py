# Standard Library Imports
from enum import Enum, auto
from typing import Dict, List, Optional, Union, TypeVar, NamedTuple
from dataclasses import dataclass

class MarketRegime(Enum):
    """Enum representing different market regimes"""
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRENDING = "trending"
    RANGING = "ranging"
    NORMAL = "normal"

class TradingSignal(Enum):
    """Enum representing different trading signals"""
    BUY = 1
    SELL = -1
    HOLD = 0

class MarketCondition(Enum):
    """Enum representing different market conditions"""
    UPTREND_STRONG = "Uptrend-Strong"
    UPTREND_WEAK = "Uptrend-Weak"
    DOWNTREND_STRONG = "Downtrend-Strong"
    DOWNTREND_WEAK = "Downtrend-Weak"
    SIDEWAYS = "Sideways"

class TradeStatus(Enum):
    """Enum representing different trade statuses"""
    PENDING = auto()
    OPEN = auto()
    CLOSED = auto()
    CANCELLED = auto()

class RiskLevel(Enum):
    """Enum representing different risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

@dataclass
class TradeInfo:
    """Data class containing trade information"""
    ticker: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    direction: TradingSignal
    status: TradeStatus
    entry_time: str
    confidence_score: float
    market_condition: MarketCondition

class PriceData(NamedTuple):
    """Named tuple for price data"""
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: str

# Type aliases
Ticker = str
Price = float
Volume = float
Timestamp = str
Signal = int
ConfidenceScore = float

# Complex type definitions
MarketData = Dict[Ticker, List[PriceData]]
PositionSize = float
StopLoss = float
TakeProfit = float

# Type variables for generic functions
T = TypeVar('T')
Number = TypeVar('Number', int, float)

class IndicatorConfig:
    """Configuration settings for technical indicators"""
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9
    BB_PERIOD: int = 20
    BB_STD: int = 2
    ATR_PERIOD: int = 14
    ADX_PERIOD: int = 14
    STOCH_K_PERIOD: int = 14
    STOCH_D_PERIOD: int = 3

class RiskConfig:
    """Configuration settings for risk management"""
    MAX_POSITION_SIZE: float = 0.1  # 10% of portfolio
    MAX_RISK_PER_TRADE: float = 0.02  # 2% risk per trade
    MIN_RISK_REWARD_RATIO: float = 2.0
    MAX_CORRELATION: float = 0.7
    MAX_DRAWDOWN: float = 0.2  # 20% maximum drawdown
    POSITION_SIZING_CONFIDENCE_MULTIPLIER: float = 1.5

class BacktestConfig:
    """Configuration settings for backtesting"""
    INITIAL_CAPITAL: float = 100000.0
    COMMISSION_RATE: float = 0.001  # 0.1% commission
    SLIPPAGE: float = 0.0001  # 1 basis point slippage
    MIN_HOLDING_PERIOD: int = 1
    MAX_HOLDING_PERIOD: int = 100