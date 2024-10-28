-- Create market data table template
CREATE TABLE IF NOT EXISTS market_data_template (
    date TIMESTAMP PRIMARY KEY,
    open REAL NOT NULL,
    high REAL NOT NULL,
    low REAL NOT NULL,
    close REAL NOT NULL,
    volume INTEGER NOT NULL,
    adjusted_close REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create market conditions table
CREATE TABLE IF NOT EXISTS market_conditions (
    condition_id INTEGER PRIMARY KEY,
    description TEXT NOT NULL,
    trend_direction TEXT NOT NULL,
    volatility_level TEXT NOT NULL,
    volume_level TEXT NOT NULL,
    momentum_state TEXT NOT NULL,
    win_rate REAL,
    avg_return REAL,
    sharpe_ratio REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create backtest results table
CREATE TABLE IF NOT EXISTS backtest_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    market_condition_id INTEGER NOT NULL,
    avg_return REAL NOT NULL,
    win_rate REAL NOT NULL,
    sharpe_ratio REAL NOT NULL,
    max_drawdown REAL NOT NULL,
    volatility REAL NOT NULL,
    trade_count INTEGER NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (market_condition_id) REFERENCES market_conditions(condition_id)
);

-- Create trading statistics table
CREATE TABLE IF NOT EXISTS trading_statistics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    market_condition_id INTEGER NOT NULL,
    signal_count INTEGER NOT NULL,
    correct_signals INTEGER NOT NULL,
    false_signals INTEGER NOT NULL,
    avg_profit_per_trade REAL,
    avg_loss_per_trade REAL,
    avg_holding_period REAL,
    period_start TIMESTAMP NOT NULL,
    period_end TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (market_condition_id) REFERENCES market_conditions(condition_id)
);