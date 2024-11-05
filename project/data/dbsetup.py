import sqlite3
import logging
from pathlib import Path
import pandas as pd
import yfinance as yf
from datetime import datetime
import requests
import time
from typing import List
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseSetup:
    def __init__(self, db_path: str = "data/market_data.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def initialize_database(self):
        """Initialize the database with required tables from schema.sql"""
        try:
            # Assuming schema.sql is in the same directory as this script
            schema_file = Path(__file__).parent / 'schema.sql'
            with open(schema_file, 'r') as f:
                sql_script = f.read()
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript(sql_script)
            logger.info("Database initialized successfully using schema.sql")
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def get_nyse_symbols(self) -> List[str]:
        """
        Fetch list of NYSE symbols from NASDAQ FTP and NYSE API for redundancy.
        """
        try:
            symbols = set()  # Use set to avoid duplicates

            # Try fetching from NASDAQ FTP
            try:
                url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
                df = pd.read_csv(url, delimiter='|')
                # Remove the last line which may be a summary
                df = df[:-1]
                # Filter symbols where Exchange is 'N' (NYSE)
                nyse_symbols = df[df['Exchange'] == 'N']['ACT Symbol'].tolist()
                symbols.update(nyse_symbols)
                logger.info(f"Retrieved {len(nyse_symbols)} symbols from NASDAQ FTP")
            except Exception as e:
                logger.warning(f"Failed to fetch from NASDAQ FTP: {e}")

            # Backup: Try NYSE website listing
            if not symbols:
                try:
                    url = "https://www.nyse.com/api/quotes/filter"
                    payload = {
                        "instrumentType": "EQUITY",
                        "pageNumber": 1,
                        "sortColumn": "NORMALIZED_TICKER",
                        "sortOrder": "ASC",
                        "maxResults": 10000,
                        "filterToken": ""
                    }
                    response = requests.post(url, json=payload, headers=self.headers)
                    nyse_data = response.json()
                    symbols.update([item['symbolTicker'] for item in nyse_data])
                    logger.info(f"Retrieved {len(symbols)} symbols from NYSE API")
                except Exception as e:
                    logger.warning(f"Failed to fetch from NYSE API: {e}")

            # Filter symbols
            valid_symbols = []
            for symbol in symbols:
                # Basic validation
                if (isinstance(symbol, str) and 
                    len(symbol) <= 5 and  # Most NYSE symbols are 1-5 characters
                    symbol.isalpha() and   # Only alphabetic characters
                    not any(c in symbol for c in ['^', '.', '/']) and  # Exclude symbols with special characters
                    not any(c in symbol for c in ['W', 'R', 'P', 'Q'])):  # Exclude warrants, rights, preferred shares
                    valid_symbols.append(symbol)

            logger.info(f"Total valid NYSE symbols: {len(valid_symbols)}")
            
            # Save symbols to file for backup
            self._save_symbols(valid_symbols)
            
            return valid_symbols

        except Exception as e:
            logger.error(f"Error fetching NYSE symbols: {e}")
            # Try to load from backup file
            return self._load_backup_symbols()

    def _save_symbols(self, symbols: List[str]) -> None:
        """Save symbols to backup file"""
        try:
            symbols_file = self.db_path.parent / 'nyse_symbols.txt'
            with open(symbols_file, 'w') as f:
                f.write('\n'.join(symbols))
        except Exception as e:
            logger.error(f"Error saving symbols to file: {e}")

    def _load_backup_symbols(self) -> List[str]:
        """Load symbols from backup file"""
        try:
            symbols_file = self.db_path.parent / 'nyse_symbols.txt'
            with open(symbols_file, 'r') as f:
                symbols = f.read().splitlines()
            logger.info(f"Loaded {len(symbols)} symbols from backup file")
            return symbols
        except Exception as e:
            logger.error(f"Error loading backup symbols: {e}")
            return ['SPY', 'DIA', 'IWM', 'GE', 'F', 'BAC', 'WFC', 'XOM', 'CVX', 'PFE']

    def validate_symbols(self, symbols: List[str]) -> List[str]:
        valid_symbols = []
        total = len(symbols)
        
        for i, symbol in enumerate(symbols, 1):
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period='1mo')
                if df.empty or 'Close' not in df.columns:
                    logger.warning(f"No price data for {symbol}")
                    continue

                # Ensure the column names are lowercase for consistency
                df.columns = [col.lower() for col in df.columns]

                latest = df.iloc[-1]
                if (latest['volume'] > 50000 and latest['close'] > 0.1):
                    valid_symbols.append(symbol)
                else:
                    logger.info(f"Symbol {symbol} does not meet volume or price criteria.")
                
                if i % 100 == 0:
                    logger.info(f"Validated {i}/{total} symbols")
                    
                time.sleep(0.1)
            except Exception as e:
                logger.warning(f"Error validating {symbol}: {e}")
                continue

        logger.info(f"Found {len(valid_symbols)} valid symbols out of {total}")
        return valid_symbols
    

    def create_symbol_table(self, symbol: str):
        """Create a new table for a trading symbol using market_data_template"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Check if table already exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (symbol,))
                if cursor.fetchone():
                    logger.info(f"Table for {symbol} already exists")
                    return
                # Get the SQL for the market_data_template table
                cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='market_data_template'")
                result = cursor.fetchone()
                if result:
                    create_table_sql = result[0]
                    # Replace the table name with the symbol (without quotes)
                    create_table_sql = create_table_sql.replace('market_data_template', symbol)
                    # Ensure 'IF NOT EXISTS' is included
                    if 'IF NOT EXISTS' not in create_table_sql.upper():
                        create_table_sql = create_table_sql.replace('CREATE TABLE', 'CREATE TABLE IF NOT EXISTS')
                    # Log the SQL statement for debugging
                    logger.debug(f"Executing SQL for {symbol}: {create_table_sql}")
                    # Execute the create table statement
                    cursor.execute(create_table_sql)
                    logger.info(f"Created table for {symbol} using market_data_template")
                else:
                    logger.error("market_data_template table does not exist.")
        except sqlite3.OperationalError as e:
            if 'already exists' in str(e):
                logger.info(f"Table for {symbol} already exists")
            else:
                logger.error(f"Error creating table for {symbol}: {e}")
                raise
        except Exception as e:
            logger.error(f"Error creating table for {symbol}: {e}")
            raise


    def load_historical_data(self, symbol: str, start_date: str, end_date: str = None):
        """Load historical market data for a symbol"""
        try:
            # Create table for symbol if it doesn't exist
            self.create_symbol_table(symbol)
            
            # Download data from yfinance
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Using start and end dates to fetch historical data
            df = yf.download(symbol, start=start_date, end=end_date)
            
            if df.empty:
                logger.warning(f"No data available for {symbol}")
                return False
                
            df.index.name = 'date'
            df.columns = [col.lower().replace(' ', '_') for col in df.columns]
            
            # Insert data into database
            with sqlite3.connect(self.db_path) as conn:
                df.to_sql(symbol, conn, if_exists='replace', index=True)
                
            logger.info(f"Loaded historical data for {symbol}")
            return True
                
        except Exception as e:
            logger.error(f"Error loading historical data for {symbol}: {e}")
            return False

    def verify_data_integrity(self, symbol: str) -> bool:
        """Verify data integrity for a symbol"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if table exists
                cursor = conn.cursor()
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{symbol}'")
                if not cursor.fetchone():
                    logger.warning(f"No table found for {symbol}")
                    return False
                        
                # Check for missing values
                df = pd.read_sql(f"SELECT * FROM '{symbol}'", conn)
                if df.isnull().any().any():
                    logger.warning(f"Found missing values in {symbol} data")
                    return False
                        
                # Check for price integrity
                price_issues = (
                    (df['high'] < df['low']).any() or
                    (df['close'] > df['high']).any() or
                    (df['close'] < df['low']).any() or
                    (df['open'] > df['high']).any() or
                    (df['open'] < df['low']).any()
                )
                    
                if price_issues:
                    logger.warning(f"Found price integrity issues in {symbol} data")
                    return False
                        
                # Check for chronological ordering
                if not pd.to_datetime(df['date']).is_monotonic_increasing:
                    logger.warning(f"Data not in chronological order for {symbol}")
                    return False
                        
                return True
                    
        except Exception as e:
            logger.error(f"Error verifying data integrity for {symbol}: {e}")
            return False

    def load_initial_market_conditions(self):
        """Load initial market conditions data"""
        try:
            conditions = [
                (1, "Uptrend-Strong-High_Vol", "Uptrend", "High", "High", "Strong", 0.65, 0.012, 1.8),
                (2, "Uptrend-Strong-Low_Vol", "Uptrend", "Low", "Normal", "Strong", 0.70, 0.008, 2.1),
                (3, "Downtrend-Strong-High_Vol", "Downtrend", "High", "High", "Weak", 0.45, -0.010, -1.2),
                (4, "Ranging-Normal-Normal_Vol", "Ranging", "Normal", "Normal", "Neutral", 0.52, 0.002, 0.5)
            ]
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executemany("""
                    INSERT OR IGNORE INTO market_conditions 
                    (condition_id, description, trend_direction, volatility_level, 
                     volume_level, momentum_state, win_rate, avg_return, sharpe_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, conditions)
                
            logger.info("Loaded initial market conditions")
            
        except Exception as e:
            logger.error(f"Error loading market conditions: {e}")
            raise

def setup_database_with_nyse_data(start_date: str = '2010-01-01'):
    """
    Complete database setup process including fetching NYSE symbols and historical data
    """
    try:
        # Create database setup instance
        db_setup = DatabaseSetup()
        
        # Initialize database using schema.sql
        db_setup.initialize_database()
        
        # Load market conditions if not already in schema.sql
        db_setup.load_initial_market_conditions()
        
        # Get NYSE symbols
        all_symbols = db_setup.get_nyse_symbols()
        logger.info(f"Retrieved {len(all_symbols)} total NYSE symbols")
        
        # Validate symbols for tradability
        valid_symbols = db_setup.validate_symbols(all_symbols)
        logger.info(f"Found {len(valid_symbols)} tradable symbols")
        
        # Load historical data for each valid symbol
        successful_loads = 0
        failed_loads = 0
        
        for symbol in valid_symbols:
            try:
                # Create a table for the symbol
                db_setup.create_symbol_table(symbol)
                # Load historical data
                if db_setup.load_historical_data(symbol, start_date):
                    if db_setup.verify_data_integrity(symbol):
                        successful_loads += 1
                    else:
                        failed_loads += 1
                else:
                    failed_loads += 1
                        
                # Log progress every 10 symbols
                if (successful_loads + failed_loads) % 10 == 0:
                    logger.info(f"Progress: {successful_loads} successful, {failed_loads} failed")
                        
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                failed_loads += 1
                continue
        
        logger.info(f"Database setup completed. Successfully loaded {successful_loads} symbols, {failed_loads} failed.")
        
    except Exception as e:
        logger.error(f"Error in database setup: {e}")
        raise

if __name__ == "__main__":
    try:
        # Set start date for historical data
        start_date = '2010-01-01'  # Adjust as needed
        
        # Run complete setup
        setup_database_with_nyse_data(start_date)
        
    except Exception as e:
        logger.error(f"Fatal error in database setup: {e}")
        raise
