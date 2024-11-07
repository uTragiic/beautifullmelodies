# Standard Library Imports
import logging
import sqlite3
from typing import Optional, List, Set

# Third-Party Imports
import pandas as pd
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class DatabaseHandler:
    """
    Handles all database operations for market data storage and retrieval.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize DatabaseHandler with database path.
        
        Args:
            db_path (str): Path to SQLite database file
        """
        self.db_path = db_path
        self._available_symbols = self._get_available_symbols()

    def _get_available_symbols(self) -> Set[str]:
        """
        Get list of available symbols (tables) in the database.
        
        Returns:
            Set of available symbol names
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get list of all tables (symbols)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            symbols = {row[0] for row in cursor.fetchall()}
            
            conn.close()
            logger.info(f"Found {len(symbols)} symbols in database")
            return symbols
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return set()

    def is_symbol_available(self, symbol: str) -> bool:
        """
        Check if a symbol exists in the database.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            bool: True if symbol exists in database
        """
        return symbol in self._available_symbols

    def get_available_symbols(self) -> List[str]:
        """
        Get list of all available symbols.
        
        Returns:
            List of symbol names
        """
        return list(self._available_symbols)

    def load_all_market_data(self, start_date: Optional[str] = None, 
                            end_date: Optional[str] = None, min_rows: int = 200,
                            symbols: Optional[List[str]] = None) -> dict:
        """
        Load market data for specified symbols or all available symbols with a single progress bar.
        
        Args:
            start_date (str, optional): Start date for data range
            end_date (str, optional): End date for data range
            min_rows (int): Minimum number of rows required for each symbol
            symbols (List[str], optional): List of specific symbols to load. If None, loads all available symbols.
            
        Returns:
            dict: A dictionary of DataFrames keyed by ticker symbol
        """
        # Use provided symbols list or all available symbols if none provided
        symbols_to_load = symbols if symbols is not None else self.get_available_symbols()
        all_data = {}
        
        with tqdm(total=len(symbols_to_load), desc="Loading market data", leave=True) as pbar:
            for symbol in symbols_to_load:
                if self.is_symbol_available(symbol):  # Check if symbol exists in database
                    df = self.load_market_data(symbol, start_date, end_date, min_rows)
                    if not df.empty:
                        all_data[symbol] = df
                else:
                    logger.warning(f"Symbol {symbol} not found in database")
                pbar.update(1)
                logger.info(f"Progress: {pbar.n}/{pbar.total} symbols loaded")
        
        return all_data


    def load_market_data(self, ticker: str, start_date: Optional[str] = None, 
                         end_date: Optional[str] = None, min_rows: int = 200) -> pd.DataFrame:
        """
        Load market data for a specific ticker with optional date range and minimum row requirement.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str, optional): Start date for data range
            end_date (str, optional): End date for data range
            min_rows (int): Minimum number of rows required (default: 200)
                
        Returns:
            pd.DataFrame: DataFrame containing market data
                
        Raises:
            ValueError: If ticker is not available in database
        """
        try:
            if not self.is_symbol_available(ticker):
                logger.warning(f"Symbol {ticker} not found in database")
                return pd.DataFrame()
            
            conn = sqlite3.connect(self.db_path)
            
            # Get the date range available in the database
            range_query = f"""
                SELECT MIN(date) as min_date, MAX(date) as max_date 
                FROM '{ticker}'
            """
            date_range = pd.read_sql_query(range_query, conn)
            db_start = pd.to_datetime(date_range['min_date'].iloc[0])
            db_end = pd.to_datetime(date_range['max_date'].iloc[0])
            
            # Validate and adjust dates if provided
            if start_date and end_date:
                requested_start = pd.to_datetime(start_date)
                requested_end = pd.to_datetime(end_date)
                
                query = f"""
                    SELECT COUNT(*) as count 
                    FROM '{ticker}'
                    WHERE date BETWEEN '{start_date}' AND '{end_date}'
                """
                row_count = pd.read_sql_query(query, conn)['count'].iloc[0]
                
                if row_count < min_rows:
                    extension_days = int((min_rows - row_count) * 1.5)
                    extended_start = requested_start - pd.Timedelta(days=extension_days)
                    final_start = max(extended_start, db_start)
                    logger.info(f"Extending start date from {start_date} to {final_start} to ensure minimum {min_rows} rows")
                    start_date = final_start.strftime('%Y-%m-%d')
            
            # Construct and execute the main query
            if start_date and end_date:
                query = f"""
                    SELECT * FROM '{ticker}'
                    WHERE date BETWEEN '{start_date}' AND '{end_date}'
                    ORDER BY date
                """
            else:
                query = f"SELECT * FROM '{ticker}' ORDER BY date"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert date column to datetime and set as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            if len(df) < min_rows:
                logger.warning(f"Retrieved {len(df)} rows for {ticker}, which is less than minimum required {min_rows} rows")
            else:
                logger.info(f"Successfully loaded {len(df)} rows for {ticker}")
            
            return df
        
        except sqlite3.Error as e:
            logger.error(f"Database error loading data for {ticker}: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading market data for {ticker}: {e}")
            return pd.DataFrame()

    def delete_market_data(self, ticker: str) -> None:
        """
        Delete market data for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol to delete
            
        Raises:
            sqlite3.Error: If there's a database error
            Exception: For other errors during deletion
        """
        try:
            if not self.is_symbol_available(ticker):
                logger.warning(f"Symbol {ticker} not found in database")
                return
                
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(f"DROP TABLE IF EXISTS '{ticker}'")
            conn.commit()
            conn.close()
            
            # Update available symbols cache
            self._available_symbols.discard(ticker)
            
            logger.info(f"Successfully deleted market data for {ticker}")
            
        except sqlite3.Error as e:
            logger.error(f"Database error while deleting data for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error deleting market data for {ticker}: {e}")
            raise

    def refresh_symbols(self) -> None:
        """Refresh the available symbols cache."""
        self._available_symbols = self._get_available_symbols()
