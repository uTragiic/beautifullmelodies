# Standard Library Imports
import logging
import sqlite3
from typing import Optional, List, Set

# Third-Party Imports
import pandas as pd
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
            # Check if symbol exists first
            if not self.is_symbol_available(ticker):
                logger.warning(f"Symbol {ticker} not found in database")
                return pd.DataFrame()
            
            conn = sqlite3.connect(self.db_path)
            
            # First, get the date range available in the database
            range_query = f"""
                SELECT MIN(date) as min_date, MAX(date) as max_date 
                FROM '{ticker}'
            """
            date_range = pd.read_sql_query(range_query, conn)
            db_start = pd.to_datetime(date_range['min_date'].iloc[0])
            db_end = pd.to_datetime(date_range['max_date'].iloc[0])
            
            # If dates are provided, validate and adjust them
            if start_date and end_date:
                requested_start = pd.to_datetime(start_date)
                requested_end = pd.to_datetime(end_date)
                
                # Extend start date backward to ensure minimum rows
                query = f"""
                    SELECT COUNT(*) as count 
                    FROM '{ticker}'
                    WHERE date BETWEEN '{start_date}' AND '{end_date}'
                """
                row_count = pd.read_sql_query(query, conn)['count'].iloc[0]
                
                if row_count < min_rows:
                    # Calculate how many additional days we need
                    extension_days = int((min_rows - row_count) * 1.5)  # Add 50% buffer
                    extended_start = requested_start - pd.Timedelta(days=extension_days)
                    
                    # Use the earlier of extended_start or db_start
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
                # If no dates specified, get all data
                query = f"SELECT * FROM '{ticker}' ORDER BY date"
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert date column to datetime and set as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Verify we have enough data
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
    
    def save_market_data(self, ticker: str, data: pd.DataFrame) -> None:
        """
        Save market data for a specific ticker.
        
        Args:
            ticker (str): Stock ticker symbol
            data (pd.DataFrame): Market data to save
            
        Raises:
            sqlite3.Error: If there's a database error
            Exception: For other errors during data saving
        """
        try:
            conn = sqlite3.connect(self.db_path)
            data.to_sql(ticker, conn, if_exists='replace', index=True)
            
            # Update available symbols cache
            self._available_symbols.add(ticker)
            
            conn.close()
            logger.info(f"Successfully saved market data for {ticker}")
            
        except sqlite3.Error as e:
            logger.error(f"Database error while saving data for {ticker}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving market data for {ticker}: {e}")
            raise

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