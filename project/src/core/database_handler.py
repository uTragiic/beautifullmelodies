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
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load market data for a specific ticker with optional date range.
        
        Args:
            ticker (str): Stock ticker symbol
            start_date (str, optional): Start date for data range
            end_date (str, optional): End date for data range
            
        Returns:
            pd.DataFrame: DataFrame containing market data
            
        Raises:
            ValueError: If ticker is not available in database
        """
        try:
            # Check if symbol exists first
            if not self.is_symbol_available(ticker):
                logger.warning(f"Symbol {ticker} not found in database")
                return pd.DataFrame()  # Return empty DataFrame instead of raising exception
            
            conn = sqlite3.connect(self.db_path)
            
            if start_date and end_date:
                query = f"""
                    SELECT * FROM '{ticker}'
                    WHERE date BETWEEN '{start_date}' AND '{end_date}'
                """
            else:
                query = f"SELECT * FROM '{ticker}'"
                
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert date column to datetime and set as index
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
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