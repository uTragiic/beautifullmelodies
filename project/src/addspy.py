"""
Script to add SPY data to the market database.
"""

import yfinance as yf
import sqlite3
import logging
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def add_spy_to_db(db_path: str, years_of_data: int = 5):
    """
    Add SPY data to the specified database.
    
    Args:
        db_path: Path to the SQLite database
        years_of_data: Number of years of historical data to fetch
    """
    try:
        logger.info("Fetching SPY data...")
        
        # Calculate start date
        start_date = (datetime.now() - timedelta(days=365 * years_of_data)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Download SPY data
        spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
        
        # Prepare data for database
        spy_data.columns = [col.lower() for col in spy_data.columns]
        spy_data.index.name = 'date'
        
        logger.info(f"Downloaded {len(spy_data)} days of SPY data")
        
        # Connect to database and save data
        conn = sqlite3.connect(db_path)
        spy_data.to_sql('SPY', conn, if_exists='replace', index=True)
        conn.close()
        
        logger.info("Successfully added SPY data to database")
        
    except Exception as e:
        logger.error(f"Error adding SPY to database: {e}")
        raise

if __name__ == "__main__":
    # Update this path to match your database location
    db_path = "data/market_data.db"
    
    # Ensure database directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    add_spy_to_db(db_path)