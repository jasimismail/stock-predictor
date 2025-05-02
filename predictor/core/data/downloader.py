"""
Stock data downloader module.
Responsible for fetching historical stock data from various sources.
"""

import logging
import pandas as pd
import yfinance as yf
from typing import Optional

logger = logging.getLogger('stock_predictor')

class StockDataDownloader:
    """
    Downloads historical stock data from Yahoo Finance or other sources.
    """
    
    def __init__(self, period: str = "6mo", interval: str = "1d"):
        """
        Initialize the downloader with time period and interval settings.
        
        Args:
            period: Time period to download (e.g., "6mo", "1y", "5y")
            interval: Data interval (e.g., "1d", "1h", "15m")
        """
        self.period = period
        self.interval = interval
        logger.info(f"StockDataDownloader initialized with period={period}, interval={interval}")
    
    def download(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Download historical stock data for the specified ticker.
        
        Args:
            ticker: Stock symbol to download data for
            
        Returns:
            DataFrame with historical data or None if download failed
        """
        try:
            logger.info(f"Downloading stock data for {ticker}")
            df = yf.download(
                ticker, 
                period=self.period, 
                interval=self.interval, 
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                logger.error(f"Downloaded data for {ticker} is empty")
                return None
                
            df = df.dropna()
            if len(df) < 50:
                logger.error(f"Insufficient data for {ticker}: only {len(df)} records")
                return None
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                logger.error(f"Missing required columns in {ticker} data")
                return None
                
            logger.info(f"Successfully downloaded {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            logger.error(f"Error downloading data for {ticker}: {str(e)}")
            return None
