"""
Technical indicators module.
Calculates various technical indicators for stock prediction.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, List

from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange

logger = logging.getLogger('stock_predictor')

class TechnicalIndicatorCalculator:
    """
    Calculates technical indicators for stock data.
    """
    
    def __init__(self):
        """Initialize the technical indicator calculator."""
        # Default feature list that will be calculated
        self.default_features = [
            'RSI', 'MACD', 'MACD_signal', 'BB_width', 'BB_pct', 
            'SMA_20', 'EMA_12', 'Volume_SMA', 'Volume_Ratio', 'ATR'
        ]
        logger.info("TechnicalIndicatorCalculator initialized")
    
    def get_available_indicators(self) -> List[str]:
        """
        Get list of available technical indicators.
        
        Returns:
            List of indicator names
        """
        return self.default_features
    
    def calculate_indicators(self, df: pd.DataFrame, 
                            indicators: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
        """
        Calculate technical indicators for the given DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicators to calculate (defaults to all available)
            
        Returns:
            DataFrame with added indicator columns or None if calculation failed
        """
        try:
            # Use a copy to avoid modifying the original
            df = df.copy()
            
            # If no indicators specified, use all available
            if indicators is None:
                indicators = self.default_features
                
            logger.info(f"Calculating indicators: {indicators}")
            
            # Get clean price and volume series
            close_series = pd.to_numeric(df['Close'].squeeze(), errors='coerce')
            volume_series = pd.to_numeric(df['Volume'].squeeze(), errors='coerce')
            high = pd.to_numeric(df['High'].squeeze(), errors='coerce')
            low = pd.to_numeric(df['Low'].squeeze(), errors='coerce')
            
            # Calculate RSI
            if 'RSI' in indicators:
                df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
            
            # Calculate MACD
            if any(ind in indicators for ind in ['MACD', 'MACD_signal', 'MACD_diff']):
                macd = MACD(close=close_series)
                df['MACD'] = macd.macd()
                df['MACD_signal'] = macd.macd_signal()
                df['MACD_diff'] = macd.macd_diff()
            
            # Calculate Bollinger Bands
            if any(ind in indicators for ind in ['BB_high', 'BB_low', 'BB_width', 'BB_pct']):
                bb = BollingerBands(close=close_series)
                df['BB_high'] = bb.bollinger_hband()
                df['BB_low'] = bb.bollinger_lband()
                df['BB_width'] = df['BB_high'] - df['BB_low']
                df['BB_pct'] = (close_series - df['BB_low']) / df['BB_width']
            
            # Calculate Moving Averages
            if 'SMA_20' in indicators:
                df['SMA_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()
            
            if 'EMA_12' in indicators:
                df['EMA_12'] = EMAIndicator(close=close_series, window=12).ema_indicator()
            
            # Calculate Volume indicators
            if 'Volume_SMA' in indicators:
                df['Volume_SMA'] = volume_series.rolling(window=20).mean()
            
            if 'Volume_Ratio' in indicators:
                # Avoid division by zero
                df['Volume_Ratio'] = np.where(
                    df['Volume_SMA'] > 0,
                    volume_series / df['Volume_SMA'],
                    1.0  # default to 1.0 when denominator is zero
                )
            
            # Calculate ATR (Average True Range)
            if 'ATR' in indicators:
                atr_indicator = AverageTrueRange(
                    high, 
                    low, 
                    close=close_series, 
                    window=14
                )
                df['ATR'] = atr_indicator.average_true_range()
            
            # Drop rows with NaN values
            df_clean = df.dropna()
            
            if len(df_clean) < 30:
                logger.error("Insufficient data after adding indicators")
                return None
                
            logger.info(f"Successfully calculated indicators on {len(df_clean)} records")
            return df_clean
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
            return None
