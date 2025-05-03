"""
Data processing module.
Handles data cleaning, preprocessing, and target generation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional, Tuple

logger = logging.getLogger('stock_predictor')

class DataProcessor:
    """
    Processes stock data for model training and prediction.
    """
    
    def __init__(self, forward_days: int = 3):
        """
        Initialize the data processor.
        
        Args:
            forward_days: Number of days to look ahead for target generation
        """
        self.forward_days = forward_days
        logger.info(f"DataProcessor initialized with forward_days={forward_days}")
    
    def get_clean_close_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """
        Extract and clean the Close price series from a DataFrame.
        
        Args:
            df: DataFrame containing stock data
            
        Returns:
            Clean Close price series or None if processing failed
        """
        try:
            if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
                logger.error("Invalid DataFrame or missing Close column")
                return None
            
            close_series = pd.to_numeric(df['Close'].squeeze(), errors='coerce')
            close_series = close_series.dropna()
            
            if len(close_series) < 30:
                logger.error("Not enough valid Close prices")
                return None
                
            return close_series
            
        except Exception as e:
            logger.error(f"Error processing Close prices: {str(e)}")
            return None
    
    def generate_target(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate target variable based on future price movement.
        
        Args:
            df: DataFrame with stock data
            
        Returns:
            DataFrame with added target column or None if generation failed
        """
        try:
            logger.info(f"Generating target for {self.forward_days}-day future return")
            
            future_price = df['Close'].shift(-self.forward_days)
            current_price = df['Close']
            pct_change = (future_price - current_price) / current_price
            
            df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
            df['Target'] = np.where(pct_change > 0, 1, 0)
            df['Target_magnitude'] = pct_change.abs()
            
            df_clean = df.dropna()
            
            up_count = sum(df_clean['Target'] == 1)
            down_count = sum(df_clean['Target'] == 0)
            total = len(df_clean)
            
            if total > 0:
                logger.info(f"Target generated: {up_count}/{total} up ({up_count/total:.1%}), "
                          f"{down_count}/{total} down ({down_count/total:.1%})")
            else:
                logger.warning("No data points remain after target generation")
            
            return df_clean
            
        except Exception as e:
            logger.error(f"Error generating target: {str(e)}")
            return None
    
    def process(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """
        Process the input DataFrame for model training or prediction.
        
        Args:
            df: DataFrame with stock data and technical indicators
            
        Returns:
            Tuple of (features DataFrame, target Series) or (None, None) if processing failed
        """
        try:
            logger.info("Processing data for model training/prediction")
            
            # Generate target for training data
            df_with_target = self.generate_target(df)
            if df_with_target is None:
                logger.error("Failed to generate target")
                return None, None
            
            # Extract features and target
            features = df_with_target.drop(['Target', 'Target_magnitude'], axis=1)
            target = df_with_target['Target']
            
            # Ensure all data is numeric
            features = features.apply(pd.to_numeric, errors='coerce')
            target = pd.to_numeric(target, errors='coerce')
            
            # Drop any remaining NaN values
            features = features.dropna()
            target = target.dropna()
            
            if len(features) < 30 or len(target) < 30:
                logger.error("Insufficient data after processing")
                return None, None
            
            logger.info(f"Successfully processed {len(features)} records")
            return features, target
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return None, None
