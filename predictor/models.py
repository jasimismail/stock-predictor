"""
Django models for the stock predictor application.
"""

from django.db import models
from django.utils import timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class PredictionResult(models.Model):
    """
    Model for storing prediction results.
    """
    
    # Model types
    MODEL_TYPES = [
        ('random_forest', 'Random Forest'),
        ('lightgbm', 'LightGBM'),
        ('xgboost', 'XGBoost'),
        ('catboost', 'CatBoost')
    ]
    
    # Signal types
    SIGNAL_TYPES = [
        ('Buy', 'Buy'),
        ('Sell', 'Sell'),
        ('Hold', 'Hold'),
        ('Error', 'Error')
    ]
    
    stock_symbol = models.CharField(max_length=20)
    signal = models.CharField(max_length=10, choices=SIGNAL_TYPES)
    confidence = models.FloatField()
    probability_up = models.FloatField(default=0.0)
    probability_down = models.FloatField(default=0.0)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    predicted_price = models.FloatField(null=True, blank=True)
    predicted_date = models.DateTimeField(null=True, blank=True)
    timestamp = models.DateTimeField(default=timezone.now)
    
    class Meta:
        indexes = [
            models.Index(fields=['stock_symbol']),
            models.Index(fields=['model_type']),
            models.Index(fields=['timestamp']),
        ]
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.stock_symbol} - {self.signal} ({self.model_type})"
    
    @property
    def is_accurate(self) -> Optional[bool]:
        """
        Check if the prediction was accurate by comparing the predicted signal
        with the actual price movement.
        
        Returns:
            True if prediction was accurate, False if inaccurate, None if cannot be determined
        """
        try:
            if not self.predicted_price or not self.predicted_date:
                return None
                
            # Get the actual price at the predicted date
            from .core.data.downloader import StockDataDownloader
            downloader = StockDataDownloader(period='1d', interval='1d')
            df = downloader.download(self.stock_symbol)
            
            if df is None or df.empty:
                return None
                
            # Get the actual price at the predicted date
            actual_price = df.loc[df.index.date == self.predicted_date.date(), 'Close'].iloc[0]
            
            # Calculate price movement
            price_movement = actual_price - self.predicted_price
            
            # Determine if prediction was accurate
            if self.signal == 'Buy':
                return price_movement > 0
            elif self.signal == 'Sell':
                return price_movement < 0
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error checking prediction accuracy: {str(e)}")
            return None
    
    @property
    def days_until_prediction(self) -> Optional[int]:
        """
        Get the number of days until the predicted date.
        
        Returns:
            Number of days or None if no predicted date
        """
        if not self.predicted_date:
            return None
        return (self.predicted_date - timezone.now()).days
