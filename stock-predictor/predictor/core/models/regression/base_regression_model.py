"""
Base regression model module.
Defines the interface for all regression models.
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from ...models.base_model import BaseModel

logger = logging.getLogger('stock_predictor')

class BaseRegressionModel(BaseModel):
    """
    Base class for all regression models.
    Extends BaseModel with regression-specific functionality.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the base regression model.
        
        Args:
            model_dir: Directory to store model files
        """
        super().__init__(model_dir)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model's performance on test data.
        
        Args:
            X_test: Test feature DataFrame
            y_test: Test target Series
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_trained:
            logger.error("Model not trained")
            return {}
            
        try:
            y_pred = self.model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def predict_interval(self, X: pd.DataFrame, confidence: float = 0.95) -> Optional[Dict[str, np.ndarray]]:
        """
        Get prediction intervals for regression estimates.
        
        Args:
            X: Feature DataFrame
            confidence: Confidence level for the interval
            
        Returns:
            Dictionary with lower and upper bounds or None if not available
        """
        if not self.is_trained:
            return None
            
        try:
            # This is a placeholder - actual implementation depends on the model
            # Some models like RandomForest can provide prediction intervals
            # Others might need to use bootstrapping or other methods
            y_pred = self.model.predict(X)
            
            # For demonstration, return simple intervals based on standard deviation
            # In practice, this should be implemented by each specific model
            if hasattr(self.model, 'predict_interval'):
                return self.model.predict_interval(X, confidence)
            else:
                logger.warning("Prediction intervals not implemented for this model")
                return None
                
        except Exception as e:
            logger.error(f"Error getting prediction intervals: {str(e)}")
            return None
