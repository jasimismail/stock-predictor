"""
Base model module.
Defines the interface for all prediction models.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger('stock_predictor')

class BaseModel(ABC):
    """
    Abstract base class for all prediction models.
    Defines the common interface that all models must implement.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the base model.
        
        Args:
            model_dir: Directory to store model files
        """
        self.model_dir = model_dir
        self.model = None
        self.features: List[str] = []
        self.is_trained = False
    
    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Train the model on the given data.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            True if training was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary containing prediction results
        """
        pass
    
    @abstractmethod
    def save(self, model_path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            True if save was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def load(self, model_path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to load the model from
            
        Returns:
            True if load was successful, False otherwise
        """
        pass
    
    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """
        Get feature importance scores from the model.
        
        Returns:
            DataFrame with feature importance scores or None if not available
        """
        if not self.is_trained or not hasattr(self.model, 'feature_importances_'):
            return None
            
        return pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
    
    def validate_features(self, X: pd.DataFrame) -> bool:
        """
        Validate that all required features are present in the data.
        
        Args:
            X: Feature DataFrame to validate
            
        Returns:
            True if all features are present, False otherwise
        """
        missing_cols = [col for col in self.features if col not in X.columns]
        if missing_cols:
            logger.error(f"Missing required features: {missing_cols}")
            return False
        return True
