"""
Base classification model module.
Defines the interface for all classification models.
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from ...models.base_model import BaseModel

logger = logging.getLogger('stock_predictor')

class BaseClassificationModel(BaseModel):
    """
    Base class for all classification models.
    Extends BaseModel with classification-specific functionality.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the base classification model.
        
        Args:
            model_dir: Directory to store model files
        """
        super().__init__(model_dir)
        self.classes_ = None
    
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
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
            
            # Get confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            metrics.update({
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'true_positives': tp
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Get probability estimates for each class.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of probability estimates or None if not available
        """
        if not self.is_trained or not hasattr(self.model, 'predict_proba'):
            return None
            
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error getting probability estimates: {str(e)}")
            return None
