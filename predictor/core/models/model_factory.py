"""
Model factory module.
Creates model instances based on model type.
"""

import logging
import traceback
from typing import Optional

from .model_registry import ModelRegistry
from .base_model import BaseModel

logger = logging.getLogger('stock_predictor')

class ModelFactory:
    """
    Factory class for creating model instances.
    """
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> Optional[BaseModel]:
        """
        Create a model instance of the specified type.
        
        Args:
            model_type: Type of model to create
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Model instance or None if creation failed
        """
        try:
            model_class = ModelRegistry.get_model_class(model_type)
            if model_class is None:
                logger.error(f"Unknown model type: {model_type}")
                return None
                
            logger.info(f"Creating model of type: {model_type}")
            return model_class(**kwargs)
            
        except Exception as e:
            logger.error(f"Error creating model of type {model_type}: {str(e)}", exc_info=True)
            return None
