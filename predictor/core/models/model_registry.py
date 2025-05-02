"""
Model registry module.
Manages registration and retrieval of model classes.
"""

import logging
from typing import Dict, Type, Optional

from .base_model import BaseModel

logger = logging.getLogger('stock_predictor')

class ModelRegistry:
    """
    Registry for model classes.
    """
    
    # Dictionary to store model classes
    _models: Dict[str, Type[BaseModel]] = {}
    
    @classmethod
    def register_model(cls, model_type: str, model_class: Type[BaseModel]) -> None:
        """
        Register a model class.
        
        Args:
            model_type: Type identifier for the model
            model_class: Model class to register
        """
        cls._models[model_type] = model_class
        logger.info(f"Registered model type: {model_type}")
    
    @classmethod
    def get_model_class(cls, model_type: str) -> Optional[Type[BaseModel]]:
        """
        Get a model class by type.
        
        Args:
            model_type: Type identifier for the model
            
        Returns:
            Model class or None if not found
        """
        if model_type not in cls._models:
            logger.error(f"Model type not registered: {model_type}")
            return None
            
        return cls._models[model_type]
    
    @classmethod
    def get_available_models(cls) -> Dict[str, Type[BaseModel]]:
        """
        Get all registered model types.
        
        Returns:
            Dictionary of model types and classes
        """
        return cls._models.copy()
