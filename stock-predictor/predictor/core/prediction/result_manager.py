"""
Result manager module.
Handles saving, retrieving, and managing prediction results.
"""

import logging
from typing import Dict, List, Optional, Union
from django.db.models import QuerySet

logger = logging.getLogger('stock_predictor')

class PredictionResultManager:
    """
    Manages prediction results for different model types.
    """
    
    @staticmethod
    def save_prediction(model_type: str, ticker: str, result: Dict[str, Union[str, float]], 
                       prediction_model=None):
        """
        Save a prediction result to the database.
        
        Args:
            model_type: Type of model that generated the prediction
            ticker: Stock symbol
            result: Prediction result dictionary
            prediction_model: Django model class for predictions
        """
        try:
            if prediction_model is None:
                # Import here to avoid circular imports
                from predictor.models import PredictionResult
                prediction_model = PredictionResult
                
            if result['signal'] != 'Error':
                prediction_model.objects.create(
                    stock_symbol=ticker,
                    signal=result['signal'],
                    confidence=result['confidence'],
                    probability_up=result['probability_up'],
                    probability_down=result['probability_down'],
                    model_type=model_type
                )
                logger.info(f"Saved {model_type} prediction for {ticker}")
            else:
                logger.warning(f"Not saving error prediction for {ticker}: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error saving prediction result: {str(e)}")
    
    @staticmethod
    def get_predictions(model_type: Optional[str] = None, limit: int = 10, 
                       prediction_model=None) -> QuerySet:
        """
        Get prediction results, optionally filtered by model type.
        
        Args:
            model_type: Type of model to filter by (None for all)
            limit: Maximum number of results to return
            prediction_model: Django model class for predictions
            
        Returns:
            QuerySet of prediction results
        """
        try:
            if prediction_model is None:
                # Import here to avoid circular imports
                from predictor.models import PredictionResult
                prediction_model = PredictionResult
                
            queryset = prediction_model.objects.all()
            
            if model_type:
                queryset = queryset.filter(model_type=model_type)
                
            return queryset.order_by('-timestamp')[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving predictions: {str(e)}")
            return []
    
    @staticmethod
    def get_predictions_for_ticker(ticker: str, model_type: Optional[str] = None, 
                                 limit: int = 10, prediction_model=None) -> QuerySet:
        """
        Get prediction results for a specific ticker, optionally filtered by model type.
        
        Args:
            ticker: Stock symbol to filter by
            model_type: Type of model to filter by (None for all)
            limit: Maximum number of results to return
            prediction_model: Django model class for predictions
            
        Returns:
            QuerySet of prediction results
        """
        try:
            if prediction_model is None:
                # Import here to avoid circular imports
                from predictor.models import PredictionResult
                prediction_model = PredictionResult
                
            queryset = prediction_model.objects.filter(stock_symbol=ticker)
            
            if model_type:
                queryset = queryset.filter(model_type=model_type)
                
            return queryset.order_by('-timestamp')[:limit]
            
        except Exception as e:
            logger.error(f"Error retrieving predictions for ticker {ticker}: {str(e)}")
            return []
