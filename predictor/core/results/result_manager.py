"""
Result manager module.
Handles saving, retrieving, and analyzing prediction results.
"""

import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger('stock_predictor')

class PredictionResultManager:
    """
    Manages prediction results for different model types.
    """
    
    def __init__(self):
        """
        Initialize the result manager.
        """
        pass
    
    def save_prediction(
        self,
        model_type: str,
        ticker: str,
        result: Dict[str, Union[str, float]],
        prediction_model=None
    ) -> bool:
        """
        Save a prediction result.
        
        Args:
            model_type: Type of model that generated the prediction
            ticker: Stock symbol
            result: Prediction result dictionary
            prediction_model: Django model class for predictions
            
        Returns:
            True if save was successful, False otherwise
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
                    probability_up=result.get('probability_up', 0.0),
                    probability_down=result.get('probability_down', 0.0),
                    model_type=model_type,
                    predicted_price=result.get('predicted_price', None),
                    predicted_date=result.get('predicted_date', None)
                )
                logger.info(f"Saved {model_type} prediction for {ticker}")
                return True
            else:
                logger.warning(f"Not saving error prediction for {ticker}: {result.get('error', 'Unknown error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error saving prediction result: {str(e)}")
            return False
    
    def get_historical_predictions(
        self,
        model_type: str,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get historical predictions for a stock.
        
        Args:
            model_type: Type of model
            ticker: Stock symbol
            start_date: Start date for predictions
            end_date: End date for predictions
            
        Returns:
            List of historical predictions
        """
        try:
            from predictor.models import PredictionResult
            
            query = PredictionResult.objects.filter(
                stock_symbol=ticker,
                model_type=model_type
            )
            
            if start_date:
                query = query.filter(timestamp__gte=start_date)
            if end_date:
                query = query.filter(timestamp__lte=end_date)
                
            predictions = query.order_by('-timestamp')
            
            return [{
                'timestamp': p.timestamp,
                'signal': p.signal,
                'confidence': p.confidence,
                'probability_up': p.probability_up,
                'probability_down': p.probability_down,
                'predicted_price': p.predicted_price,
                'predicted_date': p.predicted_date
            } for p in predictions]
            
        except Exception as e:
            logger.error(f"Error getting historical predictions: {str(e)}")
            return []
    
    def calculate_accuracy(
        self,
        model_type: str,
        ticker: str,
        days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate prediction accuracy for a stock.
        
        Args:
            model_type: Type of model
            ticker: Stock symbol
            days: Number of days to look back
            
        Returns:
            Dictionary of accuracy metrics
        """
        try:
            from predictor.models import PredictionResult
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            predictions = PredictionResult.objects.filter(
                stock_symbol=ticker,
                model_type=model_type,
                timestamp__gte=start_date,
                timestamp__lte=end_date
            ).order_by('timestamp')
            
            if not predictions:
                return {}
                
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame([{
                'timestamp': p.timestamp,
                'signal': p.signal,
                'confidence': p.confidence,
                'predicted_price': p.predicted_price,
                'predicted_date': p.predicted_date
            } for p in predictions])
            
            # Calculate accuracy metrics
            total_predictions = len(df)
            correct_predictions = len(df[df['signal'] == 'Buy'])
            
            return {
                'total_predictions': total_predictions,
                'correct_predictions': correct_predictions,
                'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0.0,
                'average_confidence': df['confidence'].mean()
            }
            
        except Exception as e:
            logger.error(f"Error calculating accuracy: {str(e)}")
            return {}
