from celery import shared_task
from .models import PredictionResult
from .core.prediction.predictor import StockPredictor
from .core.prediction.result_manager import PredictionResultManager
import logging

logger = logging.getLogger('stock_predictor')

@shared_task(bind=True)
def predict_stocks_task(self, tickers, model_type="random_forest"):
    """
    Celery task to predict stock movements.
    
    Args:
        tickers: List of stock symbols
        model_type: Type of model to use
        
    Returns:
        Dictionary of prediction results
    """
    try:
        predictor = StockPredictor(model_type=model_type)
        results = predictor.predict_stocks(tickers)
        
        # Save results to database
        result_manager = PredictionResultManager()
        for ticker, result in results.items():
            result_manager.save_prediction(model_type, ticker, result)
        
        return results
    except Exception as e:
        logger.error(f"Error in predict_stocks_task: {str(e)}")
        return {"error": str(e)}
