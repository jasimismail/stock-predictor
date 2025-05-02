"""
Prediction service module.
Handles model predictions and result management.
"""

import logging
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
import os

from ..models.model_factory import ModelFactory
from ..data.downloader import StockDataDownloader
from ..features.technical_indicators import TechnicalIndicatorCalculator
from ..data.processor import DataProcessor
from ..results.result_manager import PredictionResultManager

logger = logging.getLogger('stock_predictor')

class PredictionService:
    """
    Service for handling stock predictions.
    """
    
    def __init__(
        self,
        model_type: str = "random_forest",
        period: str = "6mo",
        interval: str = "1d",
        forward_days: int = 3,
        test_size: float = 0.2,
        model_dir: str = "models",
        log_dir: str = "logs"
    ):
        """
        Initialize the prediction service.
        
        Args:
            model_type: Type of model to use
            period: Time period for historical data
            interval: Data interval
            forward_days: Days to look ahead for prediction
            test_size: Proportion of data to use for testing
            model_dir: Directory to store models
            log_dir: Directory to store logs
        """
        self.model_type = model_type
        self.period = period
        self.interval = interval
        self.forward_days = forward_days
        self.test_size = test_size
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        # Initialize components
        self.downloader = StockDataDownloader(period=period, interval=interval)
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.data_processor = DataProcessor(forward_days=forward_days)
        self.result_manager = PredictionResultManager()
        
        logger.info(f"PredictionService initialized with model_type={model_type}")

    def _log_error(self, message: str, exc_info: bool = True):
        """Helper method to log errors with traceback."""
        logger.error(message, exc_info=exc_info)
    
    def predict_stocks(self, tickers: List[str]) -> Dict[str, Dict[str, Union[str, float]]]:
        """
        Make predictions for a list of stock symbols.
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            Dictionary of prediction results
        """
        results = {}
        
        for ticker in tickers:
            try:
                # Download data
                logger.info(f"Downloading data for {ticker}")
                df = self.downloader.download(ticker)
                if df is None:
                    results[ticker] = {"signal": "Error", "confidence": 0.0, "error": "Failed to download data"}
                    continue
                
                # Calculate indicators
                logger.info(f"Calculating indicators for {ticker}")
                df = self.indicator_calculator.calculate_indicators(df)
                if df is None:
                    results[ticker] = {"signal": "Error", "confidence": 0.0, "error": "Failed to calculate indicators"}
                    continue
                
                # Process data and generate target
                logger.info(f"Generating target for {ticker}")
                df_with_target = self.data_processor.generate_target(df)
                if df_with_target is None:
                    results[ticker] = {"signal": "Error", "confidence": 0.0, "error": "Failed to generate target"}
                    continue
                
                # Create model
                logger.info(f"Creating {self.model_type} model for {ticker}")
                model = ModelFactory.create_model(
                    self.model_type,
                    model_dir=self.model_dir,
                    test_size=self.test_size
                )
                
                if model is None:
                    results[ticker] = {"signal": "Error", "confidence": 0.0, "error": f"Failed to create {self.model_type} model"}
                    continue
                
                # Try to load existing model
                # Ensure model directory exists
                os.makedirs(self.model_dir, exist_ok=True)
                model_path = os.path.join(self.model_dir, f"{ticker.replace('.', '_')}_model.joblib")
                logger.info(f"Attempting to load model from {model_path}")
                model_loaded = model.load(model_path)
                
                # If no model exists, train a new one
                if not model_loaded:
                    logger.info(f"No existing {self.model_type} model found for {ticker}, training new model")
                    
                    # Train model
                    logger.info(f"Training {self.model_type} model for {ticker}")
                    success = model.train(
                        df_with_target[model.get_required_features()],
                        df_with_target['Target']
                    )
                    if not success:
                        results[ticker] = {"signal": "Error", "confidence": 0.0, "error": "Failed to train model"}
                        continue
                    
                    # Save model
                    logger.info(f"Saving {self.model_type} model for {ticker} to {model_path}")
                    model.save(model_path)
                
                # Make prediction on the latest data point
                logger.info(f"Making prediction for {ticker} using {self.model_type} model")
                latest_data = df_with_target.iloc[[-1]]
                result = model.predict(latest_data)
                results[ticker] = result
                
                # Save result
                logger.info(f"Saving prediction result for {ticker}: {result}")
                self.result_manager.save_prediction(self.model_type, ticker, result)
                
            except Exception as e:
                logger.error(f"Error processing {ticker}", exc_info=True)
                results[ticker] = {"signal": "Error", "confidence": 0.0, "error": str(e)}
        
        return results
    
    def get_historical_predictions(
        self,
        ticker: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict]:
        """
        Get historical predictions for a stock.
        
        Args:
            ticker: Stock symbol
            start_date: Start date for predictions
            end_date: End date for predictions
            
        Returns:
            List of historical predictions
        """
        return self.result_manager.get_historical_predictions(
            self.model_type,
            ticker,
            start_date,
            end_date
        )
    
    def get_prediction_accuracy(
        self,
        ticker: str,
        days: int = 30
    ) -> Dict[str, float]:
        """
        Calculate prediction accuracy for a stock.
        
        Args:
            ticker: Stock symbol
            days: Number of days to look back
            
        Returns:
            Dictionary of accuracy metrics
        """
        return self.result_manager.calculate_accuracy(
            self.model_type,
            ticker,
            days
        )
