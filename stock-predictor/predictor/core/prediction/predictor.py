"""
Stock predictor module.
Orchestrates the prediction process.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import pandas as pd

from ..data.downloader import StockDataDownloader
from ..features.technical_indicators import TechnicalIndicatorCalculator
from ..models.model_factory import ModelFactory
from ..data.processor import DataProcessor
from .result_manager import PredictionResultManager

logger = logging.getLogger('stock_predictor')

class StockPredictor:
    """
    Main class for stock prediction.
    Orchestrates the prediction process using various components.
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
        Initialize the stock predictor.
        
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
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize components
        self.downloader = StockDataDownloader(period=period, interval=interval)
        self.indicator_calculator = TechnicalIndicatorCalculator()
        self.data_processor = DataProcessor(forward_days=forward_days)
        self.result_manager = PredictionResultManager()
        
        logger.info(f"StockPredictor initialized with model_type={model_type}")
    
    def predict_stocks(self, tickers: List[str], save_results: bool = True) -> Dict[str, Dict]:
        """
        Predict stock movements for multiple tickers.
        
        Args:
            tickers: List of stock symbols
            save_results: Whether to save results to the database
            
        Returns:
            Dictionary of prediction results for each ticker
        """
        results = {}
        
        for ticker in tickers:
            try:
                result = self.process_ticker(ticker)
                results[ticker] = result
                logger.info(f"Prediction for {ticker}: {result}")
                
                # Save result to database if requested
                if save_results:
                    self.result_manager.save_prediction(self.model_type, ticker, result)
                    
            except Exception as e:
                logger.error(f"Error processing {ticker}: {str(e)}")
                results[ticker] = {"signal": "Error", "confidence": 0.0, "error": str(e)}
        
        return results
    
    def process_ticker(self, ticker: str) -> Dict[str, Union[str, float]]:
        """
        Process a single ticker: download data, calculate indicators, train or load model, and predict.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Prediction result
        """
        logger.info(f"Processing {ticker} with {self.model_type} model")
        
        # Download data
        df = self.downloader.download(ticker)
        if df is None:
            return {"signal": "Error", "confidence": 0.0, "error": "Failed to download data"}
        
        # Calculate indicators
        df = self.indicator_calculator.calculate_indicators(df)
        if df is None:
            return {"signal": "Error", "confidence": 0.0, "error": "Failed to calculate indicators"}
        
        # Create or load model
        model = ModelFactory.create_model(
            self.model_type, 
            model_dir=self.model_dir,
            test_size=self.test_size
        )
        
        if model is None:
            return {"signal": "Error", "confidence": 0.0, "error": f"Failed to create {self.model_type} model"}
        
        # Try to load existing model
        model_loaded = model.load(ticker)
        
        # If no model exists, train a new one
        if not model_loaded:
            logger.info(f"No existing {self.model_type} model found for {ticker}, training new model")
            
            # Generate target for training
            df_with_target = self.data_processor.generate_target(df)
            if df_with_target is None:
                return {"signal": "Error", "confidence": 0.0, "error": "Failed to generate target"}
            
            # Train model
            success = model.train(df_with_target[model.get_required_features()], df_with_target['Target'])
            if not success:
                return {"signal": "Error", "confidence": 0.0, "error": "Failed to train model"}
            
            # Save model
            model.save(ticker)
        
        # Make prediction
        latest_data = df.tail(1)
        result = model.predict(latest_data)
        # Ensure consistent signal mapping
        if 'signal' in result and result['signal'] in ['UP', 'DOWN']:
            result['signal'] = 'Buy' if result['signal'] == 'UP' else 'Sell'
        return result
