import logging
import os
import traceback
import sys
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Union

import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class StockPredictor:
    """
    Core stock prediction logic using technical indicators and Random Forest classification.
    """
    
    def __init__(
        self, 
        tickers: Union[str, List[str]] = "RELIANCE.NS",
        period: str = "6mo",
        interval: str = "1d",
        forward_days: int = 3,
        test_size: float = 0.2,
        model_dir: str = "models",
        log_dir: str = "logs"
    ):
        self.tickers = [tickers] if isinstance(tickers, str) else tickers
        self.period = period
        self.interval = interval
        self.forward_days = forward_days
        self.test_size = test_size
        self.model_dir = model_dir
        self.log_dir = log_dir
        
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        self._setup_logging()
        
        self.data: Dict[str, pd.DataFrame] = {}
        # Updated models type hint to include feature lists
        self.models: Dict[str, Tuple[RandomForestClassifier, List[str]]] = {}
        # Updated features list to include Volume and ATR
        self.features = [
            'RSI', 'MACD', 'MACD_signal', 'BB_width', 'BB_pct', 
            'SMA_20', 'EMA_12', 'Volume_SMA', 'Volume_Ratio', 'ATR'
        ]
        self.logger.info(f"StockPredictor initialized for {self.tickers}")
    
    def _setup_logging(self) -> None:
        """Configure logging to both file and console with formatted output."""
        log_file = os.path.join(self.log_dir, "predictions.log")
        
        self.logger = logging.getLogger('stock_predictor')
        self.logger.setLevel(logging.INFO)
        
        if self.logger.handlers:
            for handler in self.logger.handlers:
                self.logger.removeHandler(handler)
        
        # Enhanced formatter with line numbers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        self.logger.info(f"Logging configured. Log file: {log_file}")

    def download_stock_data(self, ticker: str) -> Optional[pd.DataFrame]:
        """Download historical stock data for the specified ticker."""
        try:
            self.logger.info(f"Downloading stock data for {ticker}")
            df = yf.download(
                ticker, 
                period=self.period, 
                interval=self.interval, 
                progress=False,
                auto_adjust=True
            )
            
            if df.empty:
                self.logger.error(f"Downloaded data for {ticker} is empty")
                return None
                
            df = df.dropna()
            if len(df) < 50:
                self.logger.error(f"Insufficient data for {ticker}: only {len(df)} records")
                return None
            
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(col in df.columns for col in required_cols):
                self.logger.error(f"Missing required columns in {ticker} data")
                return None
                
            self.logger.info(f"Successfully downloaded {len(df)} records for {ticker}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error downloading data for {ticker}: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def _get_clean_close_series(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Ensure Close prices are properly formatted as a 1D float Series."""
        try:
            if not isinstance(df, pd.DataFrame) or 'Close' not in df.columns:
                self.logger.error("Invalid DataFrame or missing Close column")
                return None
            
            # Updated line to use squeeze as suggested
            close_series = pd.to_numeric(df['Close'].squeeze(), errors='coerce')
            close_series = close_series.dropna()
            
            if len(close_series) < 30:
                self.logger.error("Not enough valid Close prices")
                return None
                
            return close_series
            
        except Exception as e:
            current_frame = sys._getframe()
            line_number = current_frame.f_lineno
            self.logger.error(f"Error processing Close prices: {str(e)} at line {line_number}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def add_technical_indicators(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculate and add technical indicators to the dataframe."""
        try:
            close_series = self._get_clean_close_series(df)
            if close_series is None:
                return None
                
            df['RSI'] = RSIIndicator(close=close_series, window=14).rsi()
            
            macd = MACD(close=close_series)
            df['MACD'] = macd.macd()
            df['MACD_signal'] = macd.macd_signal()
            df['MACD_diff'] = macd.macd_diff()
            
            bb = BollingerBands(close=close_series)
            df['BB_high'] = bb.bollinger_hband()
            df['BB_low'] = bb.bollinger_lband()
            df['BB_width'] = df['BB_high'] - df['BB_low']
            df['BB_pct'] = (close_series - df['BB_low']) / df['BB_width']
            
            df['SMA_20'] = SMAIndicator(close=close_series, window=20).sma_indicator()
            df['EMA_12'] = EMAIndicator(close=close_series, window=12).ema_indicator()
            
            # Add Volume indicators (safely)
            # Convert Volume to numeric and handle any issues
            volume_series = pd.to_numeric(df['Volume'].squeeze(), errors='coerce')

            high = pd.to_numeric(df['High'].squeeze(), errors='coerce')
            low = pd.to_numeric(df['Low'].squeeze(), errors='coerce')
            
            # Calculate 20-day volume SMA 
            df['Volume_SMA'] = volume_series.rolling(window=20).mean()
            
            # Volume ratio (current volume compared to average)
            # Avoid division by zero
            df['Volume_Ratio'] = np.where(
                df['Volume_SMA'] > 0,
                volume_series / df['Volume_SMA'],
                1.0  # default to 1.0 when denominator is zero
            )
            
            # Add ATR (Average True Range) for volatility
            atr_indicator = AverageTrueRange(
                high, 
                low, 
                close=close_series, 
                window=14
            )
            df['ATR'] = atr_indicator.average_true_range()
            
            df = df.dropna()
            if len(df) < 30:
                self.logger.error("Insufficient data after adding indicators")
                return None
                
            self.logger.info(f"Successfully calculated indicators on {len(df)} records")
            return df
            
        except Exception as e:
            current_frame = sys._getframe()
            line_number = current_frame.f_lineno
            self.logger.error(f"Error calculating indicators: {str(e)} at line {line_number}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def generate_target(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Generate target variable based on future price movement."""
        try:
            self.logger.info(f"Generating target for {self.forward_days}-day future return")
            
            future_price = df['Close'].shift(-self.forward_days)
            current_price = df['Close']
            pct_change = (future_price - current_price) / current_price
            
            df['Target'] = np.where(pct_change > 0, 1, 0)
            df['Target_magnitude'] = pct_change.abs()
            
            df_clean = df.dropna()
            
            up_count = sum(df_clean['Target'] == 1)
            down_count = sum(df_clean['Target'] == 0)
            total = len(df_clean)
            
            self.logger.info(f"Target generated: {up_count}/{total} up ({up_count/total:.1%}), "
                          f"{down_count}/{total} down ({down_count/total:.1%})")
            
            return df_clean
            
        except Exception as e:
            current_frame = sys._getframe()
            line_number = current_frame.f_lineno
            self.logger.error(f"Error generating target: {str(e)} at line {line_number}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def train_model(self, df: pd.DataFrame, ticker: str) -> bool:
        """Train a Random Forest model to predict price movements."""
        try:
            self.logger.info(f"Training model for {ticker}")
            
            missing_cols = [col for col in self.features if col not in df.columns]
            if missing_cols:
                self.logger.error(f"Missing required features: {missing_cols}")
                return False
            
            if 'Target' not in df.columns:
                self.logger.error("Missing target column")
                return False
            
            X = df[self.features]
            y = df['Target']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )
            
            model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            self.logger.info(f"Fitting model on {len(X_train)} training samples")
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            self.logger.info(f"Model performance for {ticker}:")
            self.logger.info(f"  Accuracy: {accuracy:.4f}")
            self.logger.info(f"  Precision: {precision:.4f}")
            self.logger.info(f"  Recall: {recall:.4f}")
            self.logger.info(f"  F1 Score: {f1:.4f}")
            self.logger.info(f"  True Positives: {tp}, False Positives: {fp}")
            self.logger.info(f"  True Negatives: {tn}, False Negatives: {fn}")
            
            importances = model.feature_importances_
            features_df = pd.DataFrame({
                'Feature': self.features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            self.logger.info("Feature importance:")
            for idx, row in features_df.iterrows():
                self.logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
            
            # Save model
            model_path = os.path.join(self.model_dir, f"{ticker.replace('.', '_')}_model.joblib")
            joblib.dump(model, model_path)
            self.logger.info(f"Model saved to {model_path}")
            
            # Save features list used to train this model
            features_path = os.path.join(self.model_dir, f"{ticker.replace('.', '_')}_features.joblib")
            joblib.dump(self.features, features_path)
            self.logger.info(f"Feature list saved to {features_path}")
            
            self.models[ticker] = (model, self.features)
            return True
            
        except Exception as e:
            current_frame = sys._getframe()
            line_number = current_frame.f_lineno
            self.logger.error(f"Error training model for {ticker}: {str(e)} at line {line_number}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def load_model(self, ticker: str) -> Optional[Tuple[RandomForestClassifier, List[str]]]:
        """Load a previously trained model from disk and return the model and its feature names."""
        try:
            model_path = os.path.join(self.model_dir, f"{ticker.replace('.', '_')}_model.joblib")
            features_path = os.path.join(self.model_dir, f"{ticker.replace('.', '_')}_features.joblib")
            
            if not os.path.exists(model_path):
                self.logger.warning(f"No saved model found for {ticker}")
                return None
                
            self.logger.info(f"Loading model for {ticker} from {model_path}")
            model = joblib.load(model_path)
            
            # Check if we have saved features
            if os.path.exists(features_path):
                self.logger.info(f"Loading feature list from {features_path}")
                model_features = joblib.load(features_path)
            else:
                # For backward compatibility with older models
                self.logger.warning(f"No feature list found for model {ticker}, assuming older model with 7 features")
                # Older models likely had these 7 features
                model_features = [
                    'RSI', 'MACD', 'MACD_signal', 'BB_width', 'BB_pct', 
                    'SMA_20', 'EMA_12'
                ]
                # Save feature list for future use
                joblib.dump(model_features, features_path)
                self.logger.info(f"Saved inferred feature list to {features_path}")
                
            return model, model_features
            
        except Exception as e:
            current_frame = sys._getframe()
            line_number = current_frame.f_lineno
            self.logger.error(f"Error loading model for {ticker}: {str(e)} at line {line_number}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def predict_next_move(self, ticker: str, model: RandomForestClassifier, 
                         latest_data: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """Predict the next price movement for a stock."""
        try:
            self.logger.info(f"Predicting next move for {ticker}")
            
            # Check that all required features for this model are available
            # We're not using self.features here because we're only checking what this specific model needs
            required_features = latest_data.columns.tolist()
            missing_cols = [col for col in required_features if col not in latest_data.columns]
            if missing_cols:
                self.logger.error(f"Missing required features for prediction: {missing_cols}")
                return {"signal": "Error", "confidence": 0.0, "error": "Missing features"}
            
            X = latest_data
            self.logger.info(f"Making prediction using features: {X.columns.tolist()}")
            
            prediction = model.predict(X)[0]
            probabilities = model.predict_proba(X)[0]
            confidence = probabilities[prediction]
            signal = "Up" if prediction == 1 else "Down"
            
            self.logger.info(f"PREDICTION for {ticker}: {signal.upper()} with {confidence:.2%} confidence")
            
            return {
                "signal": signal,
                "confidence": confidence,
                "probability_up": probabilities[1],
                "probability_down": probabilities[0]
            }
            
        except Exception as e:
            current_frame = sys._getframe()
            line_number = current_frame.f_lineno
            self.logger.error(f"Error predicting for {ticker}: {str(e)} at line {line_number}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"signal": "Error", "confidence": 0.0, "error": str(e)}
    
    def process_ticker(self, ticker: str) -> Dict[str, Union[str, float]]:
        """Process a single ticker: download data, calculate indicators, train or load model, and predict."""
        self.logger.info(f"Processing {ticker}")
        
        df = self.download_stock_data(ticker)
        if df is None:
            return {"signal": "Error", "confidence": 0.0, "error": "Failed to download data"}
        
        df = self.add_technical_indicators(df)
        if df is None:
            return {"signal": "Error", "confidence": 0.0, "error": "Failed to calculate indicators"}
        
        model_data = self.load_model(ticker)
        if model_data is None:
            self.logger.info(f"No existing model found for {ticker}, training new model")
            df = self.generate_target(df)
            if df is None:
                return {"signal": "Error", "confidence": 0.0, "error": "Failed to generate target"}
            
            success = self.train_model(df, ticker)
            if not success:
                return {"signal": "Error", "confidence": 0.0, "error": "Failed to train model"}
            
            model_data = self.models[ticker]
        
        model, model_features = model_data
        self.logger.info(f"Loaded model with features: {model_features}")
        
        self.data[ticker] = df
        # Only use the features that the model was trained with
        latest_data = df[model_features].tail(1)
        return self.predict_next_move(ticker, model, latest_data)
    
    def predict_stocks(self, tickers: List[str]) -> Dict[str, Dict]:
        """Main method to predict multiple stocks"""
        self.tickers = tickers
        results = {}
        
        for ticker in self.tickers:
            try:
                result = self.process_ticker(ticker)
                results[ticker] = result
                self.logger.info(f"Prediction for {ticker}: {result}")
            except Exception as e:
                current_frame = sys._getframe()
                line_number = current_frame.f_lineno
                self.logger.error(f"Error processing {ticker}: {str(e)} at line {line_number}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                results[ticker] = {"signal": "Error", "confidence": 0.0, "error": str(e)}
        
        return results
