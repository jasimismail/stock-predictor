"""
LightGBM model implementation.
"""

import logging
import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import traceback
import sys

from .base_model import BaseModel

# Configure logger to capture tracebacks
logger = logging.getLogger('stock_predictor')
logger.setLevel(logging.INFO)

class LightGBMModel(BaseModel):
    """
    LightGBM implementation for stock prediction.
    """
    
    def __init__(self, model_dir: str = "models", test_size: float = 0.2):
        """
        Initialize the LightGBM model.
        
        Args:
            model_dir: Directory to store model files
            test_size: Proportion of data to use for testing
        """
        super().__init__(model_dir)
        self.test_size = test_size
        # You can customize the features needed for LightGBM
        # or keep the same as RandomForest
        self.features = [
            'RSI', 'MACD', 'MACD_signal', 'BB_width', 'BB_pct', 
            'SMA_20', 'EMA_12', 'Volume_SMA', 'Volume_Ratio', 'ATR'
        ]
        logger.info(f"LightGBMModel initialized with test_size={test_size}")
    
    def _log_error(self, message: str, exc_info: bool = True):
        """
        Helper method to log errors with traceback.
        
        Args:
            message: Error message
            exc_info: Whether to include exception info
        """
        logger.error(message, exc_info=exc_info)
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Train the LightGBM model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            logger.info("Starting LightGBM model training")
            
            if X.empty or y.empty:
                self._log_error("Empty training data provided")
                return False
                
            if len(X) != len(y):
                self._log_error(f"Feature and target data length mismatch: X={len(X)}, y={len(y)}")
                return False
            
            # Verify all required features are present
            if not self.validate_features(X):
                return False
            
            # Use only the required features
            X = X[self.features]
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )
            
            # Create and train the model
            self.model = LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
                n_jobs=-1,
                verbose=-1  # Suppress LightGBM output
            )
            
            logger.info(f"Fitting model on {len(X_train)} training samples")
            self.model.fit(
                X_train, 
                y_train,
                eval_set=[(X_test, y_test)],
                eval_metric='binary_logloss',
                early_stopping_rounds=10,
                verbose=False
            )
            
            # Evaluate the model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
            
            logger.info(f"Model performance:")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1 Score: {f1:.4f}")
            logger.info(f"  True Positives: {tp}, False Positives: {fp}")
            logger.info(f"  True Negatives: {tn}, False Negatives: {fn}")
            
            # Log feature importance
            importances = self.model.feature_importances_
            features_df = pd.DataFrame({
                'Feature': self.features,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            logger.info("Feature importance:")
            for idx, row in features_df.iterrows():
                logger.info(f"  {row['Feature']}: {row['Importance']:.4f}")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            self._log_error(f"Error training LightGBM model: {str(e)}")
            return False
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """
        Make a prediction with the LightGBM model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with prediction results
        """
        try:
            logger.info("Making prediction with LightGBM model")
            
            if self.model is None:
                self._log_error("Model not trained or loaded")
                return {"signal": "Error", "confidence": 0.0, "error": "Model not ready"}
            
            if X.empty:
                self._log_error("Empty prediction data provided")
                return {"signal": "Error", "confidence": 0.0, "error": "Empty input data"}
            
            # Verify all required features are present
            if not self.validate_features(X):
                return {"signal": "Error", "confidence": 0.0, "error": "Missing features"}
            
            # Use only the required features
            X = X[self.features]
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = probabilities[prediction]
            signal = "Buy" if prediction == 1 else "Sell"
            
            logger.info(f"PREDICTION: {signal} with {confidence:.2%} confidence")
            
            return {
                "signal": signal,
                "confidence": confidence,
                "probability_up": probabilities[1],
                "probability_down": probabilities[0]
            }
            
        except Exception as e:
            self._log_error(f"Error making prediction: {str(e)}")
            return {"signal": "Error", "confidence": 0.0, "error": str(e)}
    
    def save(self, model_path: str) -> bool:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            True if save was successful, False otherwise
        """
        try:
            if not self.is_trained or self.model is None:
                self._log_error("Cannot save untrained model")
                return False
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Save features list
            features_path = model_path.replace('_model.joblib', '_features.joblib')
            joblib.dump(self.features, features_path)
            logger.info(f"Feature list saved to {features_path}")
            
            return True
            
        except Exception as e:
            self._log_error(f"Error saving model: {str(e)}")
            return False
    
    def load(self, model_path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to load the model from
            
        Returns:
            True if load was successful, False otherwise
        """
        try:
            if not os.path.exists(model_path):
                self._log_error(f"No saved model found at {model_path}")
                return False
          
            # Load model
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
          
            # Load features list
            features_path = model_path.replace('_model.joblib', '_features.joblib')
            if os.path.exists(features_path):
                self.features = joblib.load(features_path)
                logger.info(f"Feature list loaded from {features_path} with {len(self.features)} features: {self.features}")
            else:
                logger.warning(f"No feature list found at {features_path}, using default features: {self.features}")
          
            self.is_trained = True
            return True
          
        except Exception as e:
            self._log_error(f"Error loading model from {model_path}: {str(e)}")
            return False

    def get_required_features(self) -> List[str]:
        """
        Get the list of features required by the model.
        
        Returns:
            List of required feature names
        """
        return self.features

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
            self._log_error(f"Missing required features: {missing_cols}")
            logger.error(f"Available columns: {X.columns.tolist()}")
            logger.error(f"Required features: {self.features}")
            return False
          
        # Also check for NaN values in the features
        nan_cols = [col for col in self.features if X[col].isna().any()]
        if nan_cols:
            self._log_error(f"NaN values found in features: {nan_cols}")
            return False
          
        return True
