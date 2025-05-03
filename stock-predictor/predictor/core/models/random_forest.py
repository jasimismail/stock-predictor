"""
Random Forest model implementation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import joblib

from .base_model import BaseModel

logger = logging.getLogger('stock_predictor')

class RandomForestModel(BaseModel):
    """
    Random Forest implementation for stock prediction.
    """
    
    def __init__(self, model_dir: str = "models", test_size: float = 0.2):
        """
        Initialize the Random Forest model.
        
        Args:
            model_dir: Directory to store model files
            test_size: Proportion of data to use for testing
        """
        super().__init__(model_dir)
        self.test_size = test_size
        self.features = [
            'RSI', 'MACD', 'MACD_signal', 'BB_width', 'BB_pct', 
            'SMA_20', 'EMA_12', 'Volume_SMA', 'Volume_Ratio', 'ATR'
        ]
        logger.info(f"RandomForestModel initialized with test_size={test_size}")
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> bool:
        """
        Train the Random Forest model.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            True if training was successful, False otherwise
        """
        try:
            logger.info("Training Random Forest model")
            
            # Verify all required features are present using base class method
            if not self.validate_features(X):
                return False
            
            # Use only the required features
            X = X[self.features]
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, shuffle=False
            )
            
            # Create and train the model
            self.model = RandomForestClassifier(
                n_estimators=100, 
                max_depth=5, 
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
            
            logger.info(f"Fitting model on {len(X_train)} training samples")
            self.model.fit(X_train, y_train)
            
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
            
            return True
            
        except Exception as e:
            logger.error(f"Error training Random Forest model: {str(e)}")
            return False
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """
        Make a prediction with the Random Forest model.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Dictionary with prediction results
        """
        try:
            logger.info("Making prediction with Random Forest model")
            
            if self.model is None:
                logger.error("Model not trained or loaded")
                return {"signal": "Error", "confidence": 0.0, "error": "Model not ready"}
            
            # Verify all required features are present using base class method
            if not self.validate_features(X):
                return {"signal": "Error", "confidence": 0.0, "error": "Missing features"}
            
            # Use only the required features
            X = X[self.features]
            
            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            confidence = probabilities[prediction]
            signal = "UP" if prediction == 1 else "DOWN"
            
            logger.info(f"PREDICTION: {signal} with {confidence:.2%} confidence")
            
            return {
                "signal": signal,
                "confidence": confidence,
                "probability_up": probabilities[1],
                "probability_down": probabilities[0]
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {"signal": "Error", "confidence": 0.0, "error": str(e)}

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
                logger.warning(f"No saved model found at {model_path}")
                return False
            
            # Load model
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")
            
            # Load features list
            features_path = model_path.replace('_model.joblib', '_features.joblib')
            if os.path.exists(features_path):
                self.features = joblib.load(features_path)
                logger.info(f"Feature list loaded from {features_path}")
            else:
                logger.warning(f"No feature list found at {features_path}, using default features")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False

    def get_required_features(self) -> List[str]:
        """
        Get the list of features required by the model.
        
        Returns:
            List of required feature names
        """
        return self.features
