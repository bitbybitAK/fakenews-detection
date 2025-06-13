"""
Machine learning module for TruthGuard.
Handles model training, evaluation, and inference.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, f1_score, precision_score,
                           recall_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FakeNewsClassifier:
    """Handles fake news classification model training and inference."""

    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the classifier.
        
        Args:
            model_type: Type of model to use ('xgboost' or 'random_forest')
        """
        self.model_type = model_type
        self.model = self._initialize_model()
        self.feature_columns = None
        self.mlflow_experiment = "truthguard"

    def _initialize_model(self):
        """Initialize the specified model type."""
        if self.model_type == 'xgboost':
            return XGBClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def prepare_data(self, df: pd.DataFrame, target_column: str = 'label') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of features and target
        """
        try:
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Store feature columns
            self.feature_columns = X.columns.tolist()
            
            return X, y
        except Exception as e:
            logger.error(f"Error preparing data: {e}")
            raise

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Train the model and log metrics with MLflow.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary of training metrics
        """
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Set up MLflow
            mlflow.set_experiment(self.mlflow_experiment)
            
            with mlflow.start_run():
                # Train model
                self.model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = self.model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted'),
                    'recall': recall_score(y_test, y_pred, average='weighted'),
                    'f1': f1_score(y_test, y_pred, average='weighted')
                }
                
                # Log metrics
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Log model
                mlflow.sklearn.log_model(self.model, "model")
                
                # Log feature importance
                if hasattr(self.model, 'feature_importances_'):
                    importance_dict = dict(zip(self.feature_columns, 
                                            self.model.feature_importances_))
                    mlflow.log_dict(importance_dict, "feature_importance.json")
                
                logger.info(f"Model training completed with metrics: {metrics}")
                return metrics
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of predictions
        """
        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Array of prediction probabilities
        """
        try:
            return self.model.predict_proba(X)
        except Exception as e:
            logger.error(f"Error getting prediction probabilities: {e}")
            raise

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Evaluate model performance.
        
        Args:
            X: Feature DataFrame
            y: True labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            # Make predictions
            y_pred = self.predict(X)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted')
            }
            
            # Generate classification report
            report = classification_report(y, y_pred, output_dict=True)
            
            # Generate confusion matrix
            cm = confusion_matrix(y, y_pred)
            
            logger.info(f"Model evaluation completed with metrics: {metrics}")
            return {
                'metrics': metrics,
                'classification_report': report,
                'confusion_matrix': cm
            }
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise

    def save_model(self, path: str) -> None:
        """
        Save the trained model.
        
        Args:
            path: Path to save the model
        """
        try:
            mlflow.sklearn.save_model(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, path: str) -> None:
        """
        Load a trained model.
        
        Args:
            path: Path to the saved model
        """
        try:
            self.model = mlflow.sklearn.load_model(path)
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise 