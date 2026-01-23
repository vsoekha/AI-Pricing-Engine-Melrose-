"""Model definition and utilities."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


class MLModel:
    """Machine learning model wrapper."""
    
    def __init__(self, model=None, scaler=None):
        """Initialize the model.
        
        Args:
            model: Trained model (default: RandomForestClassifier)
            scaler: Feature scaler (default: StandardScaler)
        """
        self.model = model or RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = scaler or StandardScaler()
    
    def train(self, X, y):
        """Train the model.
        
        Args:
            X: Training features
            y: Training labels
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
    
    def predict(self, X):
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Get prediction probabilities.
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities
        """
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, model_path: str):
        """Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        model_dir = Path(model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler
        }, model_path)
    
    @classmethod
    def load(cls, model_path: str):
        """Load a model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded MLModel instance
        """
        data = joblib.load(model_path)
        return cls(model=data['model'], scaler=data['scaler'])

