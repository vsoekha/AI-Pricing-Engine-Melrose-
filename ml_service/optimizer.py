"""Optimization module for hyperparameter tuning."""

from typing import Dict, Any, Optional
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


class Optimizer:
    """Hyperparameter optimizer."""
    
    def __init__(self, model_class=RandomForestClassifier):
        """Initialize the optimizer.
        
        Args:
            model_class: Model class to optimize
        """
        self.model_class = model_class
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Optional[Dict[str, Any]] = None,
        cv: int = 5,
        scoring: str = 'accuracy',
        n_jobs: int = -1
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search.
        
        Args:
            X: Training features
            y: Training labels
            param_grid: Parameter grid for grid search
            cv: Number of cross-validation folds
            scoring: Scoring metric
            n_jobs: Number of parallel jobs
            
        Returns:
            Dictionary with best parameters and score
        """
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            }
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Create base model
        base_model = self.model_class(random_state=42)
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        # Store results
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_estimator': grid_search.best_estimator_,
            'scaler': scaler
        }
    
    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters from last optimization.
        
        Returns:
            Best parameters dictionary or None
        """
        return self.best_params
    
    def get_best_score(self) -> Optional[float]:
        """Get the best score from last optimization.
        
        Returns:
            Best score or None
        """
        return self.best_score

