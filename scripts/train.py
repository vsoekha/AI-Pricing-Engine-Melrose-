"""Training script for the ML model."""

import sys
from pathlib import Path
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_service.model import MLModel
from ml_service.optimizer import Optimizer


def generate_sample_data(n_samples=1000, n_features=10, n_classes=2):
    """Generate sample training data.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        
    Returns:
        Tuple of (X, y)
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2,
        n_redundant=n_features // 4,
        n_classes=n_classes,
        random_state=42
    )
    return X, y


def main():
    """Main training function."""
    print("Starting training...")
    
    # Generate or load your data here
    print("Generating sample data...")
    X, y = generate_sample_data(n_samples=1000, n_features=10)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Optimize hyperparameters
    print("\nOptimizing hyperparameters...")
    optimizer = Optimizer()
    optimization_result = optimizer.optimize(X_train, y_train)
    
    print(f"Best parameters: {optimization_result['best_params']}")
    print(f"Best CV score: {optimization_result['best_score']:.4f}")
    
    # Create model with best parameters
    print("\nTraining final model...")
    model = MLModel(
        model=optimization_result['best_estimator'],
        scaler=optimization_result['scaler']
    )
    
    # Train the model
    model.train(X_train, y_train)
    
    # Evaluate
    train_score = model.model.score(
        model.scaler.transform(X_train),
        y_train
    )
    test_score = model.model.score(
        model.scaler.transform(X_test),
        y_test
    )
    
    print(f"\nTraining accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Save model
    model_path = Path(__file__).parent.parent / "models" / "model.joblib"
    print(f"\nSaving model to {model_path}...")
    model.save(str(model_path))
    print("Training complete!")


if __name__ == "__main__":
    main()

