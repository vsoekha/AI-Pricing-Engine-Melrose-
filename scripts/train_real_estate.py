"""Training script for real-estate sold_within_180d prediction using LightGBM."""

import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
import joblib
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# LightGBM wrapper for sklearn calibration
class LightGBMWrapper:
    """Wrapper for LightGBM model to work with sklearn calibration."""
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        # Model is already trained, just return self
        return self
    
    def predict_proba(self, X):
        # LightGBM predict returns probabilities for binary classification
        # Reshape to (n_samples, n_classes) format
        proba = self.model.predict(X)
        # Return shape (n_samples, 2) for binary classification
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X):
        # Return class predictions
        return (self.model.predict(X) > 0.5).astype(int)


def load_data(data_path: str, delimiter: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """Load real-estate data from CSV.
    
    Args:
        data_path: Path to CSV file
        delimiter: CSV delimiter (auto-detected if None)
        
    Returns:
        Tuple of (features DataFrame, target Series)
    """
    print(f"Loading data from {data_path}...")
    
    # Try to auto-detect delimiter
    if delimiter is None:
        with open(data_path, 'r') as f:
            first_line = f.readline()
            if ';' in first_line:
                delimiter = ';'
            else:
                delimiter = ','
    
    df = pd.read_csv(data_path, delimiter=delimiter)
    
    # Ensure target column exists
    if 'sold_within_180d' not in df.columns:
        raise ValueError("Target column 'sold_within_180d' not found in data")
    
    # Separate features and target
    y = df['sold_within_180d']
    X = df.drop('sold_within_180d', axis=1)
    
    print(f"Loaded {len(X)} samples with {len(X.columns)} features")
    print(f"Target distribution:\n{y.value_counts()}")
    
    return X, y


def prepare_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    categorical_features: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """Prepare features for LightGBM.
    
    Args:
        X_train: Training features
        X_test: Test features
        categorical_features: List of categorical feature names
        
    Returns:
        Tuple of (prepared X_train, prepared X_test, feature info dict)
    """
    # Auto-detect categorical features if not provided
    if categorical_features is None:
        categorical_features = X_train.select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
    
    # Convert categorical columns to category type for LightGBM
    for col in categorical_features:
        if col in X_train.columns:
            # Combine train and test to get all categories
            all_categories = pd.concat([X_train[col], X_test[col]]).unique()
            X_train[col] = pd.Categorical(X_train[col], categories=all_categories)
            X_test[col] = pd.Categorical(X_test[col], categories=all_categories)
    
    # Get numeric features
    numeric_features = X_train.select_dtypes(
        include=[np.number]
    ).columns.tolist()
    
    # Remove any NaN values in numeric features (fill with median)
    for col in numeric_features:
        if X_train[col].isna().any():
            median_val = X_train[col].median()
            X_train[col] = X_train[col].fillna(median_val)
            X_test[col] = X_test[col].fillna(median_val)
    
    # Fill NaN in categorical features with 'missing'
    for col in categorical_features:
        if col in X_train.columns:
            X_train[col] = X_train[col].cat.add_categories(['missing']).fillna('missing')
            if 'missing' not in X_test[col].cat.categories:
                X_test[col] = X_test[col].cat.add_categories(['missing'])
            X_test[col] = X_test[col].fillna('missing')
    
    feature_info = {
        'categorical_features': categorical_features,
        'numeric_features': numeric_features,
        'all_features': list(X_train.columns)
    }
    
    print(f"\nFeature preparation:")
    print(f"  Categorical features: {len(categorical_features)}")
    print(f"  Numeric features: {len(numeric_features)}")
    
    return X_train, X_test, feature_info


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical_features: List[str],
    params: Optional[Dict] = None
) -> lgb.Booster:
    """Train LightGBM model.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        categorical_features: List of categorical feature names
        params: LightGBM parameters
        
    Returns:
        Trained LightGBM booster
    """
    if params is None:
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
    
    # Get categorical feature indices
    cat_indices = [
        X_train.columns.get_loc(col)
        for col in categorical_features
        if col in X_train.columns
    ]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train,
        label=y_train,
        categorical_feature=cat_indices,
        free_raw_data=False
    )
    
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        categorical_feature=cat_indices,
        free_raw_data=False,
        reference=train_data
    )
    
    print("\nTraining LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=100)
        ]
    )
    
    return model


def calibrate_model(
    base_model: lgb.Booster,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str = 'isotonic'
) -> CalibratedClassifierCV:
    """Calibrate model probabilities.
    
    Args:
        base_model: Trained LightGBM model
        X_train: Training features
        y_train: Training target
        method: Calibration method ('isotonic' or 'sigmoid')
        
    Returns:
        Calibrated classifier
    """
    print(f"\nCalibrating probabilities using {method} method...")
    
    # Use a subset for calibration to avoid overfitting (skip if dataset too small)
    if len(X_train) > 10:
        stratify_cal = y_train if len(X_train) > 20 else None
        X_cal, _, y_cal, _ = train_test_split(
            X_train, y_train, test_size=0.5, random_state=42, stratify=stratify_cal
        )
    else:
        # Use all training data if too small for calibration split
        X_cal, y_cal = X_train, y_train
        print("Warning: Dataset too small for calibration split, using all training data")
    
    # Create wrapper
    wrapper = LightGBMWrapper(base_model)
    
    # Determine CV folds based on dataset size
    if len(X_cal) < 6:
        # Too small for CV, skip calibration and return wrapper directly
        print("Warning: Dataset too small for calibration, returning uncalibrated model")
        return wrapper
    elif len(X_cal) < 10:
        cv_folds = 2
    else:
        cv_folds = 3
    
    # Calibrate
    calibrated = CalibratedClassifierCV(
        wrapper,
        method=method,
        cv=cv_folds,
        n_jobs=-1
    )
    calibrated.fit(X_cal, y_cal)
    
    return calibrated


def evaluate_model(
    model: lgb.Booster,
    calibrated_model: CalibratedClassifierCV,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict:
    """Evaluate model performance.
    
    Args:
        model: Base LightGBM model
        calibrated_model: Calibrated model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary of evaluation metrics
    """
    print("\nEvaluating model...")
    
    # Get base model predictions
    y_pred_base = (model.predict(X_test) > 0.5).astype(int)
    y_proba_base = model.predict(X_test)
    
    # Get calibrated predictions
    y_pred_cal = calibrated_model.predict(X_test)
    y_proba_cal = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics for base model
    metrics_base = {
        'base_accuracy': accuracy_score(y_test, y_pred_base),
        'base_precision': precision_score(y_test, y_pred_base, zero_division=0),
        'base_recall': recall_score(y_test, y_pred_base, zero_division=0),
        'base_f1': f1_score(y_test, y_pred_base, zero_division=0),
        'base_roc_auc': roc_auc_score(y_test, y_proba_base)
    }
    
    # Calculate metrics for calibrated model
    metrics_cal = {
        'calibrated_accuracy': accuracy_score(y_test, y_pred_cal),
        'calibrated_precision': precision_score(y_test, y_pred_cal, zero_division=0),
        'calibrated_recall': recall_score(y_test, y_pred_cal, zero_division=0),
        'calibrated_f1': f1_score(y_test, y_pred_cal, zero_division=0),
        'calibrated_roc_auc': roc_auc_score(y_test, y_proba_cal)
    }
    
    # Combine metrics
    metrics = {**metrics_base, **metrics_cal}
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    print("\nBase Model:")
    print(f"  Accuracy:  {metrics['base_accuracy']:.4f}")
    print(f"  Precision: {metrics['base_precision']:.4f}")
    print(f"  Recall:    {metrics['base_recall']:.4f}")
    print(f"  F1 Score:  {metrics['base_f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['base_roc_auc']:.4f}")
    
    print("\nCalibrated Model:")
    print(f"  Accuracy:  {metrics['calibrated_accuracy']:.4f}")
    print(f"  Precision: {metrics['calibrated_precision']:.4f}")
    print(f"  Recall:    {metrics['calibrated_recall']:.4f}")
    print(f"  F1 Score:  {metrics['calibrated_f1']:.4f}")
    print(f"  ROC-AUC:   {metrics['calibrated_roc_auc']:.4f}")
    
    print("\nClassification Report (Calibrated):")
    print(classification_report(y_test, y_pred_cal))
    
    print("\nConfusion Matrix (Calibrated):")
    print(confusion_matrix(y_test, y_pred_cal))
    
    return metrics


def save_model(
    model: lgb.Booster,
    calibrated_model: CalibratedClassifierCV,
    feature_info: Dict,
    metrics: Dict,
    model_dir: Path
):
    """Save model and metadata.
    
    Args:
        model: Base LightGBM model
        calibrated_model: Calibrated model
        feature_info: Feature information dictionary
        metrics: Evaluation metrics
        model_dir: Directory to save model
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Save base model
    base_model_path = model_dir / "lightgbm_base.txt"
    model.save_model(str(base_model_path))
    print(f"\nSaved base model to {base_model_path}")
    
    # Save calibrated model
    calibrated_model_path = model_dir / "lightgbm_calibrated.joblib"
    joblib.dump(calibrated_model, calibrated_model_path)
    print(f"Saved calibrated model to {calibrated_model_path}")
    
    # Save feature info
    feature_info_path = model_dir / "feature_info.json"
    with open(feature_info_path, 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"Saved feature info to {feature_info_path}")
    
    # Save metrics
    metrics_path = model_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LightGBM model for real-estate prediction')
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to CSV file with real-estate data'
    )
    parser.add_argument(
        '--categorical',
        type=str,
        nargs='*',
        default=None,
        help='List of categorical feature names (auto-detected if not provided)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Test set size (default: 0.2)'
    )
    parser.add_argument(
        '--calibration-method',
        type=str,
        default='isotonic',
        choices=['isotonic', 'sigmoid'],
        help='Calibration method (default: isotonic)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for models (default: models/)'
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        output_dir = Path(__file__).parent.parent / "models"
    else:
        output_dir = Path(args.output_dir)
    
    print("="*60)
    print("REAL-ESTATE PREDICTION MODEL TRAINING")
    print("="*60)
    
    # Load data
    X, y = load_data(args.data)
    
    # Split data (skip stratification if dataset is too small)
    stratify = y if len(X) > 10 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=stratify
    )
    
    print(f"\nData split:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    
    # Prepare features
    X_train, X_test, feature_info = prepare_features(
        X_train,
        X_test,
        categorical_features=args.categorical
    )
    
    # Further split training data for validation (skip if dataset too small)
    if len(X_train) > 5:
        stratify_val = y_train if len(X_train) > 10 else None
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=42,
            stratify=stratify_val
        )
    else:
        # Use all training data if too small for validation split
        X_train_fit, X_val, y_train_fit, y_val = X_train, X_train, y_train, y_train
        print("Warning: Dataset too small for validation split, using all data for training")
    
    # Train LightGBM model
    model = train_lightgbm(
        X_train_fit,
        y_train_fit,
        X_val,
        y_val,
        categorical_features=feature_info['categorical_features']
    )
    
    # Calibrate model
    calibrated_model = calibrate_model(
        model,
        X_train_fit,
        y_train_fit,
        method=args.calibration_method
    )
    
    # Evaluate model
    metrics = evaluate_model(
        model,
        calibrated_model,
        X_test,
        y_test
    )
    
    # Save model
    save_model(
        model,
        calibrated_model,
        feature_info,
        metrics,
        output_dir
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()

