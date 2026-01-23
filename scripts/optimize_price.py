"""Script to optimize price for real-estate listings."""

import sys
from pathlib import Path
import pandas as pd
import joblib
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml_service.price_optimizer import PriceOptimizer


def load_model_and_features(model_dir: Path):
    """Load trained model and feature information.
    
    Args:
        model_dir: Directory containing saved model
        
    Returns:
        Tuple of (calibrated_model, feature_info)
    """
    # Load calibrated model
    model_path = model_dir / "lightgbm_calibrated.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    calibrated_model = joblib.load(model_path)
    print(f"Loaded model from {model_path}")
    
    # Load feature info
    feature_info_path = model_dir / "feature_info.json"
    if feature_info_path.exists():
        with open(feature_info_path, 'r') as f:
            feature_info = json.load(f)
    else:
        feature_info = None
        print("Warning: feature_info.json not found")
    
    return calibrated_model, feature_info


def create_prediction_function(model, feature_info=None):
    """Create a prediction function for the price optimizer.
    
    Args:
        model: Trained calibrated model
        feature_info: Feature information dictionary
        
    Returns:
        Prediction function that takes features dict and returns probability
    """
    def predict_sale_probability(features: dict) -> float:
        """Predict sale probability for given features.
        
        Args:
            features: Dictionary of features including price
            
        Returns:
            Sale probability (0-1)
        """
        # Convert features dict to DataFrame
        features_df = pd.DataFrame([features])
        
        # Ensure categorical features are properly formatted
        if feature_info:
            for cat_col in feature_info.get('categorical_features', []):
                if cat_col in features_df.columns:
                    # Try to get categories from feature_info, or infer from value
                    all_categories = feature_info.get('categories', {}).get(cat_col, None)
                    if all_categories:
                        # Ensure the current value is in categories
                        current_val = features_df[cat_col].iloc[0]
                        if current_val not in all_categories:
                            all_categories = list(all_categories) + [current_val]
                        features_df[cat_col] = pd.Categorical(
                            features_df[cat_col],
                            categories=all_categories
                        )
                    else:
                        # Convert to category type if not already
                        features_df[cat_col] = features_df[cat_col].astype('category')
        
        # Get prediction probability for class 1 (sold_within_180d = 1)
        proba = model.predict_proba(features_df)[0]
        
        # Return probability of class 1
        if len(proba) == 2:
            return float(proba[1])  # Probability of sold_within_180d = 1
        else:
            return float(proba[0])
    
    return predict_sale_probability


def main():
    """Main function for price optimization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize price for real-estate listing')
    parser.add_argument(
        '--model-dir',
        type=str,
        default='models',
        help='Directory containing trained model (default: models/)'
    )
    parser.add_argument(
        '--features',
        type=str,
        required=True,
        help='Path to CSV file with property features (single row) or JSON file'
    )
    parser.add_argument(
        '--base-price',
        type=float,
        required=True,
        help='Base/reference price'
    )
    parser.add_argument(
        '--min-price',
        type=float,
        default=None,
        help='Minimum price to test (default: 0.5 * base_price)'
    )
    parser.add_argument(
        '--max-price',
        type=float,
        default=None,
        help='Maximum price to test (default: 1.5 * base_price)'
    )
    parser.add_argument(
        '--num-points',
        type=int,
        default=100,
        help='Number of price points to test (default: 100)'
    )
    parser.add_argument(
        '--price-step',
        type=float,
        default=None,
        help='Fixed step size between prices (overrides num_points)'
    )
    parser.add_argument(
        '--strategy',
        type=str,
        default='linear',
        choices=['linear', 'log'],
        help='Price spacing strategy (default: linear)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top results in leaderboard (default: 10)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for leaderboard (optional)'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("PRICE OPTIMIZATION")
    print("="*60)
    
    # Load model
    model_dir = Path(args.model_dir)
    model, feature_info = load_model_and_features(model_dir)
    
    # Load features
    features_path = Path(args.features)
    if features_path.suffix == '.json':
        import json
        with open(features_path, 'r') as f:
            features = json.load(f)
    else:
        # Load from CSV
        df = pd.read_csv(features_path)
        if len(df) != 1:
            raise ValueError("Features file must contain exactly one row")
        features = df.iloc[0].to_dict()
    
    # Remove price from features if present (will be set by optimizer)
    if 'price' in features:
        print(f"Note: Removing existing price ({features['price']}) from features")
        del features['price']
    
    print(f"\nFeatures: {list(features.keys())}")
    print(f"Base price: ${args.base_price:,.2f}")
    
    # Create prediction function
    predict_fn = create_prediction_function(model, feature_info)
    
    # Create optimizer
    optimizer = PriceOptimizer(
        prediction_function=predict_fn,
        price_column='price'
    )
    
    # Optimize
    print("\nOptimizing price...")
    best_price, leaderboard = optimizer.optimize(
        features=features,
        base_price=args.base_price,
        min_price=args.min_price,
        max_price=args.max_price,
        num_points=args.num_points,
        price_step=args.price_step,
        strategy=args.strategy,
        top_n=args.top_n
    )
    
    # Display results
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nBest Price: ${best_price.price:,.2f}")
    print(f"Sale Probability: {best_price.sale_probability:.4f} ({best_price.sale_probability*100:.2f}%)")
    print(f"Expected Revenue: ${best_price.expected_revenue:,.2f}")
    print(f"\nImprovement over base price:")
    base_revenue = args.base_price * predict_fn({**features, 'price': args.base_price})
    improvement = best_price.expected_revenue - base_revenue
    improvement_pct = (improvement / base_revenue * 100) if base_revenue > 0 else 0
    print(f"  Base expected revenue: ${base_revenue:,.2f}")
    print(f"  Improvement: ${improvement:,.2f} ({improvement_pct:+.2f}%)")
    
    # Display leaderboard
    print("\n" + "="*60)
    print(f"TOP {args.top_n} PRICE POINTS")
    print("="*60)
    leaderboard_df = optimizer.create_leaderboard_df(leaderboard)
    print(leaderboard_df.to_string(index=False))
    
    # Save leaderboard if requested
    if args.output:
        output_path = Path(args.output)
        leaderboard_df.to_csv(output_path, index=False)
        print(f"\nLeaderboard saved to {output_path}")


if __name__ == "__main__":
    main()

