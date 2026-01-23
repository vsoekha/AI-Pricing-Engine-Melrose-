"""Price optimization module for maximizing expected revenue."""

from typing import Callable, List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class PriceResult:
    """Result for a single price point."""
    price: float
    sale_probability: float
    expected_revenue: float
    rank: int = 0


class PriceOptimizer:
    """Optimizes price to maximize expected revenue (price * sale_probability)."""
    
    def __init__(
        self,
        prediction_function: Callable,
        price_column: str = 'price'
    ):
        """Initialize the price optimizer.
        
        Args:
            prediction_function: Function that takes features dict/DataFrame
                                 and returns sale probability (0-1)
            price_column: Name of the price column in features
        """
        self.prediction_function = prediction_function
        self.price_column = price_column
    
    def simulate_price_points(
        self,
        base_price: float,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        num_points: int = 100,
        price_step: Optional[float] = None,
        strategy: str = 'linear'
    ) -> np.ndarray:
        """Generate price points to simulate.
        
        Args:
            base_price: Base/reference price
            min_price: Minimum price to test (default: 0.5 * base_price)
            max_price: Maximum price to test (default: 1.5 * base_price)
            num_points: Number of price points to test
            price_step: Fixed step size between prices (overrides num_points)
            strategy: 'linear' or 'log' spacing
            
        Returns:
            Array of price points
        """
        if min_price is None:
            min_price = base_price * 0.5
        if max_price is None:
            max_price = base_price * 1.5
        
        if price_step is not None:
            prices = np.arange(min_price, max_price + price_step, price_step)
        elif strategy == 'log':
            prices = np.logspace(
                np.log10(min_price),
                np.log10(max_price),
                num_points
            )
        else:  # linear
            prices = np.linspace(min_price, max_price, num_points)
        
        return prices
    
    def predict_sale_probability(
        self,
        features: Dict,
        price: float
    ) -> float:
        """Get sale probability for a given price.
        
        Args:
            features: Feature dictionary (will be updated with price)
            price: Price to test
            
        Returns:
            Sale probability (0-1)
        """
        # Create a copy of features and update price
        test_features = features.copy()
        test_features[self.price_column] = price
        
        # Get prediction
        proba = self.prediction_function(test_features)
        
        # Ensure it's a scalar probability
        if isinstance(proba, (list, np.ndarray)):
            proba = proba[0] if len(proba) > 0 else 0.0
        elif isinstance(proba, dict):
            # If it's a dict, try to get probability for class 1
            proba = proba.get(1, proba.get('sold_within_180d', 0.0))
        
        return float(proba)
    
    def optimize(
        self,
        features: Dict,
        base_price: float,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        num_points: int = 100,
        price_step: Optional[float] = None,
        strategy: str = 'linear',
        top_n: int = 10
    ) -> Tuple[PriceResult, List[PriceResult]]:
        """Find optimal price and generate leaderboard.
        
        Args:
            features: Feature dictionary (without price or with base price)
            base_price: Base/reference price
            min_price: Minimum price to test
            max_price: Maximum price to test
            num_points: Number of price points to test
            price_step: Fixed step size between prices
            strategy: 'linear' or 'log' spacing
            top_n: Number of top results to include in leaderboard
            
        Returns:
            Tuple of (best_price_result, leaderboard)
        """
        # Generate price points
        prices = self.simulate_price_points(
            base_price=base_price,
            min_price=min_price,
            max_price=max_price,
            num_points=num_points,
            price_step=price_step,
            strategy=strategy
        )
        
        # Evaluate each price point
        results = []
        for price in prices:
            try:
                sale_prob = self.predict_sale_probability(features, price)
                expected_revenue = price * sale_prob
                
                results.append(PriceResult(
                    price=float(price),
                    sale_probability=float(sale_prob),
                    expected_revenue=float(expected_revenue)
                ))
            except Exception as e:
                # Skip price points that fail
                print(f"Warning: Failed to evaluate price {price}: {e}")
                continue
        
        if not results:
            raise ValueError("No valid price points evaluated")
        
        # Sort by expected revenue (descending)
        results.sort(key=lambda x: x.expected_revenue, reverse=True)
        
        # Add ranks
        for i, result in enumerate(results, 1):
            result.rank = i
        
        # Get best price
        best_price = results[0]
        
        # Get leaderboard (top N)
        leaderboard = results[:top_n]
        
        return best_price, leaderboard
    
    def optimize_dataframe(
        self,
        features_df: pd.DataFrame,
        base_price: float,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        num_points: int = 100,
        price_step: Optional[float] = None,
        strategy: str = 'linear',
        top_n: int = 10
    ) -> Tuple[PriceResult, List[PriceResult]]:
        """Optimize price using DataFrame input.
        
        Args:
            features_df: DataFrame with features (single row)
            base_price: Base/reference price
            min_price: Minimum price to test
            max_price: Maximum price to test
            num_points: Number of price points to test
            price_step: Fixed step size between prices
            strategy: 'linear' or 'log' spacing
            top_n: Number of top results in leaderboard
            
        Returns:
            Tuple of (best_price_result, leaderboard)
        """
        # Convert DataFrame to dict
        if len(features_df) != 1:
            raise ValueError("features_df must contain exactly one row")
        
        features = features_df.iloc[0].to_dict()
        
        return self.optimize(
            features=features,
            base_price=base_price,
            min_price=min_price,
            max_price=max_price,
            num_points=num_points,
            price_step=price_step,
            strategy=strategy,
            top_n=top_n
        )
    
    def create_leaderboard_df(
        self,
        leaderboard: List[PriceResult]
    ) -> pd.DataFrame:
        """Convert leaderboard to DataFrame for easy viewing.
        
        Args:
            leaderboard: List of PriceResult objects
            
        Returns:
            DataFrame with columns: rank, price, sale_probability, expected_revenue
        """
        data = {
            'rank': [r.rank for r in leaderboard],
            'price': [r.price for r in leaderboard],
            'sale_probability': [r.sale_probability for r in leaderboard],
            'expected_revenue': [r.expected_revenue for r in leaderboard]
        }
        return pd.DataFrame(data)
    
    def plot_optimization(
        self,
        features: Dict,
        base_price: float,
        min_price: Optional[float] = None,
        max_price: Optional[float] = None,
        num_points: int = 100,
        price_step: Optional[float] = None,
        strategy: str = 'linear'
    ) -> None:
        """Plot price optimization results (requires matplotlib).
        
        Args:
            features: Feature dictionary
            base_price: Base/reference price
            min_price: Minimum price to test
            max_price: Maximum price to test
            num_points: Number of price points to test
            price_step: Fixed step size between prices
            strategy: 'linear' or 'log' spacing
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting")
        
        # Generate price points
        prices = self.simulate_price_points(
            base_price=base_price,
            min_price=min_price,
            max_price=max_price,
            num_points=num_points,
            price_step=price_step,
            strategy=strategy
        )
        
        # Get predictions
        probabilities = []
        revenues = []
        
        for price in prices:
            try:
                prob = self.predict_sale_probability(features, price)
                probabilities.append(prob)
                revenues.append(price * prob)
            except:
                probabilities.append(np.nan)
                revenues.append(np.nan)
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot 1: Sale probability vs price
        ax1.plot(prices, probabilities, 'b-', linewidth=2)
        ax1.axvline(base_price, color='r', linestyle='--', label='Base Price')
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Sale Probability')
        ax1.set_title('Sale Probability vs Price')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Expected revenue vs price
        ax2.plot(prices, revenues, 'g-', linewidth=2)
        optimal_price = prices[np.nanargmax(revenues)]
        optimal_revenue = np.nanmax(revenues)
        ax2.axvline(optimal_price, color='r', linestyle='--', label=f'Optimal Price: ${optimal_price:.2f}')
        ax2.axvline(base_price, color='orange', linestyle='--', label='Base Price')
        ax2.scatter([optimal_price], [optimal_revenue], color='red', s=100, zorder=5)
        ax2.set_xlabel('Price')
        ax2.set_ylabel('Expected Revenue (Price × Probability)')
        ax2.set_title('Expected Revenue vs Price')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

