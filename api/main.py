"""
Clean AI Demo API for Real Estate Price Recommendations
Single endpoint that provides AI-powered price optimization.
"""

from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

# ============================================================================
# Configuration
# ============================================================================

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_ROOT / "models" / "lightgbm_calibrated.joblib"
FEATURE_INFO_PATH = APP_ROOT / "models" / "feature_info.json"

# ============================================================================
# LightGBM Wrapper (required for model loading)
# ============================================================================

class LightGBMWrapper:
    """Wrapper for LightGBM model to work with sklearn calibration."""
    
    def __init__(self, model):
        self.model = model
    
    def fit(self, X, y):
        return self
    
    def predict_proba(self, X):
        proba = self.model.predict(X)
        return np.column_stack([1 - proba, proba])
    
    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int)

# Make wrapper available for unpickling
import sys
sys.modules['__main__'].LightGBMWrapper = LightGBMWrapper

# ============================================================================
# Model Loading
# ============================================================================

try:
    model = joblib.load(MODEL_PATH)
    with open(FEATURE_INFO_PATH, "r") as f:
        feature_info = json.load(f)
    ALL_FEATURES = feature_info["all_features"]
    CATEGORICAL_FEATURES = feature_info.get("categorical_features", [])
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    print(f"Warning: Failed to load model: {e}")

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Real Estate AI Price Recommendation API",
    description="AI-powered price optimization for real estate listings",
    version="1.0.0"
)

# ============================================================================
# Request/Response Models
# ============================================================================

class PropertyRequest(BaseModel):
    """Property features for price recommendation."""
    asset_type: str = Field(..., description="Property type (e.g., 'logistics', 'office', 'resi')")
    city: str = Field(..., description="City name")
    size_m2: float = Field(..., gt=0, description="Property size in square meters")
    quality_score: float = Field(..., ge=0, le=1, description="Quality score (0-1)")
    noi_annual: float = Field(..., ge=0, description="Net Operating Income (annual)")
    cap_rate_market: float = Field(..., ge=0, le=1, description="Market cap rate (e.g., 0.065)")
    interest_rate: float = Field(..., ge=0, le=1, description="Interest rate (e.g., 0.025)")
    liquidity_index: float = Field(..., ge=0, le=1, description="Liquidity index (0-1)")
    list_price: float = Field(..., gt=0, description="Current listing price")
    comp_median_price: float = Field(..., gt=0, description="Comparable median price")

class PriceRecommendationResponse(BaseModel):
    """AI price recommendation response."""
    base_price: float = Field(..., description="Original listing price")
    base_sale_probability: float = Field(..., description="Sale probability at base price")
    recommended_price: float = Field(..., description="AI-recommended optimal price")
    recommended_sale_probability: float = Field(..., description="Sale probability at recommended price")
    expected_uplift: float = Field(..., description="Expected revenue uplift (recommended - base)")

# ============================================================================
# Helper Functions
# ============================================================================

def prepare_features(features: dict, price: float) -> pd.DataFrame:
    """
    Prepare features DataFrame for model prediction.
    
    Args:
        features: Property features dictionary
        price: Price to use for list_price
        
    Returns:
        DataFrame ready for model prediction
    """
    # Create feature dict with specified price
    feature_dict = features.copy()
    feature_dict["list_price"] = price
    
    # Ensure all features are present in correct order
    row = {k: feature_dict.get(k, None) for k in ALL_FEATURES}
    df = pd.DataFrame([row])
    
    # Handle categorical features
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("category")
    
    # Convert numeric features
    numeric_features = feature_info.get("numeric_features", [])
    for col in numeric_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if df[col].isna().any():
                df[col] = df[col].fillna(0)
    
    return df

def predict_sale_probability(features: dict, price: float) -> float:
    """
    Predict sale probability for given features and price.
    
    Args:
        features: Property features dictionary
        price: Listing price
        
    Returns:
        Sale probability (0-1)
    """
    df = prepare_features(features, price)
    proba = model.predict_proba(df)
    
    # Handle both calibrated and uncalibrated models
    if proba.shape[1] == 2:
        return float(proba[:, 1][0])
    else:
        return float(proba[0])

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Real Estate AI Price Recommendation API",
        "version": "1.0.0",
        "status": "operational" if MODEL_LOADED else "model_not_loaded",
        "endpoint": "/recommend_price"
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "features_count": len(ALL_FEATURES) if MODEL_LOADED else 0
    }

@app.post("/recommend_price", response_model=PriceRecommendationResponse)
def recommend_price(request: PropertyRequest):
    """
    Get AI-powered price recommendation.
    
    Computes optimal listing price by:
    1. Calculating sale probability at current price
    2. Simulating multiple price points (+/- 15%)
    3. Scoring each price as: price × sale_probability
    4. Returning the price with highest expected revenue
    """
    if not MODEL_LOADED:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure model files exist in models/ directory."
        )
    
    try:
        # Extract features
        features = request.model_dump()
        base_price = features["list_price"]
        
        # Calculate base sale probability
        base_prob = predict_sale_probability(features, base_price)
        
        # Simulate price points: +/- 15% in 1% increments
        price_multipliers = np.arange(0.85, 1.16, 0.01)
        best_score = -1
        best_price = base_price
        best_prob = base_prob
        
        for multiplier in price_multipliers:
            test_price = base_price * multiplier
            
            # Predict sale probability at this price
            prob = predict_sale_probability(features, test_price)
            
            # Calculate expected revenue (price × probability)
            score = test_price * prob
            
            # Track best option
            if score > best_score:
                best_score = score
                best_price = test_price
                best_prob = prob
        
        # Calculate expected uplift
        base_revenue = base_price * base_prob
        recommended_revenue = best_price * best_prob
        expected_uplift = recommended_revenue - base_revenue
        
        return PriceRecommendationResponse(
            base_price=round(base_price, 2),
            base_sale_probability=round(base_prob, 4),
            recommended_price=round(best_price, 2),
            recommended_sale_probability=round(best_prob, 4),
            expected_uplift=round(expected_uplift, 2)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Price recommendation failed: {str(e)}"
        )
