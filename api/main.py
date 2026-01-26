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
from fastapi.middleware.cors import CORSMiddleware
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

model = None
feature_info = None
ALL_FEATURES = []
CATEGORICAL_FEATURES = []
MODEL_LOADED = False
LOAD_ERROR = None

try:
    # Check if files exist
    if not MODEL_PATH.exists():
        LOAD_ERROR = f"Model file not found: {MODEL_PATH}"
        print(f"ERROR: {LOAD_ERROR}")
    elif not FEATURE_INFO_PATH.exists():
        LOAD_ERROR = f"Feature info file not found: {FEATURE_INFO_PATH}"
        print(f"ERROR: {LOAD_ERROR}")
    else:
        print(f"Loading model from: {MODEL_PATH}")
        print(f"Loading feature info from: {FEATURE_INFO_PATH}")
        model = joblib.load(MODEL_PATH)
        with open(FEATURE_INFO_PATH, "r") as f:
            feature_info = json.load(f)
        ALL_FEATURES = feature_info["all_features"]
        CATEGORICAL_FEATURES = feature_info.get("categorical_features", [])
        MODEL_LOADED = True
        print(f"Model loaded successfully! Features: {len(ALL_FEATURES)}")
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)
    import traceback
    print(f"ERROR: Failed to load model: {e}")
    print(traceback.format_exc())

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="Real Estate AI Price Recommendation API",
    description="AI-powered price optimization for real estate listings",
    version="1.0.0"
)

# ============================================================================
# CORS Configuration
# ============================================================================

# Allow CORS from Base44 and other common frontend domains
# For production, you can restrict to specific origins
# For testing, allowing all origins is easier
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for testing - restrict in production)
    # Uncomment below and comment above to restrict to specific domains:
    # allow_origins=[
    #     "https://preview--ai-property-pricer-ba64a598.base44.app",
    #     "https://*.base44.app",
    #     "http://localhost:3000",
    #     "http://localhost:5173",
    #     "http://localhost:8080",
    # ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
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
        "endpoint": "/recommend_price",
        "method": "POST",
        "docs": "/docs",
        "example_request": {
            "asset_type": "logistics",
            "city": "Rotterdam",
            "size_m2": 12000,
            "quality_score": 0.82,
            "noi_annual": 620000,
            "cap_rate_market": 0.065,
            "interest_rate": 0.025,
            "liquidity_index": 0.71,
            "list_price": 9500000,
            "comp_median_price": 9900000
        }
    }

@app.get("/health")
def health():
    """Health check endpoint."""
    import os
    response = {
        "status": "healthy" if MODEL_LOADED else "unhealthy",
        "model_loaded": MODEL_LOADED,
        "features_count": len(ALL_FEATURES) if MODEL_LOADED else 0
    }
    if not MODEL_LOADED:
        response["error"] = LOAD_ERROR
        response["model_path"] = str(MODEL_PATH)
        response["feature_info_path"] = str(FEATURE_INFO_PATH)
        response["model_path_exists"] = MODEL_PATH.exists()
        response["feature_info_path_exists"] = FEATURE_INFO_PATH.exists()
        response["current_directory"] = str(Path.cwd())
        response["app_root"] = str(APP_ROOT)
        response["api_file_location"] = str(Path(__file__).resolve())
        # List files in models directory
        models_dir = APP_ROOT / "models"
        if models_dir.exists():
            response["models_directory_files"] = [f.name for f in models_dir.iterdir()]
        else:
            response["models_directory_exists"] = False
    return response

@app.get("/recommend_price")
def recommend_price_info():
    """GET endpoint - Shows how to use the POST /recommend_price endpoint."""
    return {
        "message": "This endpoint requires a POST request with JSON data",
        "method": "POST",
        "content_type": "application/json",
        "example_request": {
            "asset_type": "logistics",
            "city": "Rotterdam",
            "size_m2": 12000,
            "quality_score": 0.82,
            "noi_annual": 620000,
            "cap_rate_market": 0.065,
            "interest_rate": 0.025,
            "liquidity_index": 0.71,
            "list_price": 9500000,
            "comp_median_price": 9900000
        },
        "example_curl": 'curl -X POST "http://localhost:8000/recommend_price" -H "Content-Type: application/json" -d \'{"asset_type": "logistics", "city": "Rotterdam", "size_m2": 12000, "quality_score": 0.82, "noi_annual": 620000, "cap_rate_market": 0.065, "interest_rate": 0.025, "liquidity_index": 0.71, "list_price": 9500000, "comp_median_price": 9900000}\'',
        "interactive_docs": "/docs",
        "note": "Visit /docs for an interactive interface to test this endpoint"
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
