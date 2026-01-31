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

class ScenarioAnalysis(BaseModel):
    """Scenario analysis for different interest rate environments."""
    scenario_name: str = Field(..., description="Scenario name (base/optimistic/downside/stress)")
    interest_rate: float = Field(..., description="Interest rate for this scenario")
    recommended_price: float = Field(..., description="Optimal price in this scenario")
    sale_probability: float = Field(..., description="Sale probability in this scenario")
    expected_revenue: float = Field(..., description="Expected revenue (price × probability)")
    value_impact: float = Field(..., description="Value impact vs base case (%)")

class PriceRecommendationResponse(BaseModel):
    """Comprehensive AI price recommendation response."""
    # Core pricing
    base_price: float = Field(..., description="Original listing price")
    base_sale_probability: float = Field(..., description="Sale probability at base price")
    recommended_price: float = Field(..., description="AI-recommended optimal price")
    recommended_sale_probability: float = Field(..., description="Sale probability at recommended price")
    expected_uplift: float = Field(..., description="Expected revenue uplift (recommended - base)")
    
    # Price sensitivity
    price_range_up: float = Field(..., description="Maximum viable price increase (%)")
    price_range_down: float = Field(..., description="Maximum viable price decrease (%)")
    price_sensitivity: str = Field(..., description="Price sensitivity level (low/medium/high)")
    
    # Timing recommendation
    optimal_timing: str = Field(..., description="Best time to sell (e.g., 'Q1 2026')")
    timing_reason: str = Field(..., description="Reason for timing recommendation")
    timing_impact: float = Field(..., description="Expected impact of optimal timing vs now (%)")
    
    # Buyer demand
    buyer_demand_score: float = Field(..., description="Buyer demand score (0-10)")
    buyer_demand_level: str = Field(..., description="Buyer demand level (low/medium/high)")
    buyer_profiles: List[str] = Field(..., description="Types of buyers interested (pension funds, PE, etc.)")
    
    # Scenario analysis
    scenarios: List[ScenarioAnalysis] = Field(..., description="Analysis for different interest rate scenarios")
    
    # Market insights
    market_liquidity: str = Field(..., description="Market liquidity assessment")
    market_trend: str = Field(..., description="Current market trend (bullish/bearish/neutral)")
    comparable_yield: float = Field(..., description="Comparable yield in market")

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
        
        # ========================================================================
        # Additional Analytics (Demo - calculated from input data)
        # ========================================================================
        
        # 1. Price Sensitivity Range
        # Find price range where probability stays above threshold
        min_prob_threshold = base_prob * 0.8  # 80% of base probability
        price_range_up = 0.0
        price_range_down = 0.0
        
        # Test upward prices
        for multiplier in np.arange(1.0, 1.20, 0.01):
            test_price = base_price * multiplier
            prob = predict_sale_probability(features, test_price)
            if prob >= min_prob_threshold:
                price_range_up = (multiplier - 1.0) * 100
            else:
                break
        
        # Test downward prices
        for multiplier in np.arange(1.0, 0.80, -0.01):
            test_price = base_price * multiplier
            prob = predict_sale_probability(features, test_price)
            if prob >= min_prob_threshold:
                price_range_down = (1.0 - multiplier) * 100
            else:
                break
        
        # Price sensitivity level
        total_range = price_range_up + price_range_down
        if total_range > 20:
            price_sensitivity = "low"
        elif total_range > 10:
            price_sensitivity = "medium"
        else:
            price_sensitivity = "high"
        
        # 2. Timing Recommendation
        # Based on interest rate, liquidity, and market conditions
        current_interest = features["interest_rate"]
        liquidity = features["liquidity_index"]
        
        # Calculate optimal timing (simplified logic for demo)
        if current_interest < 0.03 and liquidity > 0.7:
            # Good conditions - recommend sooner
            optimal_timing = "Q1 2026"
            timing_reason = "Gunstige marktomstandigheden: lage rente en hoge liquiditeit"
            timing_impact = 8.5
        elif current_interest > 0.04 or liquidity < 0.5:
            # Challenging conditions - recommend waiting
            optimal_timing = "Q3 2027"
            timing_reason = "Wachten op betere marktomstandigheden: rente of liquiditeit verbeteren"
            timing_impact = 11.2
        else:
            # Neutral - recommend mid-term
            optimal_timing = "Q2 2026"
            timing_reason = "Gebalanceerde marktomstandigheden, optimale timing mid-term"
            timing_impact = 5.3
        
        # 3. Buyer Demand Score (0-10)
        # Based on asset type, quality, location, and market conditions
        asset_type_score = {"logistics": 8, "office": 6, "resi": 7}.get(features["asset_type"], 5)
        quality_score_normalized = features["quality_score"] * 10
        liquidity_score = features["liquidity_index"] * 10
        city_score = {"Rotterdam": 8, "Amsterdam": 9, "Utrecht": 7}.get(features["city"], 6)
        
        buyer_demand_score = (
            asset_type_score * 0.3 +
            quality_score_normalized * 0.2 +
            liquidity_score * 0.3 +
            city_score * 0.2
        )
        buyer_demand_score = min(10, max(0, buyer_demand_score))
        
        if buyer_demand_score >= 7:
            buyer_demand_level = "high"
        elif buyer_demand_score >= 4:
            buyer_demand_level = "medium"
        else:
            buyer_demand_level = "low"
        
        # Buyer profiles based on asset type
        buyer_profiles_map = {
            "logistics": ["Pensioenfondsen", "Institutionele investeerders", "Private equity"],
            "office": ["Pensioenfondsen", "Family offices", "Internationale investeerders"],
            "resi": ["Family offices", "Private equity", "HNW individuen"]
        }
        buyer_profiles = buyer_profiles_map.get(features["asset_type"], ["Gemengde investeerders"])
        
        # 4. Scenario Analysis (different interest rates)
        scenarios = []
        scenario_configs = [
            {"name": "base", "interest": current_interest, "label": "Base Case"},
            {"name": "optimistic", "interest": max(0.01, current_interest - 0.005), "label": "Optimistic (rente -0.5%)"},
            {"name": "downside", "interest": current_interest + 0.01, "label": "Downside (rente +1.0%)"},
            {"name": "stress", "interest": current_interest + 0.015, "label": "Stress Test (rente +1.5%)"}
        ]
        
        base_scenario_revenue = recommended_revenue
        
        for scenario in scenario_configs:
            # Adjust features with new interest rate
            scenario_features = features.copy()
            scenario_features["interest_rate"] = scenario["interest"]
            
            # Find optimal price for this scenario
            scenario_best_score = -1
            scenario_best_price = base_price
            scenario_best_prob = base_prob
            
            for multiplier in np.arange(0.85, 1.16, 0.01):
                test_price = base_price * multiplier
                prob = predict_sale_probability(scenario_features, test_price)
                score = test_price * prob
                
                if score > scenario_best_score:
                    scenario_best_score = score
                    scenario_best_price = test_price
                    scenario_best_prob = prob
            
            scenario_revenue = scenario_best_price * scenario_best_prob
            value_impact = ((scenario_revenue - base_scenario_revenue) / base_scenario_revenue) * 100
            
            scenarios.append(ScenarioAnalysis(
                scenario_name=scenario["label"],
                interest_rate=round(scenario["interest"], 4),
                recommended_price=round(scenario_best_price, 2),
                sale_probability=round(scenario_best_prob, 4),
                expected_revenue=round(scenario_revenue, 2),
                value_impact=round(value_impact, 2)
            ))
        
        # 5. Market Insights
        # Market liquidity assessment
        if liquidity >= 0.7:
            market_liquidity = "Hoog - Actieve markt met veel kopers"
        elif liquidity >= 0.5:
            market_liquidity = "Gemiddeld - Normale marktactiviteit"
        else:
            market_liquidity = "Laag - Beperkte marktliquiditeit"
        
        # Market trend
        cap_rate = features["cap_rate_market"]
        if cap_rate < 0.05 and liquidity > 0.6:
            market_trend = "Bullish - Sterke vraag, lage cap rates"
        elif cap_rate > 0.07 or liquidity < 0.4:
            market_trend = "Bearish - Zwakke vraag, hoge cap rates"
        else:
            market_trend = "Neutral - Gebalanceerde markt"
        
        # Comparable yield
        comparable_yield = features["cap_rate_market"]
        
        return PriceRecommendationResponse(
            # Core pricing
            base_price=round(base_price, 2),
            base_sale_probability=round(base_prob, 4),
            recommended_price=round(best_price, 2),
            recommended_sale_probability=round(best_prob, 4),
            expected_uplift=round(expected_uplift, 2),
            
            # Price sensitivity
            price_range_up=round(price_range_up, 1),
            price_range_down=round(price_range_down, 1),
            price_sensitivity=price_sensitivity,
            
            # Timing
            optimal_timing=optimal_timing,
            timing_reason=timing_reason,
            timing_impact=round(timing_impact, 1),
            
            # Buyer demand
            buyer_demand_score=round(buyer_demand_score, 1),
            buyer_demand_level=buyer_demand_level,
            buyer_profiles=buyer_profiles,
            
            # Scenarios
            scenarios=scenarios,
            
            # Market insights
            market_liquidity=market_liquidity,
            market_trend=market_trend,
            comparable_yield=round(comparable_yield, 4)
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Price recommendation failed: {str(e)}"
        )
