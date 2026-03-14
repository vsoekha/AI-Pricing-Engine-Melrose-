"""
Clean AI Demo API for Real Estate Price Recommendations
Single endpoint that provides AI-powered price optimization.
"""

from pathlib import Path
import json
import csv
import urllib.request
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Optional

# ============================================================================
# Configuration
# ============================================================================

APP_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = APP_ROOT / "models" / "lightgbm_calibrated.joblib"
FEATURE_INFO_PATH = APP_ROOT / "models" / "feature_info.json"

# ECB Data Portal configuration for live interest rate (main refinancing operations rate)
ECB_API_BASE = "https://data-api.ecb.europa.eu/service/data"
# Dataset: FM (Financial market data)
# Series key: D.U2.EUR.4F.KR.MRR_FR.LEV  -> Main refinancing operations - fixed rate tenders (level, daily, euro area)
ECB_MAIN_REFI_FLOW = "FM"
ECB_MAIN_REFI_SERIES_KEY = "D.U2.EUR.4F.KR.MRR_FR.LEV"

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
    """Property features for price recommendation (Category 1: Existing Assets)."""
    asset_type: str = Field(..., description="Property type (e.g., 'logistics', 'office', 'resi')")
    city: str = Field(..., description="City name")
    size_m2: float = Field(..., gt=0, description="Property size in square meters")
    quality_score: float = Field(..., ge=0, le=1, description="Quality score (0-1)")
    noi_annual: float = Field(..., ge=0, description="Net Operating Income (annual)")
    cap_rate_market: float = Field(..., ge=0, le=1, description="Market cap rate (e.g., 0.065)")
    interest_rate: float = Field(..., ge=0, le=100, description="Interest rate: decimal (0.025) or percent (2.5)")

    @field_validator("interest_rate", mode="before")
    @classmethod
    def normalize_interest_rate(cls, v):
        """ECB geeft percentage (2.15); API verwacht decimaal (0.0215)."""
        if isinstance(v, (int, float)) and v > 1:
            return v / 100.0
        return v

    liquidity_index: float = Field(..., ge=0, le=1, description="Liquidity index (0-1)")
    list_price: float = Field(..., gt=0, description="Current listing price")
    comp_median_price: float = Field(..., gt=0, description="Comparable median price")

class DevelopmentProjectRequest(BaseModel):
    """Development project inputs (Category 2: Project Development)."""
    location: str = Field(..., description="City/location name")
    project_type: str = Field(..., description="Project type (e.g., 'residential', 'mixed-use', 'commercial')")
    units_count: int = Field(..., gt=0, description="Number of units")
    total_area_m2: float = Field(..., gt=0, description="Total development area in square meters")
    expected_sale_price_per_m2: float = Field(..., gt=0, description="Expected sale price per m²")
    build_cost_per_m2: float = Field(..., gt=0, description="Build cost per m²")
    soft_cost_pct: float = Field(..., ge=0, le=1, description="Soft costs as percentage (e.g., 0.12 = 12%)")
    contingency_pct: float = Field(..., ge=0, le=1, description="Contingency as percentage (e.g., 0.07 = 7%)")
    land_cost: float = Field(..., ge=0, description="Land acquisition cost")
    duration_months: int = Field(..., gt=0, description="Project duration in months")
    interest_rate: float = Field(..., ge=0, le=1, description="Financing interest rate (e.g., 0.045 = 4.5%)")
    target_margin_pct: float = Field(..., ge=0, le=1, description="Target profit margin (e.g., 0.18 = 18%)")
    exit_cap_rate: Optional[float] = Field(None, ge=0, le=1, description="Exit cap rate (optional)")
    absorption_months: Optional[int] = Field(None, gt=0, description="Expected absorption period in months (optional)")

class ScenarioAnalysis(BaseModel):
    """Scenario analysis for different interest rate environments."""
    scenario_name: str = Field(..., description="Scenario name (base/optimistic/downside/stress)")
    interest_rate: float = Field(..., description="Interest rate for this scenario")
    recommended_price: float = Field(..., description="Optimal price in this scenario")
    sale_probability: float = Field(..., description="Sale probability in this scenario")
    expected_revenue: float = Field(..., description="Expected revenue (price × probability)")
    value_impact: float = Field(..., description="Value impact vs base case (%)")

class DevelopmentScenario(BaseModel):
    """Development project scenario analysis."""
    scenario_name: str = Field(..., description="Scenario name (base/downside/upside)")
    total_revenue: float = Field(..., description="Total project revenue")
    total_project_cost: float = Field(..., description="Total project cost")
    profit: float = Field(..., description="Project profit")
    margin: float = Field(..., description="Profit margin (0-1)")
    meets_target_margin: bool = Field(..., description="Whether target margin is met")

class DevelopmentProjectResponse(BaseModel):
    """Development project analysis response."""
    project_summary: Dict[str, Any] = Field(..., description="Project summary information")
    base_scenario: DevelopmentScenario = Field(..., description="Base case scenario")
    downside_scenario: DevelopmentScenario = Field(..., description="Downside scenario")
    upside_scenario: DevelopmentScenario = Field(..., description="Upside scenario")
    recommended_action: str = Field(..., description="Recommended action: PROCEED, REVIEW, REPRICE, or WAIT")
    recommendation_reason: str = Field(..., description="Reason for recommendation")

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


def fetch_ecb_main_refi_rate() -> Optional[Dict[str, Any]]:
    """
    Fetch the latest ECB main refinancing operations rate from the ECB Data Portal.

    Uses the SDMX 2.1 REST API with CSV output for simple parsing.
    Docs: https://data.ecb.europa.eu/help/api/data
    Example dataset: https://data.ecb.europa.eu/data/datasets/FM/FM.D.U2.EUR.4F.KR.MRR_FR.LEV
    """
    series_path = f"{ECB_MAIN_REFI_FLOW}/{ECB_MAIN_REFI_SERIES_KEY}"
    # lastNObservations=1 -> only latest value
    # detail=dataonly -> exclude attributes
    # format=csvdata -> easier to parse than SDMX-JSON
    url = f"{ECB_API_BASE}/{series_path}?lastNObservations=1&detail=dataonly&format=csvdata"

    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status != 200:
                raise RuntimeError(f"ECB API returned status {resp.status}")
            raw = resp.read().decode("utf-8").splitlines()

        reader = csv.DictReader(raw)
        last_row = next(iter(reader), None)
        if not last_row:
            raise RuntimeError("ECB response bevat geen data")

        value_str = last_row.get("OBS_VALUE")
        time_str = last_row.get("TIME_PERIOD")
        if value_str is None:
            raise RuntimeError("ECB response mist OBS_VALUE kolom")

        rate = float(value_str)
        # ECB geeft percentage (bijv. 2.15). Voor API: rate_decimal = 0.0215
        rate_decimal = rate / 100.0 if rate > 1 else rate

        return {
            "source": "ECB",
            "series_key": ECB_MAIN_REFI_SERIES_KEY,
            "rate": rate,
            "rate_percent": rate,
            "rate_decimal": rate_decimal,
            "date": time_str,
            "api_url": url,
        }
    except Exception as e:
        # Log to stdout for debugging, but fail gracefully
        print(f"ERROR: Failed to fetch ECB main refi rate: {e}")
        return None


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


def calculate_development_scenario(
    total_area_m2: float,
    sale_price_per_m2: float,
    build_cost_per_m2: float,
    soft_cost_pct: float,
    contingency_pct: float,
    land_cost: float,
    duration_months: int,
    interest_rate: float,
    target_margin_pct: float
) -> Dict[str, float]:
    """
    Calculate development project financials for a given scenario.
    
    Returns:
        Dictionary with total_revenue, total_project_cost, profit, margin, meets_target_margin
    """
    # Revenue
    total_revenue = total_area_m2 * sale_price_per_m2
    
    # Costs
    hard_cost = total_area_m2 * build_cost_per_m2
    soft_cost = hard_cost * soft_cost_pct
    contingency = hard_cost * contingency_pct
    total_cost_before_finance = hard_cost + soft_cost + contingency + land_cost
    
    # Financing
    financing_cost = total_cost_before_finance * interest_rate * (duration_months / 12)
    total_project_cost = total_cost_before_finance + financing_cost
    
    # Profitability
    profit = total_revenue - total_project_cost
    margin = profit / total_revenue if total_revenue > 0 else 0.0
    meets_target_margin = margin >= target_margin_pct
    
    return {
        "total_revenue": total_revenue,
        "total_project_cost": total_project_cost,
        "profit": profit,
        "margin": margin,
        "meets_target_margin": meets_target_margin
    }

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/")
def root():
    """Root endpoint with API information."""
    return {
        "name": "Vastgoed AI Analyse API",
        "version": "2.0.0",
        "status": "operationeel" if MODEL_LOADED else "model_niet_geladen",
        "categories": {
            "category_1": {
                "name": "Bestaand Vastgoed",
                "endpoint": "/recommend_price",
                "method": "POST",
                "description": "AI-gestuurde prijsoptimalisatie voor bestaande vastgoedobjecten"
            },
            "category_2": {
                "name": "Projectontwikkeling",
                "endpoint": "/dev_project_analysis",
                "method": "POST",
                "description": "Financiële analyse voor nieuwe ontwikkelingsprojecten"
            }
        },
        "demo_cases": "/demo_cases",
        "docs": "/docs",
        "live_data": {
            "ecb_main_refi_rate": "/ecb/main_refi_rate"
        }
    }


@app.get("/ecb/main_refi_rate")
def get_ecb_main_refi_rate():
    """
    Haal de actuele ECB main refinancing operations rente op (live datastroom).

    Bron: ECB Data Portal (dataset FM, serie D.U2.EUR.4F.KR.MRR_FR.LEV)
    Documentatie: https://data.ecb.europa.eu/help/api/data
    """
    data = fetch_ecb_main_refi_rate()
    if data is None:
        raise HTTPException(
            status_code=503,
            detail="Kon actuele ECB-rente niet ophalen van de ECB Data Portal."
        )
    return data

@app.get("/health")
def health():
    """Health check endpoint."""
    import os
    response = {
        "status": "gezond" if MODEL_LOADED else "ongezond",
        "model_geladen": MODEL_LOADED,
        "aantal_features": len(ALL_FEATURES) if MODEL_LOADED else 0
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
def recommend_price(request: PropertyRequest, gebruik_ecb_rente: bool = False):
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

        # Optioneel: gebruik live ECB-rente i.p.v. handmatige invoer
        # ECB geeft percentage (2.15); ons model verwacht decimaal (0.0215)
        if gebruik_ecb_rente:
            ecb_data = fetch_ecb_main_refi_rate()
            if ecb_data and "rate" in ecb_data:
                r = float(ecb_data["rate"])
                features["interest_rate"] = r / 100.0 if r > 1 else r
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
        
        # Price sensitivity level (Nederlands)
        total_range = price_range_up + price_range_down
        if total_range > 20:
            price_sensitivity = "laag"
        elif total_range > 10:
            price_sensitivity = "gemiddeld"
        else:
            price_sensitivity = "hoog"
        
        # 2. Timing Recommendation
        # Based on interest rate, liquidity, and market conditions
        current_interest = features["interest_rate"]
        liquidity = features["liquidity_index"]
        
        # Calculate optimal timing (simplified logic for demo) - Nederlands
        if current_interest < 0.03 and liquidity > 0.7:
            # Goede omstandigheden - eerder verkopen
            optimal_timing = "Q1 2026"
            timing_reason = "Gunstige marktomstandigheden: lage rente en hoge liquiditeit. Ideaal moment om te verkopen voor maximale opbrengst."
            timing_impact = 8.5
        elif current_interest > 0.04 or liquidity < 0.5:
            # Uitdagende omstandigheden - wachten
            optimal_timing = "Q3 2027"
            timing_reason = "Wachten op betere marktomstandigheden: rente of liquiditeit moeten verbeteren. Verkoop nu levert lagere opbrengst op."
            timing_impact = 11.2
        else:
            # Neutraal - mid-term
            optimal_timing = "Q2 2026"
            timing_reason = "Gebalanceerde marktomstandigheden, optimale timing mid-term. Goed moment voor verkoop met redelijke opbrengst."
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
            buyer_demand_level = "hoog"
        elif buyer_demand_score >= 4:
            buyer_demand_level = "gemiddeld"
        else:
            buyer_demand_level = "laag"
        
        # Buyer profiles based on asset type (Nederlandse terminologie)
        buyer_profiles_map = {
            "logistics": ["Pensioenfondsen", "Institutionele investeerders", "Private equity fondsen"],
            "office": ["Pensioenfondsen", "Family offices", "Internationale investeerders"],
            "resi": ["Family offices", "Private equity", "High-net-worth individuen"],
            "retail": ["Retail investeerders", "Vastgoedfondsen", "Particuliere investeerders"],
            "mixed": ["Gemengde investeerders", "Vastgoedontwikkelaars", "Institutionele partijen"]
        }
        buyer_profiles = buyer_profiles_map.get(features["asset_type"], ["Gemengde investeerders"])
        
        # 4. Scenario Analysis (different interest rates)
        scenarios = []
        scenario_configs = [
            {"name": "base", "interest": current_interest, "label": "Basis Scenario"},
            {"name": "optimistic", "interest": max(0.01, current_interest - 0.005), "label": "Optimistisch (rente -0.5%)"},
            {"name": "downside", "interest": current_interest + 0.01, "label": "Negatief (rente +1.0%)"},
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
            detail=f"Prijsaanbeveling mislukt: {str(e)}"
        )


@app.get("/dev_project_analysis")
def dev_project_analysis_info():
    """GET endpoint - Toont hoe je POST /dev_project_analysis gebruikt."""
    return {
        "message": "Dit endpoint vereist een POST request met JSON data",
        "method": "POST",
        "content_type": "application/json",
        "categorie": "Projectontwikkeling",
        "beschrijving": "Financiële analyse voor nieuwe ontwikkelingsprojecten met scenario modellering",
        "example_request": {
            "location": "Amsterdam",
            "project_type": "residential",
            "units_count": 50,
            "total_area_m2": 5000,
            "expected_sale_price_per_m2": 6000,
            "build_cost_per_m2": 3500,
            "soft_cost_pct": 0.12,
            "contingency_pct": 0.07,
            "land_cost": 5000000,
            "duration_months": 24,
            "interest_rate": 0.045,
            "target_margin_pct": 0.18
        },
        "interactive_docs": "/docs"
    }


@app.post("/dev_project_analysis", response_model=DevelopmentProjectResponse)
def dev_project_analysis(request: DevelopmentProjectRequest):
    """
    Analyze development project financials with scenario modeling.
    
    Category 2: Development Projects
    Performs deterministic financial calculations for new build/development projects.
    Returns base, downside, and upside scenarios with recommendations.
    """
    try:
        # Extract inputs
        base_sale_price = request.expected_sale_price_per_m2
        base_build_cost = request.build_cost_per_m2
        base_interest = request.interest_rate
        
        # Calculate base scenario
        base_result = calculate_development_scenario(
            total_area_m2=request.total_area_m2,
            sale_price_per_m2=base_sale_price,
            build_cost_per_m2=base_build_cost,
            soft_cost_pct=request.soft_cost_pct,
            contingency_pct=request.contingency_pct,
            land_cost=request.land_cost,
            duration_months=request.duration_months,
            interest_rate=base_interest,
            target_margin_pct=request.target_margin_pct
        )
        
        # Calculate downside scenario
        downside_sale_price = base_sale_price * 0.93  # -7%
        downside_build_cost = base_build_cost * 1.07   # +7%
        downside_interest = base_interest + 0.005      # +0.5%
        
        downside_result = calculate_development_scenario(
            total_area_m2=request.total_area_m2,
            sale_price_per_m2=downside_sale_price,
            build_cost_per_m2=downside_build_cost,
            soft_cost_pct=request.soft_cost_pct,
            contingency_pct=request.contingency_pct,
            land_cost=request.land_cost,
            duration_months=request.duration_months,
            interest_rate=downside_interest,
            target_margin_pct=request.target_margin_pct
        )
        
        # Calculate upside scenario
        upside_sale_price = base_sale_price * 1.05    # +5%
        upside_build_cost = base_build_cost * 0.97    # -3%
        upside_interest = max(0.001, base_interest - 0.005)  # -0.5% (min 0.1%)
        
        upside_result = calculate_development_scenario(
            total_area_m2=request.total_area_m2,
            sale_price_per_m2=upside_sale_price,
            build_cost_per_m2=upside_build_cost,
            soft_cost_pct=request.soft_cost_pct,
            contingency_pct=request.contingency_pct,
            land_cost=request.land_cost,
            duration_months=request.duration_months,
            interest_rate=upside_interest,
            target_margin_pct=request.target_margin_pct
        )
        
        # Determine recommended action
        base_meets_target = base_result["meets_target_margin"]
        downside_margin_acceptable = downside_result["margin"] >= (request.target_margin_pct * 0.6)
        
        # Recommendation logic (Nederlands)
        if base_meets_target and downside_margin_acceptable:
            recommended_action = "DOORGAAN"
            recommendation_reason = "Basis scenario voldoet aan doelmarge en negatief scenario blijft acceptabel (≥60% van doel). Project is financieel gezond en kan worden gestart."
        elif base_meets_target and not downside_margin_acceptable:
            recommended_action = "HERZIEN"
            recommendation_reason = "Basis scenario is winstgevend maar negatief scenario is risicovol. Herzie aannames en overweeg risicobeperkende maatregelen."
        elif not base_meets_target and base_result["margin"] > 0:
            recommended_action = "HERPRIJZEN"
            recommendation_reason = "Project is winstgevend maar onder doelmarge. Overweeg herprijzing, kostenoptimalisatie of aanpassing van projectscope."
        else:
            recommended_action = "WACHTEN"
            recommendation_reason = "Project is niet winstgevend in basis scenario. Wacht op betere marktomstandigheden of herzie het projectconcept."
        
        # Project summary
        project_summary = {
            "location": request.location,
            "project_type": request.project_type,
            "units_count": request.units_count,
            "total_area_m2": request.total_area_m2,
            "duration_months": request.duration_months,
            "target_margin_pct": request.target_margin_pct
        }
        
        return DevelopmentProjectResponse(
            project_summary=project_summary,
            base_scenario=DevelopmentScenario(
                scenario_name="Basis Scenario",
                total_revenue=round(base_result["total_revenue"], 2),
                total_project_cost=round(base_result["total_project_cost"], 2),
                profit=round(base_result["profit"], 2),
                margin=round(base_result["margin"], 4),
                meets_target_margin=base_result["meets_target_margin"]
            ),
            downside_scenario=DevelopmentScenario(
                scenario_name="Negatief Scenario (-7% verkoopprijs, +7% bouwkosten, +0.5% rente)",
                total_revenue=round(downside_result["total_revenue"], 2),
                total_project_cost=round(downside_result["total_project_cost"], 2),
                profit=round(downside_result["profit"], 2),
                margin=round(downside_result["margin"], 4),
                meets_target_margin=downside_result["meets_target_margin"]
            ),
            upside_scenario=DevelopmentScenario(
                scenario_name="Positief Scenario (+5% verkoopprijs, -3% bouwkosten, -0.5% rente)",
                total_revenue=round(upside_result["total_revenue"], 2),
                total_project_cost=round(upside_result["total_project_cost"], 2),
                profit=round(upside_result["profit"], 2),
                margin=round(upside_result["margin"], 4),
                meets_target_margin=upside_result["meets_target_margin"]
            ),
            recommended_action=recommended_action,
            recommendation_reason=recommendation_reason
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Projectanalyse mislukt: {str(e)}"
        )


# ============================================================================
# Demo Cases Endpoints
# ============================================================================

@app.get("/demo_cases")
def list_demo_cases(category: Optional[str] = None):
    """Lijst alle demo cases op, optioneel gefilterd op category."""
    try:
        sys.path.insert(0, str(APP_ROOT))
        # Try Supabase first, fallback to SQLite
        try:
            from database_supabase import get_demo_cases
        except (ImportError, ValueError):
            from database_setup import get_demo_cases
        
        cases = get_demo_cases(category=category)
        return {
            "demo_cases": cases,
            "total": len(cases),
            "category_filter": category
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fout bij ophalen demo cases: {str(e)}"
        )


@app.get("/demo_cases/{case_id}")
def get_demo_case(case_id: int):
    """Haal een specifieke demo case op."""
    try:
        sys.path.insert(0, str(APP_ROOT))
        # Try Supabase first, fallback to SQLite
        try:
            from database_supabase import get_demo_case as db_get_case
        except (ImportError, ValueError):
            from database_setup import get_demo_case as db_get_case
        
        case = db_get_case(case_id)
        if not case:
            raise HTTPException(status_code=404, detail=f"Demo case {case_id} niet gevonden")
        return case
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fout bij ophalen demo case: {str(e)}"
        )


class DemoCaseRequest(BaseModel):
    """Request model for creating a demo case."""
    case_name: str = Field(..., description="Naam van de demo case")
    case_type: str = Field(..., description="Type: 'existing_asset' of 'development_project'")
    category: str = Field(..., description="Categorie: 'category_1' of 'category_2'")
    case_data: Dict[str, Any] = Field(..., description="Case data (volledige request body)")
    description: Optional[str] = Field(None, description="Beschrijving van de case")


@app.post("/demo_cases")
def create_demo_case(request: DemoCaseRequest):
    """Sla een nieuwe demo case op."""
    try:
        sys.path.insert(0, str(APP_ROOT))
        # Try Supabase first, fallback to SQLite
        try:
            from database_supabase import save_demo_case
        except (ImportError, ValueError):
            from database_setup import save_demo_case
        
        case_id = save_demo_case(
            request.case_name,
            request.case_type,
            request.category,
            request.case_data,
            request.description
        )
        if case_id:
            return {
                "success": True,
                "case_id": case_id,
                "message": f"Demo case '{request.case_name}' opgeslagen"
            }
        else:
            raise HTTPException(status_code=400, detail="Fout bij opslaan demo case")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fout bij opslaan demo case: {str(e)}"
        )


@app.delete("/demo_cases/{case_id}")
def delete_demo_case(case_id: int):
    """Verwijder een demo case."""
    try:
        sys.path.insert(0, str(APP_ROOT))
        # Try Supabase first, fallback to SQLite
        try:
            from database_supabase import delete_demo_case as db_delete_case
        except (ImportError, ValueError):
            from database_setup import delete_demo_case as db_delete_case
        
        success = db_delete_case(case_id)
        if success:
            return {"success": True, "message": f"Demo case {case_id} verwijderd"}
        else:
            raise HTTPException(status_code=404, detail=f"Demo case {case_id} niet gevonden")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Fout bij verwijderen demo case: {str(e)}"
        )
