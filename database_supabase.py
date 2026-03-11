"""
Supabase database module voor AI Pricing Engine.
Vervangt SQLite met Supabase (PostgreSQL).
"""

import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
import pandas as pd

try:
    from supabase import create_client, Client
    from postgrest.exceptions import APIError
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: supabase-py not installed. Install with: pip install supabase")

# Environment variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # service_role key
DATABASE_URL = os.getenv("DATABASE_URL")  # PostgreSQL connection string

# Initialize Supabase client
_supabase_client: Optional[Client] = None


def get_supabase_client() -> Client:
    """Get or create Supabase client."""
    global _supabase_client
    
    if not SUPABASE_AVAILABLE:
        raise ImportError("supabase-py is not installed. Run: pip install supabase")
    
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError(
                "Supabase credentials not found. Set SUPABASE_URL and SUPABASE_KEY environment variables."
            )
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    return _supabase_client


# ============================================================================
# Customers
# ============================================================================

def add_customer(name: str, email: Optional[str] = None, phone: Optional[str] = None) -> int:
    """Add a new customer to the database."""
    supabase = get_supabase_client()
    
    try:
        result = supabase.table("customers").insert({
            "customer_name": name,
            "contact_email": email,
            "contact_phone": phone
        }).execute()
        
        customer_id = result.data[0]["id"]
        print(f"Customer '{name}' added with ID: {customer_id}")
        return customer_id
    except APIError as e:
        if "duplicate key" in str(e).lower() or "unique" in str(e).lower():
            # Customer already exists, fetch it
            result = supabase.table("customers").select("id").eq("customer_name", name).execute()
            if result.data:
                customer_id = result.data[0]["id"]
                print(f"Customer '{name}' already exists with ID: {customer_id}")
                return customer_id
        raise


def get_customer(customer_id: int) -> Optional[Dict]:
    """Get customer by ID."""
    supabase = get_supabase_client()
    result = supabase.table("customers").select("*").eq("id", customer_id).execute()
    return result.data[0] if result.data else None


# ============================================================================
# Properties
# ============================================================================

def import_properties_from_csv(customer_id: int, csv_path: str) -> int:
    """Import properties from CSV file for a customer."""
    df = pd.read_csv(csv_path, sep=";")
    
    # Ensure required columns exist
    required_cols = ['sold_within_180d']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    supabase = get_supabase_client()
    
    properties = []
    for _, row in df.iterrows():
        properties.append({
            "customer_id": customer_id,
            "asset_type": row.get('asset_type'),
            "city": row.get('city'),
            "size_m2": row.get('size_m2'),
            "quality_score": row.get('quality_score'),
            "noi_annual": row.get('noi_annual'),
            "cap_rate_market": row.get('cap_rate_market'),
            "interest_rate": row.get('interest_rate'),
            "liquidity_index": row.get('liquidity_index'),
            "list_price": row.get('list_price'),
            "comp_median_price": row.get('comp_median_price'),
            "sold_within_180d": int(row.get('sold_within_180d', 0))
        })
    
    # Insert in batches
    imported = 0
    batch_size = 100
    for i in range(0, len(properties), batch_size):
        batch = properties[i:i + batch_size]
        try:
            supabase.table("properties").insert(batch).execute()
            imported += len(batch)
        except Exception as e:
            print(f"Error importing batch: {e}")
    
    print(f"Imported {imported} properties for customer {customer_id}")
    return imported


def export_customer_data(customer_id: int, output_path: str) -> pd.DataFrame:
    """Export customer data to CSV for training."""
    supabase = get_supabase_client()
    
    result = supabase.table("properties").select(
        "asset_type, city, size_m2, quality_score, "
        "noi_annual, cap_rate_market, interest_rate, liquidity_index, "
        "list_price, comp_median_price, sold_within_180d"
    ).eq("customer_id", customer_id).execute()
    
    df = pd.DataFrame(result.data)
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} properties to {output_path}")
    return df


def get_customer_stats(customer_id: int) -> Dict:
    """Get statistics about customer's data."""
    supabase = get_supabase_client()
    
    # Total properties
    total_result = supabase.table("properties").select("id", count="exact").eq("customer_id", customer_id).execute()
    total = total_result.count if hasattr(total_result, 'count') else len(total_result.data)
    
    # Sold vs not sold
    all_properties = supabase.table("properties").select("sold_within_180d").eq("customer_id", customer_id).execute()
    sold = sum(1 for p in all_properties.data if p.get("sold_within_180d") == 1)
    not_sold = total - sold
    
    # Asset types
    asset_types_result = supabase.table("properties").select("asset_type").eq("customer_id", customer_id).execute()
    asset_types = {}
    for prop in asset_types_result.data:
        asset_type = prop.get("asset_type")
        if asset_type:
            asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
    
    return {
        "total_properties": total,
        "sold": sold,
        "not_sold": not_sold,
        "asset_types": asset_types,
        "ready_for_training": total >= 50
    }


# ============================================================================
# Demo Cases
# ============================================================================

def save_demo_case(
    case_name: str,
    case_type: str,
    category: str,
    case_data: dict,
    description: Optional[str] = None
) -> Optional[int]:
    """Save a demo case to the database."""
    supabase = get_supabase_client()
    
    try:
        result = supabase.table("demo_cases").insert({
            "case_name": case_name,
            "case_type": case_type,
            "category": category,
            "case_data": case_data,  # Supabase handles JSONB automatically
            "description": description
        }).execute()
        
        case_id = result.data[0]["id"]
        print(f"Demo case '{case_name}' saved with ID: {case_id}")
        return case_id
    except Exception as e:
        print(f"Error saving demo case: {e}")
        return None


def get_demo_cases(category: Optional[str] = None) -> List[Dict]:
    """Get all demo cases, optionally filtered by category."""
    supabase = get_supabase_client()
    
    query = supabase.table("demo_cases").select("*")
    
    if category:
        query = query.eq("category", category)
    
    query = query.order("case_name")
    result = query.execute()
    
    cases = []
    for row in result.data:
        cases.append({
            "id": row["id"],
            "case_name": row["case_name"],
            "case_type": row["case_type"],
            "category": row["category"],
            "case_data": row["case_data"],  # Already a dict (JSONB)
            "description": row.get("description"),
            "created_at": row.get("created_at")
        })
    
    return cases


def get_demo_case(case_id: int) -> Optional[Dict]:
    """Get a specific demo case by ID."""
    supabase = get_supabase_client()
    
    result = supabase.table("demo_cases").select("*").eq("id", case_id).execute()
    
    if result.data:
        row = result.data[0]
        return {
            "id": row["id"],
            "case_name": row["case_name"],
            "case_type": row["case_type"],
            "category": row["category"],
            "case_data": row["case_data"],
            "description": row.get("description"),
            "created_at": row.get("created_at")
        }
    return None


def delete_demo_case(case_id: int) -> bool:
    """Delete a demo case."""
    supabase = get_supabase_client()
    
    result = supabase.table("demo_cases").delete().eq("id", case_id).execute()
    deleted = len(result.data) > 0
    
    print(f"Deleted {1 if deleted else 0} demo case(s)")
    return deleted


# ============================================================================
# API Usage Tracking
# ============================================================================

def log_api_usage(
    customer_id: Optional[int],
    endpoint: str,
    response_time_ms: int,
    success: bool
):
    """Log API usage for analytics."""
    supabase = get_supabase_client()
    
    try:
        supabase.table("api_usage").insert({
            "customer_id": customer_id,
            "endpoint": endpoint,
            "response_time_ms": response_time_ms,
            "success": success
        }).execute()
    except Exception as e:
        print(f"Error logging API usage: {e}")




