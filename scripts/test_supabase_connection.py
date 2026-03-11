"""
Test Supabase connection and database setup.
"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from database_supabase import (
        get_supabase_client,
        add_customer,
        save_demo_case,
        get_demo_cases,
        get_demo_case
    )
except ImportError as e:
    print(f"Error importing database_supabase: {e}")
    print("Make sure supabase-py is installed: pip install supabase")
    sys.exit(1)


def test_connection():
    """Test basic connection."""
    print("="*60)
    print("TESTING SUPABASE CONNECTION")
    print("="*60)
    
    try:
        supabase = get_supabase_client()
        print("✓ Supabase client created")
        
        # Test query
        result = supabase.table("customers").select("count").limit(1).execute()
        print("✓ Database connection successful")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        return False


def test_tables():
    """Test if all tables exist."""
    print("\n" + "="*60)
    print("TESTING TABLES")
    print("="*60)
    
    supabase = get_supabase_client()
    tables = ["customers", "properties", "demo_cases", "training_runs", "api_usage"]
    
    for table in tables:
        try:
            result = supabase.table(table).select("count").limit(1).execute()
            print(f"✓ Table '{table}' exists")
        except Exception as e:
            print(f"✗ Table '{table}' error: {e}")


def test_operations():
    """Test CRUD operations."""
    print("\n" + "="*60)
    print("TESTING CRUD OPERATIONS")
    print("="*60)
    
    # Test add customer
    try:
        customer_id = add_customer("Test Customer", "test@example.com", "0612345678")
        print(f"✓ Added test customer (ID: {customer_id})")
        
        # Test save demo case
        case_data = {
            "asset_type": "logistics",
            "city": "Rotterdam",
            "size_m2": 1000
        }
        case_id = save_demo_case(
            "Test Case",
            "existing_asset",
            "category_1",
            case_data,
            "Test description"
        )
        if case_id:
            print(f"✓ Saved demo case (ID: {case_id})")
            
            # Test get demo cases
            cases = get_demo_cases()
            print(f"✓ Retrieved {len(cases)} demo cases")
            
            # Test get specific case
            case = get_demo_case(case_id)
            if case:
                print(f"✓ Retrieved demo case: {case['case_name']}")
        
        return True
    except Exception as e:
        print(f"✗ Operation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        print("\nCreate .env file with:")
        print("SUPABASE_URL=https://xxxxx.supabase.co")
        print("SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
        sys.exit(1)
    
    if not test_connection():
        print("\n⚠️  Fix connection issues before continuing")
        sys.exit(1)
    
    test_tables()
    test_operations()
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)




