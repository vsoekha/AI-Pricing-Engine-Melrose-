"""
Script om Supabase database schema aan te maken.
"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from supabase import create_client
    from database_supabase import get_supabase_client
except ImportError:
    print("Error: supabase-py not installed. Run: pip install supabase")
    sys.exit(1)


def setup_schema():
    """Run SQL schema in Supabase."""
    schema_path = Path(__file__).parent.parent / "supabase" / "schema.sql"
    
    if not schema_path.exists():
        print(f"Error: Schema file not found at {schema_path}")
        return False
    
    print("="*60)
    print("SUPABASE DATABASE SETUP")
    print("="*60)
    
    print("\n⚠️  MANUAL STEP REQUIRED:")
    print("1. Open Supabase Dashboard → SQL Editor")
    print("2. Copy the contents of supabase/schema.sql")
    print("3. Paste and run in SQL Editor")
    print(f"\nSchema file location: {schema_path}")
    
    # Read and display schema
    with open(schema_path, 'r') as f:
        schema = f.read()
        print("\n" + "="*60)
        print("SCHEMA PREVIEW (first 500 chars):")
        print("="*60)
        print(schema[:500] + "...")
    
    return True


def test_connection():
    """Test Supabase connection."""
    print("\n" + "="*60)
    print("TESTING SUPABASE CONNECTION")
    print("="*60)
    
    try:
        supabase = get_supabase_client()
        
        # Test query
        result = supabase.table("customers").select("count").execute()
        print("✓ Connection successful!")
        return True
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        print("\nCheck:")
        print("1. SUPABASE_URL is set in .env")
        print("2. SUPABASE_KEY is set in .env")
        print("3. Schema is created (run SQL script)")
        return False


if __name__ == "__main__":
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        print("\nCreate .env file with:")
        print("SUPABASE_URL=https://xxxxx.supabase.co")
        print("SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")
        sys.exit(1)
    
    setup_schema()
    
    print("\n" + "="*60)
    print("After running the SQL schema, test connection:")
    print("python scripts/test_supabase_connection.py")
    print("="*60)




