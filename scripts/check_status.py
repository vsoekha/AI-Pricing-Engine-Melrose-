"""
Quick status check: Is Supabase configured and working?
"""

import sys
from pathlib import Path
import os

# Load .env if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

print("="*60)
print("DATABASE STATUS CHECK")
print("="*60)

# Check environment variables
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

if supabase_url and supabase_key:
    print("✓ Supabase credentials found in .env")
    print(f"  URL: {supabase_url[:30]}...")
    print(f"  Key: {supabase_key[:20]}...")
    
    # Try to connect
    try:
        from database_supabase import get_supabase_client
        supabase = get_supabase_client()
        
        # Test query
        result = supabase.table("demo_cases").select("count").limit(1).execute()
        print("\n✓ Supabase connection: WORKING")
        print("  → API gebruikt Supabase database")
        
        # Count demo cases
        cases = supabase.table("demo_cases").select("id").execute()
        print(f"  → Demo cases in database: {len(cases.data)}")
        
    except Exception as e:
        print(f"\n✗ Supabase connection: FAILED")
        print(f"  Error: {e}")
        print("\n  → API valt terug op SQLite (lokaal)")
else:
    print("✗ Supabase credentials NOT found in .env")
    print("  → API gebruikt SQLite (lokaal)")
    print("\n  Om Supabase te gebruiken:")
    print("  1. Maak .env bestand met:")
    print("     SUPABASE_URL=https://xxxxx.supabase.co")
    print("     SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...")

print("\n" + "="*60)
print("SERVER STARTEN:")
print("="*60)
print("cd ~/Desktop/ml-service")
print("source .venv/bin/activate")
print("export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib:$DYLD_LIBRARY_PATH")
print("uvicorn api.main:app --reload")
print("\nDan open: http://localhost:8000/docs")
print("="*60)



