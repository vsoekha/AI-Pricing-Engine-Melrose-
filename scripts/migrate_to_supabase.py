"""
Script om data van SQLite naar Supabase te migreren.
"""

import sys
from pathlib import Path
import os
from dotenv import load_dotenv
import sqlite3
import json

# Load environment variables
load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from database_supabase import (
        add_customer, import_properties_from_csv,
        save_demo_case, get_demo_cases as get_sqlite_demo_cases
    )
except ImportError:
    print("Error: Required modules not found")
    sys.exit(1)


def migrate_from_sqlite():
    """Migrate data from SQLite to Supabase."""
    sqlite_db = Path(__file__).parent.parent / "data" / "customer_data.db"
    
    if not sqlite_db.exists():
        print(f"SQLite database not found at {sqlite_db}")
        print("Nothing to migrate.")
        return
    
    print("="*60)
    print("MIGRATING SQLITE → SUPABASE")
    print("="*60)
    
    conn = sqlite3.connect(sqlite_db)
    cursor = conn.cursor()
    
    # Migrate customers
    print("\n1. Migrating customers...")
    cursor.execute("SELECT id, customer_name, contact_email, contact_phone FROM customers")
    customers = cursor.fetchall()
    
    customer_id_map = {}  # Map old ID to new ID
    for old_id, name, email, phone in customers:
        try:
            new_id = add_customer(name, email, phone)
            customer_id_map[old_id] = new_id
            print(f"  ✓ Migrated customer: {name} ({old_id} → {new_id})")
        except Exception as e:
            print(f"  ✗ Error migrating customer {name}: {e}")
    
    # Migrate properties (if any)
    print("\n2. Migrating properties...")
    cursor.execute("SELECT DISTINCT customer_id FROM properties")
    property_customers = cursor.fetchall()
    
    for (old_customer_id,) in property_customers:
        if old_customer_id not in customer_id_map:
            print(f"  ⚠ Skipping properties for unknown customer {old_customer_id}")
            continue
        
        new_customer_id = customer_id_map[old_customer_id]
        
        # Export to CSV first, then import
        cursor.execute("""
            SELECT asset_type, city, size_m2, quality_score,
                   noi_annual, cap_rate_market, interest_rate, liquidity_index,
                   list_price, comp_median_price, sold_within_180d
            FROM properties
            WHERE customer_id = ?
        """, (old_customer_id,))
        
        import pandas as pd
        df = pd.DataFrame(cursor.fetchall(), columns=[
            'asset_type', 'city', 'size_m2', 'quality_score',
            'noi_annual', 'cap_rate_market', 'interest_rate', 'liquidity_index',
            'list_price', 'comp_median_price', 'sold_within_180d'
        ])
        
        temp_csv = Path(__file__).parent.parent / "data" / f"temp_migration_{old_customer_id}.csv"
        df.to_csv(temp_csv, index=False, sep=';')
        
        try:
            count = import_properties_from_csv(new_customer_id, str(temp_csv))
            print(f"  ✓ Migrated {count} properties for customer {new_customer_id}")
            temp_csv.unlink()  # Clean up
        except Exception as e:
            print(f"  ✗ Error migrating properties: {e}")
    
    # Migrate demo cases
    print("\n3. Migrating demo cases...")
    cursor.execute("""
        SELECT case_name, case_type, category, case_data, description
        FROM demo_cases
    """)
    demo_cases = cursor.fetchall()
    
    for case_name, case_type, category, case_data_json, description in demo_cases:
        try:
            case_data = json.loads(case_data_json)
            case_id = save_demo_case(case_name, case_type, category, case_data, description)
            if case_id:
                print(f"  ✓ Migrated demo case: {case_name}")
        except Exception as e:
            print(f"  ✗ Error migrating demo case {case_name}: {e}")
    
    conn.close()
    
    print("\n" + "="*60)
    print("MIGRATION COMPLETE!")
    print("="*60)
    print(f"\nMigrated:")
    print(f"  - {len(customer_id_map)} customers")
    print(f"  - {len(demo_cases)} demo cases")
    print("\nNote: Original SQLite database is unchanged.")


if __name__ == "__main__":
    if not os.getenv("SUPABASE_URL") or not os.getenv("SUPABASE_KEY"):
        print("Error: SUPABASE_URL and SUPABASE_KEY must be set in .env file")
        sys.exit(1)
    
    migrate_from_sqlite()




