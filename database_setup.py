"""
Database setup for customer data storage and model training.
This prepares the infrastructure for storing customer data and training custom models.
"""

import sqlite3
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime

# Database path
DB_PATH = Path(__file__).parent / "data" / "customer_data.db"


def init_database():
    """Initialize SQLite database for customer data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Properties table - stores all property listings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            asset_type TEXT,
            city TEXT,
            size_m2 REAL,
            quality_score REAL,
            noi_annual REAL,
            cap_rate_market REAL,
            interest_rate REAL,
            liquidity_index REAL,
            list_price REAL,
            comp_median_price REAL,
            sold_within_180d INTEGER,
            sale_date DATE,
            sale_price REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Customers table - tracks different customers/clients
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_name TEXT UNIQUE NOT NULL,
            contact_email TEXT,
            contact_phone TEXT,
            status TEXT DEFAULT 'demo',
            model_trained BOOLEAN DEFAULT 0,
            model_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Training runs table - tracks model training history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT NOT NULL,
            training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            data_count INTEGER,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            roc_auc REAL,
            model_path TEXT,
            notes TEXT
        )
    """)
    
    # API usage tracking
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id TEXT,
            endpoint TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            response_time_ms INTEGER,
            success BOOLEAN
        )
    """)
    
    # Demo cases table - voor het opslaan van demo cases
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS demo_cases (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            case_name TEXT NOT NULL,
            case_type TEXT NOT NULL,
            category TEXT NOT NULL,
            case_data TEXT NOT NULL,
            description TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DB_PATH}")


def add_customer(name: str, email: Optional[str] = None, phone: Optional[str] = None):
    """Add a new customer to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO customers (customer_name, contact_email, contact_phone)
            VALUES (?, ?, ?)
        """, (name, email, phone))
        conn.commit()
        customer_id = cursor.lastrowid
        print(f"Customer '{name}' added with ID: {customer_id}")
        return customer_id
    except sqlite3.IntegrityError:
        print(f"Customer '{name}' already exists")
        cursor.execute("SELECT id FROM customers WHERE customer_name = ?", (name,))
        return cursor.fetchone()[0]
    finally:
        conn.close()


def import_properties_from_csv(customer_id: str, csv_path: str):
    """Import properties from CSV file for a customer."""
    df = pd.read_csv(csv_path, sep=";")
    
    # Ensure required columns exist
    required_cols = ['sold_within_180d']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    imported = 0
    for _, row in df.iterrows():
        try:
            cursor.execute("""
                INSERT INTO properties (
                    customer_id, asset_type, city, size_m2, quality_score,
                    noi_annual, cap_rate_market, interest_rate, liquidity_index,
                    list_price, comp_median_price, sold_within_180d
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                customer_id,
                row.get('asset_type'),
                row.get('city'),
                row.get('size_m2'),
                row.get('quality_score'),
                row.get('noi_annual'),
                row.get('cap_rate_market'),
                row.get('interest_rate'),
                row.get('liquidity_index'),
                row.get('list_price'),
                row.get('comp_median_price'),
                row.get('sold_within_180d')
            ))
            imported += 1
        except Exception as e:
            print(f"Error importing row: {e}")
    
    conn.commit()
    conn.close()
    print(f"Imported {imported} properties for customer {customer_id}")
    return imported


def export_customer_data(customer_id: str, output_path: str):
    """Export customer data to CSV for training."""
    conn = sqlite3.connect(DB_PATH)
    
    df = pd.read_sql_query("""
        SELECT 
            asset_type, city, size_m2, quality_score,
            noi_annual, cap_rate_market, interest_rate, liquidity_index,
            list_price, comp_median_price, sold_within_180d
        FROM properties
        WHERE customer_id = ?
    """, conn, params=(customer_id,))
    
    conn.close()
    
    df.to_csv(output_path, index=False)
    print(f"Exported {len(df)} properties to {output_path}")
    return df


def get_customer_stats(customer_id: str):
    """Get statistics about customer's data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Total properties
    cursor.execute("SELECT COUNT(*) FROM properties WHERE customer_id = ?", (customer_id,))
    total = cursor.fetchone()[0]
    
    # Sold vs not sold
    cursor.execute("""
        SELECT 
            SUM(CASE WHEN sold_within_180d = 1 THEN 1 ELSE 0 END) as sold,
            SUM(CASE WHEN sold_within_180d = 0 THEN 1 ELSE 0 END) as not_sold
        FROM properties WHERE customer_id = ?
    """, (customer_id,))
    sold, not_sold = cursor.fetchone()
    
    # Asset types
    cursor.execute("""
        SELECT asset_type, COUNT(*) 
        FROM properties 
        WHERE customer_id = ? 
        GROUP BY asset_type
    """, (customer_id,))
    asset_types = dict(cursor.fetchall())
    
    conn.close()
    
    return {
        "total_properties": total,
        "sold": sold or 0,
        "not_sold": not_sold or 0,
        "asset_types": asset_types,
        "ready_for_training": total >= 50  # Minimum for training
    }


def save_demo_case(case_name: str, case_type: str, category: str, case_data: dict, description: Optional[str] = None):
    """Save a demo case to the database."""
    import json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO demo_cases (case_name, case_type, category, case_data, description)
            VALUES (?, ?, ?, ?, ?)
        """, (case_name, case_type, category, json.dumps(case_data), description))
        conn.commit()
        case_id = cursor.lastrowid
        print(f"Demo case '{case_name}' saved with ID: {case_id}")
        return case_id
    except Exception as e:
        print(f"Error saving demo case: {e}")
        return None
    finally:
        conn.close()


def get_demo_cases(category: Optional[str] = None):
    """Get all demo cases, optionally filtered by category."""
    import json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    if category:
        cursor.execute("""
            SELECT id, case_name, case_type, category, case_data, description, created_at
            FROM demo_cases
            WHERE category = ?
            ORDER BY case_name
        """, (category,))
    else:
        cursor.execute("""
            SELECT id, case_name, case_type, category, case_data, description, created_at
            FROM demo_cases
            ORDER BY category, case_name
        """)
    
    cases = []
    for row in cursor.fetchall():
        cases.append({
            "id": row[0],
            "case_name": row[1],
            "case_type": row[2],
            "category": row[3],
            "case_data": json.loads(row[4]),
            "description": row[5],
            "created_at": row[6]
        })
    
    conn.close()
    return cases


def get_demo_case(case_id: int):
    """Get a specific demo case by ID."""
    import json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT id, case_name, case_type, category, case_data, description, created_at
        FROM demo_cases
        WHERE id = ?
    """, (case_id,))
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return {
            "id": row[0],
            "case_name": row[1],
            "case_type": row[2],
            "category": row[3],
            "case_data": json.loads(row[4]),
            "description": row[5],
            "created_at": row[6]
        }
    return None


def delete_demo_case(case_id: int):
    """Delete a demo case."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("DELETE FROM demo_cases WHERE id = ?", (case_id,))
    conn.commit()
    deleted = cursor.rowcount
    conn.close()
    
    print(f"Deleted {deleted} demo case(s)")
    return deleted > 0


if __name__ == "__main__":
    # Initialize database
    init_database()
    
    # Example: Add demo customer
    demo_id = add_customer("Demo Customer", "demo@example.com")
    
    # Example: Import data
    # import_properties_from_csv(demo_id, "data/listings.csv")
    
    # Example: Get stats
    # stats = get_customer_stats(demo_id)
    # print(stats)




