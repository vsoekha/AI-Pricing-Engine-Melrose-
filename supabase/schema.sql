-- Supabase Database Schema voor AI Pricing Engine
-- Run dit script in Supabase SQL Editor

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Customers table
CREATE TABLE IF NOT EXISTS customers (
    id SERIAL PRIMARY KEY,
    customer_name TEXT UNIQUE NOT NULL,
    contact_email TEXT,
    contact_phone TEXT,
    status TEXT DEFAULT 'demo',
    model_trained BOOLEAN DEFAULT FALSE,
    model_path TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Properties table
CREATE TABLE IF NOT EXISTS properties (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
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
);

-- Training runs table
CREATE TABLE IF NOT EXISTS training_runs (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER NOT NULL REFERENCES customers(id) ON DELETE CASCADE,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_count INTEGER,
    accuracy REAL,
    precision_score REAL,
    recall_score REAL,
    f1_score REAL,
    roc_auc REAL,
    model_path TEXT,
    notes TEXT
);

-- API usage tracking
CREATE TABLE IF NOT EXISTS api_usage (
    id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(id) ON DELETE SET NULL,
    endpoint TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    response_time_ms INTEGER,
    success BOOLEAN
);

-- Demo cases table
CREATE TABLE IF NOT EXISTS demo_cases (
    id SERIAL PRIMARY KEY,
    case_name TEXT NOT NULL,
    case_type TEXT NOT NULL,
    category TEXT NOT NULL,
    case_data JSONB NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_properties_customer_id ON properties(customer_id);
CREATE INDEX IF NOT EXISTS idx_properties_asset_type ON properties(asset_type);
CREATE INDEX IF NOT EXISTS idx_properties_city ON properties(city);
CREATE INDEX IF NOT EXISTS idx_training_runs_customer_id ON training_runs(customer_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_customer_id ON api_usage(customer_id);
CREATE INDEX IF NOT EXISTS idx_api_usage_timestamp ON api_usage(timestamp);
CREATE INDEX IF NOT EXISTS idx_demo_cases_category ON demo_cases(category);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_properties_updated_at BEFORE UPDATE ON properties
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_demo_cases_updated_at BEFORE UPDATE ON demo_cases
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Comments for documentation
COMMENT ON TABLE customers IS 'Klanten/clienten die het systeem gebruiken';
COMMENT ON TABLE properties IS 'Vastgoed listings voor training en analyse';
COMMENT ON TABLE training_runs IS 'Geschiedenis van model training runs';
COMMENT ON TABLE api_usage IS 'Tracking van API gebruik voor analytics';
COMMENT ON TABLE demo_cases IS 'Vooraf opgeslagen demo cases voor presentaties';




