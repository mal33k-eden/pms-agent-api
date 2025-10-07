-- 1. Main drugs table
CREATE TABLE IF NOT EXISTS drugs(
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    generic_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW()
);

-- 2. Combined safety data table
CREATE TABLE IF NOT EXISTS drug_safety_data(
    id SERIAL PRIMARY KEY,
    drug_id INTEGER REFERENCES drugs(id) ON DELETE CASCADE,
    -- FDA data
    pregnancy_category CHAR(1),
    pregnancy_text TEXT,
    breastfeeding_text TEXT,
    -- Processed/AI analysis
    pregnancy_safety VARCHAR(20),
    breastfeeding_safety VARCHAR(20),
    ai_summary TEXT,
    key_warnings TEXT[],
    -- Meta
    data_source VARCHAR(50),
    confidence_score DECIMAL(3, 2),
    study_count INTEGER DEFAULT 0,
    fetched_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP DEFAULT (NOW() + INTERVAL '30 days')
);

-- 5. Processing queue
CREATE TABLE IF NOT EXISTS processing_queue(
    id SERIAL PRIMARY KEY,
    drug_name VARCHAR(255),
    priority INTEGER DEFAULT 5,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_drug_name ON drugs(LOWER(name));
CREATE INDEX IF NOT EXISTS idx_safety_drug ON drug_safety_data(drug_id);
