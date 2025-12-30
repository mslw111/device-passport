import os
import psycopg2
import json

# DB Connection
DB_URL = "postgresql://postgres:postgres@localhost:5432/device_passport"

def seed():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    
    # Create Table if missing
    cur.execute("""
        CREATE TABLE IF NOT EXISTS device_registry (
            device_id VARCHAR(50) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            device_payload JSONB
        );
    """)
    
    # Insert a Test Device
    payload = {
        "brand": "Samsung",
        "model": "Galaxy S21",
        "gdpr_flags": {"account_tokens_present": False, "gdpr_erasure_claimed": True},
        "r2v3_flags": {"battery_health": "Good", "cosmetic_grade": "A"},
        "fft_samples": [0.1, 0.5, 0.2, 0.8, 0.3],
        "fft_sample_rate_hz": 100
    }
    
    cur.execute("""
        INSERT INTO device_registry (device_id, device_payload)
        VALUES (%s, %s)
        ON CONFLICT (device_id) DO NOTHING;
    """, ("TEST-DEV-001", json.dumps(payload)))
    
    conn.commit()
    conn.close()
    print("✅ Database seeded with TEST-DEV-001")

if __name__ == "__main__":
    seed()
