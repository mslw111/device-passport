import os
import json
import random
import psycopg2
from dotenv import load_dotenv

# Force load local .env to find Neon URL
load_dotenv(override=True)
DB_URL = os.getenv("DATABASE_URL")

if not DB_URL or "localhost" in DB_URL:
    print("⚠️ WARNING: You seem to be connected to Localhost, not Cloud.")
    print("   Please check your .env file for the Neon URL.")

print(f"🔌 CONNECTING TO: {DB_URL.split('@')[1] if '@' in DB_URL else 'DB'}...")

def generate_watch(i):
    # Guaranteed Data - No N/As
    brands = ["Apple", "Samsung", "Garmin"]
    models = {
        "Apple": ["Watch Series 9", "Watch Ultra 2", "Watch SE"],
        "Samsung": ["Galaxy Watch 6", "Galaxy Watch 5 Pro"],
        "Garmin": ["Fenix 7X", "Epix Gen 2", "Venu 3"]
    }
    
    brand = random.choice(brands)
    model = random.choice(models[brand])
    
    # Specific Specs per model
    if "Ultra" in model or "Fenix" in model: size = "49mm"
    elif "Pro" in model: size = "45mm"
    else: size = random.choice(["41mm", "45mm"])
    
    # Condition Logic
    is_broken = random.random() < 0.15
    grade = "Grade D (Salvage)" if is_broken else random.choice(["Grade A (Like New)", "Grade B (Minor Wear)"])
    
    return {
        "brand": brand,
        "model": model,
        "specs": {
            "case_size": size,
            "material": "Titanium" if "Ultra" in model else "Aluminum",
            "band": "Alpine Loop" if "Ultra" in model else "Silicone Sport"
        },
        "condition": {
            "grade": grade,
            "battery_health": random.randint(82, 100),
            "scratches": "Deep" if is_broken else "None"
        },
        "telemetry": {
            "sensor_oxidation": "Detected" if is_broken else "Clean",
            "haptic_response": "Weak" if is_broken else "Normal"
        }
    }

conn = psycopg2.connect(DB_URL)
cur = conn.cursor()

print("🧹 Wiping old data...")
cur.execute("DROP TABLE IF EXISTS audit_log, runs, device_registry CASCADE")

print("🏗️ Recreating Tables...")
cur.execute("CREATE TABLE device_registry (device_id VARCHAR(50) PRIMARY KEY, device_payload JSONB)")
cur.execute("CREATE TABLE runs (run_id VARCHAR(50) PRIMARY KEY, device_id VARCHAR(50), created_at TIMESTAMP DEFAULT NOW())")
cur.execute("CREATE TABLE audit_log (id SERIAL PRIMARY KEY, run_id VARCHAR(50), event_type VARCHAR(50), payload_json JSONB, created_at TIMESTAMP DEFAULT NOW())")

print("🚀 Injecting 1000 Clean Smartwatches...")
for i in range(1, 1001):
    data = generate_watch(i)
    did = f"{data['brand'].upper()}-{i:04d}"
    
    # Insert Device
    cur.execute("INSERT INTO device_registry VALUES (%s, %s)", (did, json.dumps(data)))
    
    # Insert distinct history (NOT a chat log)
    run_id = f"HIST-{i}"
    cur.execute("INSERT INTO runs (run_id, device_id) VALUES (%s, %s)", (run_id, did))
    
    # Log 1: Import
    cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)",
                (run_id, "SYSTEM_INGEST", json.dumps({"location": "Warehouse A", "agent": "Scanner_Bot_4"})))
    
    # Log 2: Triage (Only for some)
    if data['condition']['grade'] == "Grade A (Like New)":
         cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)",
                (run_id, "AUTO_TRIAGE", json.dumps({"status": "Direct-to-Resale", "price_est": "$250"})))

conn.commit()
conn.close()
print("✅ SUCCESS: 1000 Units Seeded.")
