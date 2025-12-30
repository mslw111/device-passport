import json
import numpy as np
import random
from db_access import init_db_schema, get_conn
from db_persist import persist_run, persist_audit_log

print("🌱 Seeding 100 Neural Devices (Phones + Watches)...")
init_db_schema()

conn = get_conn()
cur = conn.cursor()

# CONFIGS
BRANDS = ["Apple", "Samsung", "Google", "Garmin"]
TYPES = ["Phone", "Watch", "Tablet"]

for i in range(1, 101):
    brand = random.choice(BRANDS)
    dev_type = random.choice(TYPES)
    did = f"{brand.upper()}-{dev_type.upper()}-{i:03d}"
    
    # 1. Base Timeseries (Linear Decay + Seasonality)
    t = np.linspace(0, 50, 50)
    decay = random.uniform(0.1, 0.4)
    ts = (100 - (t * decay) + np.sin(t/3)).tolist()
    
    # 2. Base FFT (Noisy Sine)
    fft_s = (np.sin(np.linspace(0, 100, 200)) + np.random.normal(0, 0.3, 200)).tolist()

    # 3. TYPE SPECIFIC FLAGS
    gdpr = {"tokens_found": False, "erasure_verified": True}
    r2v3 = {"battery_health": "Good", "cosmetics": "Grade A"}
    
    if dev_type == "Watch":
        # Add Smartwatch Specifics
        gdpr.update({
            "esim_profile_deleted": random.choice([True, False]),
            "biometric_cache_cleared": True
        })
        r2v3.update({
            "band_condition": "Original",
            "water_resistance_seal": "Pass" if random.random() > 0.2 else "Fail"
        })
        model = f"Watch Series {random.randint(5,9)}"
        
    elif dev_type == "Phone":
        model = f"Galaxy S{random.randint(20,24)}" if brand == "Samsung" else f"iPhone {random.randint(12,15)}"
    else:
        model = "Tab S8"
        
    payload = {
        "brand": brand,
        "model": model,
        "type": dev_type,
        "gdpr_flags": gdpr,
        "r2v3_flags": r2v3,
        "sensor_timeseries": ts,
        "fft_samples": fft_s
    }
    
    cur.execute("INSERT INTO device_registry VALUES (%s, %s)", (did, json.dumps(payload)))
    
    # 4. Create Mock History (For 'Entire Database' feel)
    if random.random() > 0.5:
        rid = f"RUN-HIST-{i:03d}"
        cur.execute("INSERT INTO runs (run_id, device_id) VALUES (%s, %s)", (rid, did))
        
        # Log 1: Ingest
        log1 = json.dumps({"text": "Device received at facility via Batch-A."})
        cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)", (rid, "LOGISTICS_RECEIVE", log1))
        
        # Log 2: Auto-Check
        status = "Pass" if gdpr["erasure_verified"] else "Fail"
        log2 = json.dumps({"text": f"Automated erasure verification returned: {status}"})
        cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)", (rid, "AUTO_ERASURE_CHECK", log2))

conn.commit()
conn.close()
print("✅ Database populated with 100 mixed devices.")
