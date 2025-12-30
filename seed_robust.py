import json
import numpy as np
import random
from db_access import init_db_schema, get_conn

print("🔥 Wiping and Re-Seeding Database...")
init_db_schema()

conn = get_conn()
cur = conn.cursor()

brands = ["Apple", "Samsung", "Google", "Garmin"]
types = ["Phone", "Watch", "Tablet"]

for i in range(1, 101):
    # 1. Create Device
    brand = random.choice(brands)
    dtype = "Watch" if i < 30 else random.choice(types) # Ensure we have plenty of watches
    did = f"{brand.upper()}-{dtype.upper()}-{i:03d}"
    
    payload = {
        "brand": brand,
        "model": f"{dtype} Series {random.randint(5,9)}",
        "type": dtype,
        "gdpr_flags": {"tokens_found": False, "esim_deleted": True},
        "r2v3_flags": {"battery": "Good", "cosmetics": "A"},
        "sensor_timeseries": (100 - np.linspace(0,10,50)).tolist(),
        "fft_samples": np.random.normal(0,1,100).tolist()
    }
    
    cur.execute("INSERT INTO device_registry VALUES (%s, %s)", (did, json.dumps(payload)))
    
    # 2. FORCE HISTORY (No Randomness)
    run_id = f"RUN-HIST-{i:03d}"
    
    # Insert Run
    cur.execute("INSERT INTO runs (run_id, device_id) VALUES (%s, %s)", (run_id, did))
    
    # Insert Log Event 1: Ingest
    log1 = json.dumps({"text": f"Device received. Batch: {random.randint(100,999)}"})
    cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)", 
                (run_id, "LOGISTICS_RECEIVE", log1))
                
    # Insert Log Event 2: Initial Scan
    log2 = json.dumps({"text": "Initial automated diagnostic passed."})
    cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)", 
                (run_id, "DIAGNOSTIC_SCAN", log2))

conn.commit()

# 3. VERIFY
cur.execute("SELECT COUNT(*) FROM runs")
run_count = cur.fetchone()[0]
cur.execute("SELECT COUNT(*) FROM audit_log")
log_count = cur.fetchone()[0]
print(f"✅ SUCCESS: Created {i} Devices, {run_count} Runs, {log_count} Audit Logs.")
conn.close()
