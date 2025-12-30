import json
import numpy as np
import random
from db_access import init_db_schema, get_conn
from db_persist import persist_run, persist_audit_log

print("🌱 Initializing Database Schema...")
init_db_schema()

# DATA CONFIG
BRANDS = ["Samsung", "Apple", "Google", "Sony", "Motorola"]
MODELS = {
    "Samsung": ["Galaxy S21", "Galaxy S22 Ultra", "Galaxy Z Fold"],
    "Apple": ["iPhone 13", "iPhone 14 Pro", "iPhone 12 Mini"],
    "Google": ["Pixel 6", "Pixel 7 Pro"],
    "Sony": ["Xperia 1", "Xperia 5"],
    "Motorola": ["Razr 5G", "Edge+"]
}

print("🏭 Generating 100 Neural Devices...")

devices_to_insert = []
audit_logs_to_insert = []

conn = get_conn()
cur = conn.cursor()

for i in range(1, 101):
    brand = random.choice(BRANDS)
    model = random.choice(MODELS[brand])
    dev_id = f"{brand.upper()}-{model.split()[-1].upper()}-{i:03d}"
    
    # 1. Randomized Degradation (Forecasting Data)
    # Some devices degrade fast (steep slope), some slow
    start_health = random.uniform(95, 100)
    degradation_rate = random.uniform(0.05, 0.5) 
    t = np.linspace(0, 50, 50)
    noise = np.random.normal(0, 0.2, 50)
    # Formula: Start - (Time * Rate) + Periodic Seasonality + Noise
    ts_data = (start_health - (t * degradation_rate) + np.sin(t/5)*0.5 + noise).tolist()

    # 2. Randomized Spectral Signature (FFT Data)
    # Different 'defects' create different frequency spikes
    freq_shift = random.uniform(0.5, 2.0)
    fft_data = (np.sin(np.linspace(0, 100, 200) * freq_shift) + np.random.normal(0, 0.5, 200)).tolist()

    # 3. Compliance Flags
    is_compliant = random.random() > 0.2 # 80% chance of being compliant
    
    payload = {
        "brand": brand,
        "model": model,
        "batch_id": f"BATCH-{random.randint(1000, 9999)}",
        "gdpr_flags": {
            "account_tokens_present": not is_compliant, 
            "gdpr_erasure_claimed": is_compliant
        },
        "r2v3_flags": {
            "battery_health": "Good" if start_health > 90 else "Degraded",
            "screen_cracked": random.choice([True, False]) if not is_compliant else False
        },
        "sensor_timeseries": ts_data,
        "fft_samples": fft_data,
        "fft_sample_rate_hz": 100
    }
    
    # INSERT DEVICE
    cur.execute("INSERT INTO device_registry (device_id, device_payload) VALUES (%s, %s)", 
                (dev_id, json.dumps(payload)))
    
    # INSERT AUDIT TRAIL (So the log isn't empty)
    # We add a "System Import" event for every device
    run_id = f"RUN-IMPORT-{i:03d}"
    cur.execute("INSERT INTO runs (run_id, device_id) VALUES (%s, %s)", (run_id, dev_id))
    
    log_payload = json.dumps({"status": "Imported", "compliance_check": "Pending"})
    cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)",
                (run_id, "SYSTEM_INGEST", log_payload))

conn.commit()
conn.close()
print("✅ Database populated with 100 unique devices.")
