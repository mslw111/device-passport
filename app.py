import os
import sys
import psycopg2
import json
import subprocess
import time

# 1. SETUP CONNECTION
# We hardcode it here to guarantee it works without environment variables
DB_HOST = "localhost"
DB_NAME = "device_passport"
DB_USER = "postgres"
DB_PASS = "postgres"
DB_URL = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/{DB_NAME}"

def seed_database():
    print(f"🔌 Connecting to {DB_URL}...")
    try:
        conn = psycopg2.connect(DB_URL)
        conn.autocommit = True
        cur = conn.cursor()
    except Exception as e:
        print(f"❌ CONNECTION FAILED: {e}")
        print("Ensure Docker is running: 'docker start dp-db'")
        return False

    # 2. RESET TABLE
    print("🧹 Cleaning old data...")
    cur.execute("DROP TABLE IF EXISTS device_registry;")
    cur.execute("""
        CREATE TABLE device_registry (
            device_id VARCHAR(50) PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            device_payload JSONB
        );
    """)

    # 3. INSERT DATA
    print("🌱 Seeding devices...")
    devices = [
        {
            "id": "SAMSUNG-S21-001", 
            "payload": {
                "brand": "Samsung", "model": "Galaxy S21", 
                "gdpr_flags": {"account_tokens_present": True, "gdpr_erasure_claimed": False},
                "r2v3_flags": {"screen_cracked": False, "battery_health": "Good"}
            }
        },
        {
            "id": "IPHONE-13-X99", 
            "payload": {
                "brand": "Apple", "model": "iPhone 13", 
                "gdpr_flags": {"account_tokens_present": False, "gdpr_erasure_claimed": True},
                "r2v3_flags": {"screen_cracked": True, "battery_health": "Degraded"}
            }
        }
    ]

    for d in devices:
        cur.execute(
            "INSERT INTO device_registry (device_id, device_payload) VALUES (%s, %s)",
            (d["id"], json.dumps(d["payload"]))
        )
    
    print(f"✅ Successfully seeded {len(devices)} devices.")
    conn.close()
    return True

def launch_app():
    print("🚀 Launching Streamlit App...")
    # Set the env var specifically for this subprocess
    env = os.environ.copy()
    env["DP_DATABASE_URL"] = DB_URL
    
    # Run Streamlit
    subprocess.run(["streamlit", "run", "app.py", "--server.port", "8502"], env=env)

if __name__ == "__main__":
    if seed_database():
        time.sleep(1)
        launch_app()
st.sidebar.title("✅ NEURAL AUDIT")
role = st.sidebar.selectbox("Role", ["Management", "Engineer"])
show_device_dashboard(role)
