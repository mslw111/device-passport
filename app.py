import streamlit as st
import pandas as pd
import json
import random
import os
import psycopg2
import psycopg2.extras
from db_access import * # Keeps your existing DB logic
from db_persist import persist_run, persist_audit_log
from llm_client import LLMClient

st.set_page_config(page_title="RefurbOS", layout="wide", page_icon="⌚")

# --- ADMIN TOOL: SEED DATABASE FROM THE APP ---
def admin_seed_database():
    """
    Since we are in the cloud, this function allows you to wipe and 
    re-seed the database directly from the browser.
    """
    # 1. Get Connection
    DB_URL = os.getenv("DATABASE_URL")
    if not DB_URL and "DATABASE_URL" in st.secrets:
        DB_URL = st.secrets["DATABASE_URL"]
    
    if not DB_URL:
        st.error("Missing Database URL in Secrets.")
        return
        
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # 2. Wipe & Recreate
    cur.execute("DROP TABLE IF EXISTS audit_log, runs, device_registry CASCADE")
    cur.execute("CREATE TABLE device_registry (device_id VARCHAR(50) PRIMARY KEY, device_payload JSONB)")
    cur.execute("CREATE TABLE runs (run_id VARCHAR(50) PRIMARY KEY, device_id VARCHAR(50), created_at TIMESTAMP DEFAULT NOW())")
    cur.execute("CREATE TABLE audit_log (id SERIAL PRIMARY KEY, run_id VARCHAR(50), event_type VARCHAR(50), payload_json JSONB, created_at TIMESTAMP DEFAULT NOW())")

    # 3. Generate 1000 Watches
    brands = ["Apple", "Samsung", "Garmin"]
    models = {
        "Apple": ["Watch Series 9", "Watch Ultra 2", "Watch SE"],
        "Samsung": ["Galaxy Watch 6", "Galaxy Watch 5 Pro"],
        "Garmin": ["Fenix 7X", "Epix Gen 2", "Venu 3"]
    }

    progress = st.progress(0)
    for i in range(1, 1001):
        brand = random.choice(brands)
        model = random.choice(models[brand])
        
        # Logic for "Grade"
        is_broken = random.random() < 0.15
        grade = "Grade D (Salvage)" if is_broken else random.choice(["Grade A (Like New)", "Grade B (Minor Wear)"])
        
        data = {
            "brand": brand,
            "model": model,
            "specs": {
                "case_size": "49mm" if "Ultra" in model else random.choice(["41mm", "45mm"]),
                "band": "Alpine Loop" if "Ultra" in model else "Silicone Sport"
            },
            "condition": {
                "grade": grade,
                "battery_health": random.randint(82, 100)
            },
            "telemetry": {
                "sensor_oxidation": "Detected" if is_broken else "Clean",
                "haptic_response": "Weak" if is_broken else "Normal"
            }
        }
        
        did = f"{brand.upper()}-{i:04d}"
        cur.execute("INSERT INTO device_registry VALUES (%s, %s)", (did, json.dumps(data)))
        
        if i % 100 == 0:
            progress.progress(i / 1000)

    conn.commit()
    conn.close()
    st.success("✅ Database Reset Complete! Reloading...")
    st.rerun()

# --- MAIN DASHBOARD ---
def show_dashboard():
    # SIDEBAR
    st.sidebar.title("⌚ RefurbOS")
    
    # THE MAGIC BUTTON (Only visible if you expand the section)
    with st.sidebar.expander("⚠️ Admin Tools"):
        st.warning("Clicking this wipes the database.")
        if st.button("RESET DATABASE (1000 UNITS)"):
            admin_seed_database()

    # 1. LOAD DATA
    try: 
        devices = list_devices()
        if not devices: 
            st.info("Database is empty. Open 'Admin Tools' in sidebar and click RESET.")
            return
    except Exception as e:
        st.error(f"Database Connection Error: {e}")
        return

    # 2. SELECTOR
    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.sidebar.selectbox("Select Unit", list(dev_map.keys()))
    data = get_device_payload(sel_id)
    
    # Unpack
    specs = data.get('specs', {})
    cond = data.get('condition', {})
    tel = data.get('telemetry', {})
    
    # 3. HEADER
    st.title(f"{data.get('brand')} {data.get('model')}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Size", specs.get('case_size', 'N/A'))
    c2.metric("Battery", f"{cond.get('battery_health')}%")
    c3.metric("Haptics", tel.get('haptic_response'))
    
    # Color-coded Grade
    grade_color = "green" if "Grade A" in cond.get('grade') else "red"
    c4.markdown(f"### :{grade_color}[{cond.get('grade')}]")

    # 4. TABS
    t1, t2 = st.tabs(["🤖 AI Tribunal", "📜 Audit Log"])
    
    with t1:
        if st.button("Start AI Evaluation"):
            llm = LLMClient("openai")
            rid = f"RUN-{pd.Timestamp.now().strftime('%H%M%S')}"
            persist_run(rid, sel_id)
            
            # Risk Agent
            with st.chat_message("user", avatar="🛑"):
                st.write("**Risk Agent**")
                p1 = f"Analyze {data['model']} ({cond['grade']}). Risk of return due to {tel['sensor_oxidation']}?"
                r1 = llm.complete(p1)
                st.write(r1)
                persist_audit_log(rid, "RISK", {"msg": r1})
            
            # Valuation Agent
            with st.chat_message("assistant", avatar="💰"):
                st.write("**Valuation Agent**")
                p2 = f"Based on {r1}, give me a cash offer in USD."
                r2 = llm.complete(p2)
                st.write(r2)
                persist_audit_log(rid, "VALUE", {"msg": r2})

    with t2:
        runs = fetch_runs_for_device(sel_id)
        if not runs.empty:
            for rid in runs['run_id']:
                chain = fetch_audit_chain(rid)
                st.caption(f"Run ID: {rid}")
                st.table(chain[['created_at', 'event_type', 'payload_json']])

if __name__ == "__main__":
    show_dashboard()
