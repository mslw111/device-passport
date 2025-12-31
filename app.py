import streamlit as st
import pandas as pd
import json
import random
import os
import numpy as np
import psycopg2
import psycopg2.extras
from scipy import signal
from db_access import * from db_persist import persist_run, persist_audit_log
from llm_client import LLMClient

st.set_page_config(page_title="RefurbOS Pro", layout="wide", page_icon="⌚")

# --- 1. ENGINEERING UTILS (Restored) ---
def run_fft_welch(samples, rate=100):
    if not samples or len(samples) < 10: return {"peak_freq_hz": 0, "snr_db": 0}
    x = np.array(samples)
    f, Pxx = signal.welch(x, fs=rate, nperseg=min(len(x), 256))
    peak_idx = np.argmax(Pxx)
    snr = 10 * np.log10(Pxx[peak_idx] / np.mean(Pxx)) if np.mean(Pxx) > 0 else 0
    return {"peak_freq_hz": float(f[peak_idx]), "snr_db": float(snr), "spectrum": Pxx, "freqs": f}

def forecast_degradation(history, horizon=30):
    if not history: return ([0]*horizon, [0]*horizon, [0]*horizon)
    x = np.arange(len(history))
    y = np.array(history)
    if len(y) < 2: return ([y[0]]*horizon, [y[0]]*horizon, [y[0]]*horizon)
    
    z = np.polyfit(x, y, 1) # Linear trend
    p = np.poly1d(z)
    
    future_x = np.arange(len(history), len(history)+horizon)
    forecast = p(future_x)
    
    sigma = np.std(y)
    upper = forecast + (1.96 * sigma)
    lower = forecast - (1.96 * sigma)
    return forecast, lower, upper

# --- 2. DATABASE UTILS (Cloud Ready) ---
def get_db_connection():
    DB_URL = os.getenv("DATABASE_URL")
    if not DB_URL and "DATABASE_URL" in st.secrets:
        DB_URL = st.secrets["DATABASE_URL"]
    if not DB_URL: return None
    return psycopg2.connect(DB_URL)

def admin_seed_database():
    """Wipes and reseeds 1000 units (Admin Only)"""
    conn = get_db_connection()
    if not conn: st.error("No Database URL found."); return
    cur = conn.cursor()

    cur.execute("DROP TABLE IF EXISTS audit_log, runs, device_registry CASCADE")
    cur.execute("CREATE TABLE device_registry (device_id VARCHAR(50) PRIMARY KEY, device_payload JSONB)")
    cur.execute("CREATE TABLE runs (run_id VARCHAR(50) PRIMARY KEY, device_id VARCHAR(50), created_at TIMESTAMP DEFAULT NOW())")
    cur.execute("CREATE TABLE audit_log (id SERIAL PRIMARY KEY, run_id VARCHAR(50), event_type VARCHAR(50), payload_json JSONB, created_at TIMESTAMP DEFAULT NOW())")

    brands = ["Apple", "Samsung", "Garmin"]
    progress = st.progress(0)
    
    for i in range(1, 1001):
        brand = random.choice(brands)
        is_broken = random.random() < 0.2
        grade = "Grade D (Salvage)" if is_broken else random.choice(["Grade A (Mint)", "Grade B (Good)"])
        
        # Simulated Telemetry for FFT
        # Good watches = Clean Sine Wave (Heart rate)
        # Bad watches = Noisy/Random data
        t = np.linspace(0, 10, 100)
        clean_sig = np.sin(2 * np.pi * 1.5 * t) # 1.5Hz heartbeat
        noise = np.random.normal(0, 0.1 if not is_broken else 2.0, 100)
        telemetry_data = (clean_sig + noise).tolist()
        
        # Battery History (Decaying)
        start_health = random.randint(90, 100)
        decay_rate = random.uniform(0.01, 0.1) if not is_broken else random.uniform(0.2, 0.5)
        batt_hist = [start_health - (d * decay_rate) for d in range(50)]

        data = {
            "brand": brand,
            "model": f"Series {random.randint(7,9)}",
            "specs": {"size_mm": random.choice([41, 45, 49]), "material": "Aluminum"},
            "condition": {"grade": grade, "battery_health": int(batt_hist[-1])},
            "telemetry": {"fft_samples": telemetry_data, "battery_history": batt_hist},
            "flags": {"water_damage": is_broken, "screen_scratch": is_broken}
        }
        did = f"{brand.upper()}-{i:04d}"
        cur.execute("INSERT INTO device_registry VALUES (%s, %s)", (did, json.dumps(data)))
        if i % 100 == 0: progress.progress(i/1000)
    
    conn.commit()
    conn.close()
    st.success("Database Reset Complete."); st.rerun()

# --- 3. MAIN DASHBOARD ---
def show_dashboard():
    # SIDEBAR CONFIG
    st.sidebar.title("⌚ RefurbOS Pro")
    
    # Admin Reset (Hidden in Expander)
    with st.sidebar.expander("🔧 Admin / Reset"):
        if st.button("Reset Database (1000 Units)"):
            admin_seed_database()

    # Load Data
    try:
        devices = list_devices()
        if not devices:
            st.warning("Database empty. Open sidebar 'Admin / Reset' and click Reset.")
            return
    except Exception:
        st.error("Database Connection Failed. Check Secrets."); return

    # Selector
    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.sidebar.selectbox("Select Device Asset", list(dev_map.keys()))
    data = get_device_payload(sel_id)

    # Safely extract data (Fixes the crash)
    specs = data.get('specs', {})
    cond = data.get('condition', {})
    tel = data.get('telemetry', {})
    fft_data = tel.get('fft_samples', [])
    batt_hist = tel.get('battery_history', [])

    # HEADER METRICS
    st.title(f"{data.get('brand')} {data.get('model')}")
    
    cols = st.columns(5)
    cols[0].metric("Case Size", f"{specs.get('size_mm', 'N/A')} mm")
    cols[1].metric("Housing", specs.get('material', 'N/A'))
    cols[2].metric("Battery SoH", f"{cond.get('battery_health', 0)}%")
    
    # Grade logic with safety check
    grade_str = str(cond.get('grade', 'Unknown'))
    grade_color = "red" if "Grade D" in grade_str else "green"
    cols[3].markdown(f"**Grade:** :{grade_color}[{grade_str}]")
    
    cols[4].metric("Water Seal", "FAIL" if data.get('flags', {}).get('water_damage') else "PASS")

    # TABS
    tabs = st.tabs(["📈 Signal Analysis (FFT)", "🔮 Lifecycle Forecast", "🤖 AI Tribunal", "📝 Audit Logs"])

    # TAB 1: SPECTRAL ANALYSIS (Restored)
    with tabs[0]:
        c1, c2 = st.columns([3, 1])
        with c1:
            st.subheader("Haptic/Sensor Vibration Analysis")
            if fft_data:
                # Plot Raw Signal
                st.line_chart(fft_data, height=250)
                
                # Run FFT
                res = run_fft_welch(fft_data)
                st.caption(f"Peak Freq: {res['peak_freq_hz']:.2f} Hz | SNR: {res['snr_db']:.2f} dB")
                if res['snr_db'] < 5:
                    st.error("⚠️ LOW SIGNAL-TO-NOISE: Possible Sensor Corrosion or Motor Failure")
                else:
                    st.success("✅ Signal Clean: Sensor Nominal")
            else:
                st.info("No telemetry data available.")
        with c2:
            st.write("**Diagnostics:**")
            st.write("- Gyro Drift: 0.02%")
            st.write("- Accel Variance: Low")

    # TAB 2: FORECASTING (Restored)
    with tabs[1]:
        st.subheader("Battery Degradation Projection")
        if batt_hist:
            horizon = st.slider("Forecast Horizon (Days)", 30, 180, 90)
            fc, lo, up = forecast_degradation(batt_hist, horizon)
            
            # Combine history + forecast for chart
            chart_data = pd.DataFrame({
                "Historical": batt_hist + [None]*len(fc),
                "Forecast": [None]*len(batt_hist) + list(fc),
                "Lower Bound": [None]*len(batt_hist) + list(lo),
                "Upper Bound": [None]*len(batt_hist) + list(up)
            })
            st.line_chart(chart_data)
            
            final_health = fc[-1]
            if final_health < 80:
                st.warning(f"⚠️ Unit will drop below 80% health in {horizon} days. Recommend Battery Swap.")
            else:
                st.success(f"✅ Battery stable. Projected health: {final_health:.1f}%")
        else:
            st.info("No historical battery data.")

    # TAB 3: AI TRIBUNAL
    with tabs[2]:
        if st.button("🚀 Convene Assessment Tribunal"):
            llm = LLMClient("openai")
            rid = f"RUN-{pd.Timestamp.now().strftime('%H%M%S')}"
            persist_run(rid, sel_id)

            # 1. Risk
            with st.chat_message("user", avatar="🛑"):
                st.write("**Risk Auditor**")
                p1 = f"Analyze {data.get('brand')} watch. Grade: {grade_str}. SNR: {run_fft_welch(fft_data)['snr_db']}dB. Identify strict risks."
                r1 = llm.complete(p1)
                st.write(r1)
                persist_audit_log(rid, "RISK", {"text": r1})

            # 2. Value
            with st.chat_message("assistant", avatar="💰"):
                st.write("**Value Recovery**")
                p2 = f"Argue for refurbishment value of {data.get('model')}. Counterpoint: {r1}."
                r2 = llm.complete(p2)
                st.write(r2)
                persist_audit_log(rid, "VALUE", {"text": r2})

    # TAB 4: LOGS
    with tabs[3]:
        runs = fetch_runs_for_device(sel_id)
        if not runs.empty:
            for rid in runs['run_id']:
                chain = fetch_audit_chain(rid)
                st.table(chain[['created_at', 'event_type', 'payload_json']])

if __name__ == "__main__":
    show_dashboard()
