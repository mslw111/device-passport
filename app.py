import streamlit as st
import pandas as pd
import json
import random
import os
import numpy as np
import psycopg2
import psycopg2.extras
from scipy import signal

# --- FIXED IMPORTS (SEPARATED LINES) ---
from db_access import *
from db_persist import persist_run, persist_audit_log
from llm_client import LLMClient

st.set_page_config(page_title="RefurbOS Pro", layout="wide", page_icon="⌚")

# --- 1. ENGINEERING LOGIC ---
def analyze_spectrum(samples, rate=50):
    """
    Performs FFT (Fast Fourier Transform) to find frequency anomalies.
    """
    if not samples or len(samples) < 10: 
        return None
    
    x = np.array(samples)
    # Welchs method for spectral density
    freqs, power = signal.welch(x, fs=rate, nperseg=len(x))
    
    # Find dominant frequency
    peak_idx = np.argmax(power)
    peak_freq = freqs[peak_idx]
    
    return {
        "frequencies": freqs,
        "power_density": power,
        "peak_freq": peak_freq,
        "signal_mean": np.mean(x),
        "signal_std": np.std(x)
    }

def forecast_linear(history, days=90):
    """
    Simple linear regression forecast for battery health.
    """
    if not history or len(history) < 2:
        return [], [], []
        
    x = np.arange(len(history))
    y = np.array(history)
    
    # Fit line
    slope, intercept = np.polyfit(x, y, 1)
    
    # Forecast
    future_x = np.arange(len(history), len(history) + days)
    forecast = slope * future_x + intercept
    
    # Simple confidence intervals based on historical noise
    noise = np.std(y - (slope * x + intercept))
    upper = forecast + (1.96 * noise)
    lower = forecast - (1.96 * noise)
    
    return forecast, lower, upper

# --- 2. DB AUTO-INIT (NO BUTTONS) ---
def get_db_connection():
    DB_URL = os.getenv("DATABASE_URL")
    if not DB_URL and "DATABASE_URL" in st.secrets:
        DB_URL = st.secrets["DATABASE_URL"]
    if not DB_URL: return None
    return psycopg2.connect(DB_URL)

def check_and_seed_if_empty():
    """
    Automatically populates DB if empty. No user action required.
    """
    conn = get_db_connection()
    if not conn: return
    cur = conn.cursor()
    
    # Check count
    try:
        cur.execute("SELECT COUNT(*) FROM device_registry")
        count = cur.fetchone()[0]
    except:
        count = 0
        
    if count == 0:
        with st.spinner("⚡ Initializing RefurbOS Database (First Run Only)..."):
            # Recreate Schema
            cur.execute("DROP TABLE IF EXISTS audit_log, runs, device_registry CASCADE")
            cur.execute("CREATE TABLE device_registry (device_id VARCHAR(50) PRIMARY KEY, device_payload JSONB)")
            cur.execute("CREATE TABLE runs (run_id VARCHAR(50) PRIMARY KEY, device_id VARCHAR(50), created_at TIMESTAMP DEFAULT NOW())")
            cur.execute("CREATE TABLE audit_log (id SERIAL PRIMARY KEY, run_id VARCHAR(50), event_type VARCHAR(50), payload_json JSONB, created_at TIMESTAMP DEFAULT NOW())")
            
            # Seed Data
            brands = ["Apple", "Samsung", "Garmin"]
            for i in range(1, 1001):
                brand = random.choice(brands)
                
                # Generate Telemetry Signals
                # 1. Clean Signal (Sine wave)
                t = np.linspace(0, 4, 100)
                clean_sig = 10 * np.sin(2 * np.pi * 2.0 * t) # 2Hz signal
                
                # 2. Add Defects
                is_broken = random.random() < 0.2
                if is_broken:
                    # Add random noise (Rattle)
                    noise = np.random.normal(0, 5, 100)
                    sig = clean_sig + noise
                    grade = "Grade D (Salvage)"
                    batt_decay = 0.5
                else:
                    noise = np.random.normal(0, 0.5, 100)
                    sig = clean_sig + noise
                    grade = "Grade A (Mint)"
                    batt_decay = 0.05

                # Battery History
                start_batt = random.randint(90, 100)
                history = [start_batt - (x * batt_decay) for x in range(30)]

                data = {
                    "brand": brand,
                    "model": f"Series {random.randint(7,9)}",
                    "specs": {"size": random.choice([41, 45, 49])},
                    "condition": {"grade": grade, "battery": int(history[-1])},
                    "telemetry": {
                        "haptic_signal": sig.tolist(),
                        "battery_history": history
                    }
                }
                did = f"{brand.upper()}-{i:04d}"
                cur.execute("INSERT INTO device_registry VALUES (%s, %s)", (did, json.dumps(data)))
            
            conn.commit()
            st.success("Database Ready.")
            st.rerun()
    conn.close()

# --- 3. DASHBOARD ---
def show_dashboard():
    # 1. Auto-Init
    check_and_seed_if_empty()
    
    st.sidebar.title("⌚ RefurbOS")
    
    # 2. Load Data
    try: devices = list_devices()
    except: st.error("Database connection failed."); return
    
    if not devices: st.warning("Initializing..."); return

    # 3. Sidebar Selection
    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.sidebar.selectbox("Select Asset", list(dev_map.keys()))
    data = get_device_payload(sel_id)
    
    # 4. Extract
    telemetry = data.get('telemetry', {})
    signal_data = telemetry.get('haptic_signal', [])
    batt_hist = telemetry.get('battery_history', [])
    
    # 5. Header
    st.title(f"{data.get('brand')} {data.get('model')}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Grade", data['condition']['grade'])
    col2.metric("Battery Health", f"{data['condition']['battery']}%")
    col3.metric("Case Size", f"{data['specs']['size']}mm")

    # 6. TABS
    tabs = st.tabs(["📉 Spectral Analysis (FFT)", "🔮 Forecasting", "🤖 AI Tribunal", "📝 Logs"])

    # TAB 1: FOURIER TRANSFORM
    with tabs[0]:
        st.subheader("Haptic Engine Frequency Response")
        if signal_data:
            c1, c2 = st.columns([2, 1])
            with c1:
                # A. Raw Signal Chart
                st.caption("Raw Vibration Telemetry (Time Domain)")
                st.line_chart(signal_data)
            
            with c2:
                # B. FFT Chart
                res = analyze_spectrum(signal_data)
                if res:
                    st.caption("Frequency Spectrum (Frequency Domain)")
                    
                    # Create a dataframe for the spectrum chart
                    fft_df = pd.DataFrame({
                        "Frequency (Hz)": res['frequencies'],
                        "Power Density": res['power_density']
                    })
                    st.line_chart(fft_df, x="Frequency (Hz)", y="Power Density")
                    
                    st.metric("Dominant Frequency", f"{res['peak_freq']:.2f} Hz")
                    if res['signal_std'] > 4:
                        st.error("⚠️ High Noise Detected (Possible Loose Component)")
                    else:
                        st.success("✅ Signal Clean")
        else:
            st.info("No telemetry data found.")

    # TAB 2: FORECASTING
    with tabs[1]:
        st.subheader("Lifecycle Prediction")
        if batt_hist:
            horizon = st.slider("Project Days", 30, 365, 90)
            fc, lo, up = forecast_linear(batt_hist, horizon)
            
            # Combine for chart
            hist_len = len(batt_hist)
            fc_len = len(fc)
            
            # Pad data so they align on the chart
            chart_data = pd.DataFrame({
                "Historical": batt_hist + [None]*fc_len,
                "Forecast": [None]*hist_len + list(fc),
                "Lower Bound": [None]*hist_len + list(lo),
                "Upper Bound": [None]*hist_len + list(up)
            })
            
            st.line_chart(chart_data)
            st.caption(f"Projected Health in {horizon} days: {fc[-1]:.1f}%")
        else:
            st.info("No battery history available.")

    # TAB 3: TRIBUNAL
    with tabs[2]:
        if st.button("Run AI Assessment"):
            llm = LLMClient("openai")
            rid = f"RUN-{pd.Timestamp.now().strftime('%H%M%S')}"
            persist_run(rid, sel_id)
            
            with st.chat_message("user", avatar="🛑"):
                st.write("Analyzing Telemetry...")
                prompt = f"Device: {data['model']}. Grade: {data['condition']['grade']}. Battery Trend: {batt_hist[-1]}%. Signal Noise: {'High' if np.std(signal_data) > 4 else 'Low'}. Assess Refurb Viability."
                resp = llm.complete(prompt)
                st.write(resp)
                persist_audit_log(rid, "AI_ANALYSIS", {"text": resp})

    # TAB 4: LOGS
    with tabs[3]:
        runs = fetch_runs_for_device(sel_id)
        if not runs.empty:
            for rid in runs['run_id']:
                st.write(f"Run: {rid}")
                chain = fetch_audit_chain(rid)
                st.table(chain[['created_at', 'event_type', 'payload_json']])

if __name__ == "__main__":
    show_dashboard()
