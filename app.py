import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import psycopg2
import psycopg2.extras
from scipy import signal

# --- IMPORTS ---
from db_access import *
from db_persist import persist_run, persist_audit_log
from llm_client import LLMClient

st.set_page_config(page_title="RefurbOS Pro", layout="wide", page_icon="⌚")

# --- 1. CONNECT TO NEON ---
def get_db_connection():
    # Looks for secret first (Cloud), then env (Local)
    if "DATABASE_URL" in st.secrets:
        return psycopg2.connect(st.secrets["DATABASE_URL"])
    elif os.getenv("DATABASE_URL"):
        return psycopg2.connect(os.getenv("DATABASE_URL"))
    else:
        return None

# --- 2. ENGINEERING LOGIC ---
def analyze_spectrum(samples):
    if not samples: return None
    x = np.array(samples)
    freqs, power = signal.welch(x, fs=50)
    peak_idx = np.argmax(power)
    return {"freqs": freqs, "power": power, "peak": freqs[peak_idx]}

def forecast_linear(history):
    if not history: return [], [], []
    y = np.array(history)
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    future_x = np.arange(len(y), len(y) + 90)
    return p(future_x), [], [] # keeping it simple to prevent errors

# --- 3. DASHBOARD ---
def show_dashboard():
    st.sidebar.title("⌚ RefurbOS")
    
    # Connect
    conn = get_db_connection()
    if not conn:
        st.error("❌ No Database Connection found in Secrets.")
        return
        
    # Fetch Data
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM device_registry LIMIT 1000")
            devices = cur.fetchall()
    except Exception as e:
        st.error(f"Database Error: {e}")
        return

    if not devices:
        st.warning("Database is empty. Please run the SQL script in Neon Console.")
        return

    # Selector
    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.sidebar.selectbox("Select Asset", list(dev_map.keys()))
    
    # Parse JSON
    data = dev_map[sel_id]['device_payload']
    
    # UI Layout
    st.title(f"{data.get('brand')} {data.get('model')}")
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Grade", data['condition'].get('grade', 'N/A'))
    c2.metric("Battery", f"{data['condition'].get('battery', 0)}%")
    c3.metric("Size", f"{data['specs'].get('size', 0)}mm")

    # Tabs
    t1, t2, t3 = st.tabs(["📉 Haptics", "🔮 Forecast", "🤖 AI Tribunal"])
    
    with t1:
        sig = data['telemetry'].get('haptic_signal', [])
        if sig:
            st.line_chart(sig)
            res = analyze_spectrum(sig)
            if res:
                st.metric("Peak Frequency", f"{res['peak']:.2f} Hz")
    
    with t2:
        hist = data['telemetry'].get('battery_history', [])
        if hist:
            fc, _, _ = forecast_linear(hist)
            chart_data = pd.DataFrame({"Historical": hist + [None]*len(fc), "Forecast": [None]*len(hist) + list(fc)})
            st.line_chart(chart_data)
            
    with t3:
        if st.button("Run Analysis"):
            llm = LLMClient("openai")
            prompt = f"Evaluate {data.get('model')} ({data['condition'].get('grade')}). Battery: {data['condition'].get('battery')}%. Est. Value?"
            st.write(llm.complete(prompt))

if __name__ == "__main__":
    show_dashboard()
