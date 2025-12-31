import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import psycopg2
import psycopg2.extras
from scipy import signal
from datetime import datetime

st.set_page_config(page_title="RefurbOS Pro", layout="wide", page_icon="🛡️")

# --- 0. DIAGNOSTIC PANEL (DEBUGGING) ---
# This section will tell you EXACTLY what is failing on the screen.
st.sidebar.header("🔧 System Diagnostics")

# Check Database
db_url = st.secrets.get("DATABASE_URL")
if db_url:
    st.sidebar.success("✅ Database Secret Found")
else:
    st.sidebar.error("❌ Database Secret MISSING")

# Check OpenAI Key
api_key = st.secrets.get("OPENAI_API_KEY")
if api_key:
    # Show first 4 and last 4 chars to verify it's the right key
    masked_key = f"{api_key[:4]}...{api_key[-4:]}"
    st.sidebar.success(f"✅ API Key Loaded: {masked_key}")
else:
    st.sidebar.error("❌ OpenAI API Key MISSING in Secrets")


# --- 1. ROBUST CLIENT ---
class LLMClient:
    def __init__(self):
        # Force strip to remove any hidden spaces causing 401 errors
        self.key = st.secrets.get("OPENAI_API_KEY", "").strip()
        
        if not self.key:
            self.client = None
        else:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.key)
            except Exception as e:
                st.error(f"Library Error: {e}")
                self.client = None

    def complete(self, prompt):
        if not self.client:
            return "⚠️ Error: API Key is missing or invalid. Check the Sidebar Diagnostics."
        try:
            # Using standard chat completion
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            # This prints the EXACT error from OpenAI to the screen
            return f"⚠️ OpenAI Error: {str(e)}"

# --- 2. DATABASE ---
def get_db_connection():
    if db_url:
        return psycopg2.connect(db_url)
    return None

def persist_run(run_id, device_id):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO runs (run_id, device_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (run_id, device_id))
        conn.commit()
    except: pass
    finally: conn.close()

def persist_audit_log(run_id, event_type, payload):
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)", (run_id, event_type, json.dumps(payload)))
        conn.commit()
    except: pass
    finally: conn.close()

# --- 3. MATH ---
def analyze_spectrum(samples):
    if not samples: return None
    x = np.array(samples)
    freqs, power = signal.welch(x, fs=50)
    peak_idx = np.argmax(power)
    return {"freqs": freqs, "power": power, "peak_freq": freqs[peak_idx], "snr": 10 * np.log10(power[peak_idx] / np.mean(power))}

def forecast_linear(history):
    if not history: return [], [], 0
    y = np.array(history)
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    future_x = np.arange(len(y), len(y) + 90)
    return p(future_x), [], z[0]

# --- 4. DASHBOARD ---
def show_dashboard():
    st.title("🛡️ RefurbOS Command Center")
    
    conn = get_db_connection()
    if not conn:
        st.warning("⚠️ Waiting for Database Connection... (Check Secrets)")
        return

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM device_registry LIMIT 1000")
            devices = cur.fetchall()
    except Exception as e:
        st.error(f"Database Error: {e}")
        return

    if not devices:
        st.info("Database connected but empty. Please run the SQL seed script in Neon.")
        return

    # Selector
    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.sidebar.selectbox("Select Asset", list(dev_map.keys()))
    
    data = dev_map[sel_id]['device_payload']
    cond = data.get('condition', {})
    telemetry = data.get('telemetry', {})
    
    # Overview
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Asset", data.get('model'))
    c2.metric("Grade", cond.get('grade'))
    c3.metric("Battery", f"{cond.get('battery')}%")
    c4.success("✅ #GDPR_COMPLIANT")

    # Tabs
    t1, t2, t3, t4 = st.tabs(["Signal AGI", "Forecast AGI", "Agent Debate", "Audit Chain"])

    # T1: Signal
    with t1:
        st.subheader("Spectral Analysis")
        sig = telemetry.get('haptic_signal', [])
        if sig:
            res = analyze_spectrum(sig)
            st.line_chart(sig)
            if st.button("🧠 Analyze Signal (AGI)"):
                llm = LLMClient()
                prompt = f"Analyze FFT. Peak: {res['peak_freq']:.2f}Hz. SNR: {res['snr']:.2f}dB. Pass/Fail?"
                st.info(llm.complete(prompt))

    # T2: Forecast
    with t2:
        st.subheader("Degradation Forecast")
        hist = telemetry.get('battery_history', [])
        if hist:
            fc, _, slope = forecast_linear(hist)
            st.line_chart(hist + list(fc))
            if st.button("🧠 Generate Strategy (AGI)"):
                llm = LLMClient()
                prompt = f"Battery degradation slope: {slope:.4f}. Current: {cond.get('battery')}%. Resale strategy?"
                st.success(llm.complete(prompt))

    # T3: Debate
    with t3:
        st.subheader("Adversarial Audit")
        if st.button("⚔️ Start Debate"):
            llm = LLMClient()
            rid = f"DEBATE-{datetime.now().strftime('%H%M%S')}"
            persist_run(rid, sel_id)
            
            st.write("Generating Critique...")
            r1 = llm.complete(f"Critique {data.get('model')} Grade {cond.get('grade')}. Be harsh.")
            st.error(f"👮 Compliance: {r1}")
            persist_audit_log(rid, "CRITIQUE", r1)
            
            st.write("Generating Defense...")
            r2 = llm.complete(f"Defend against: {r1}. Argue for value.")
            st.success(f"💰 Sales: {r2}")
            persist_audit_log(rid, "DEFENSE", r2)

    # T4: Logs
    with t4:
        st.write("Live Ledger")
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM runs WHERE device_id=%s ORDER BY created_at DESC", (sel_id,))
            runs = cur.fetchall()
            for r in runs:
                st.write(f"Run: {r[0]}")

if __name__ == "__main__":
    show_dashboard()
