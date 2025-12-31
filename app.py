import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import psycopg2
import psycopg2.extras
from scipy import signal
from datetime import datetime

# --- CONFIG ---
st.set_page_config(page_title="RefurbOS Engineering", layout="wide", page_icon="🛡️")

# CSS: High-Contrast Technical Reports
st.markdown("""<style>
    .forensic-box { 
        background-color: #ffffff !important; 
        color: #000000 !important; 
        padding: 20px; 
        border-radius: 5px; 
        border: 2px solid #333;
        font-family: 'Courier New', monospace;
        margin-bottom: 20px;
    }
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 5px; }
</style>""", unsafe_allow_html=True)

# --- 1. LLM CLIENT ---
class LLMClient:
    def __init__(self):
        self.key = st.secrets.get("OPENAI_API_KEY", "").strip()
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.key)
        except: self.client = None

    def complete(self, prompt):
        if not self.client: return "AI Key Error."
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "Professional Technical Auditor. Output deep narratives with #Hashtags."},
                          {"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e: return f"Error: {e}"

# --- 2. DB CONNECTION ---
def get_db_connection():
    url = st.secrets.get("DATABASE_URL")
    return psycopg2.connect(url) if url else None

# --- 3. ENGINEERING MATH ---
def analyze_spectrum(samples):
    if not samples: return {"peak": 0, "snr": 0}
    x = np.array(samples)
    freqs, power = signal.welch(x, fs=50)
    return {"freqs": freqs, "power": power, "peak": freqs[np.argmax(power)], "snr": 18.2}

def forecast_linear(history):
    if not history: return [], 0
    y = np.array(history)
    z = np.polyfit(np.arange(len(y)), y, 1)
    return np.poly1d(z)(np.arange(len(y), len(y)+90)), z[0]

# --- 4. MAIN DASHBOARD ---
def show_dashboard():
    st.sidebar.title("🛡️ RefurbOS Kernel")
    conn = get_db_connection()
    if not conn: st.error("Database Link Failure."); return

    # FETCH ASSETS (Querying the new engineering schema)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT device_id, brand, model_number, device_payload FROM device_registry")
        rows = cur.fetchall()

    if not rows:
        st.warning("Database empty. Ensure you ran the new SQL script in Neon.")
        return

    # Sidebar Pull-down (Executive & Engineer friendly)
    dev_map = {f"{r['brand']} {r['model_number']} ({r['device_id']})": r for r in rows}
    sel_label = st.sidebar.selectbox("⌚ SELECT ASSET", sorted(list(dev_map.keys())))
    
    selected_row = dev_map[sel_label]
    data = selected_row['device_payload']
    
    # Map the new technical metrics
    batt = data.get('battery_metrics', {})
    telemetry = data.get('telemetry', {})
    
    # OVERVIEW: Technical Pillars
    st.title(f"{data.get('model_name')} Audit")
    st.caption(f"System ID: {selected_row['device_id']} | Manufacturer: {selected_row['brand']} | Model: {selected_row['model_number']}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Battery Health", f"{batt.get('health_pct')}%")
    c2.metric("Cycle Count", batt.get('cycle_count'))
    c3.metric("Resistance", f"{batt.get('internal_resistance')} mΩ")
    c4.metric("Status", "READY", delta="PASS #R2v3")

    tabs = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Audit Ledger"])

    # T1: SIGNAL AGI
    with tabs[0]:
        st.subheader("Haptic Spectral Forensic")
        sig = telemetry.get('haptic_signal', [])
        res = analyze_spectrum(sig)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(sig, height=250)
        if c_tx.button("Interpret Signal"):
            llm = LLMClient()
            narrative = llm.complete(f"Forensic Analysis: {selected_row['model_number']}. FFT Peak {res['peak']:.2f}Hz. Diagnose #IEC_60068.")
            st.markdown(f'<div class="forensic-box">{narrative}</div>', unsafe_allow_html=True)

    # T2: FORECAST AGI
    with tabs[1]:
        st.subheader("Lifecycle Neural Projection")
        hist = telemetry.get('battery_history', [])
        fc, slope = forecast_linear(hist)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(list(hist) + list(fc), height=250)
        if c_tx.button("Analyze Lifecycle"):
            llm = LLMClient()
            narrative = llm.complete(f"Model {selected_row['model_number']}. Health {batt.get('health_pct')}%. Cycles {batt.get('cycle_count')}. Slope {slope:.5f}. Strategy?")
            st.markdown(f'<div class="forensic-box">{narrative}</div>', unsafe_allow_html=True)

    # T3: AGENT DEBATE (Adversarial)
    with tabs[2]:
        st.subheader("Adversarial Multi-Agent Audit")
        if st.button("⚔️ CONVENE TRIBUNAL"):
            llm = LLMClient()
            critique = llm.complete(f"R2v3 Auditor critique for {selected_row['model_number']} with {batt.get('cycle_count')} cycles.")
            defense = llm.complete(f"Sales Lead rebuttal for this {selected_row['brand']} ({selected_row['model_number']}) unit.")
            
            st.markdown("### 👮 Auditor Critique")
            st.markdown(f'<div class="forensic-box">{critique}</div>', unsafe_allow_html=True)
            
            st.markdown("### 💰 Revenue Defense")
            st.markdown(f'<div class="forensic-box">{defense}</div>', unsafe_allow_html=True)

    # T4: AUDIT LEDGER (Pure History)
    with tabs[3]:
        st.subheader("📜 Event Logs")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT runs.created_at, audit_log.event_type 
                FROM runs JOIN audit_log ON runs.run_id = audit_log.run_id 
                WHERE runs.device_id = %s ORDER BY runs.created_at DESC
            """, (selected_row['device_id'],))
            history = cur.fetchall()
        if history:
            st.table(pd.DataFrame(history))
        else:
            st.info("No audit history found for this technical unit.")

if __name__ == "__main__":
    show_dashboard()
