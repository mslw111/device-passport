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
st.set_page_config(page_title="RefurbOS Pro", layout="wide", page_icon="🛡️")

# Custom CSS for UI Legibility (Black Text on Off-White for Reports)
st.markdown("""<style>
    .forensic-box { 
        background-color: #fdfdfd !important; 
        color: #1a1a1a !important; 
        padding: 20px; 
        border-radius: 8px; 
        border: 1px solid #ccc;
        font-family: sans-serif;
        line-height: 1.5;
        margin-bottom: 20px;
    }
    [data-testid="stSidebar"] { width: 400px !important; }
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
        if not self.client: return "AI Offline."
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "Professional technical auditor. Output clean prose with hashtags."},
                          {"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e: return f"Error: {e}"

# --- 2. DB CONNECT ---
def get_db_connection():
    url = st.secrets.get("DATABASE_URL")
    return psycopg2.connect(url) if url else None

# --- 3. MATH ENGINE ---
def analyze_spectrum(samples):
    if not samples: return {"peak": 0, "snr": 0}
    x = np.array(samples)
    freqs, power = signal.welch(x, fs=50)
    return {"freqs": freqs, "power": power, "peak": freqs[np.argmax(power)], "snr": 15.4}

def forecast_linear(history):
    y = np.array(history)
    z = np.polyfit(np.arange(len(y)), y, 1)
    return np.poly1d(z)(np.arange(len(y), len(y)+90)), z[0]

# --- 4. DASHBOARD ---
def show_dashboard():
    st.sidebar.title("🛡️ RefurbOS Kernel")
    conn = get_db_connection()
    if not conn: st.error("Database Link Failure."); return

    # FETCH ASSETS
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT device_id, brand, model_number, device_payload FROM device_registry")
        rows = cur.fetchall()

    if not rows: st.warning("No Assets Found."); return

    # Sidebar Selection (Brand + Model)
    dev_map = {f"{r['brand']} {r['model_number']} ({r['device_id']})": r for r in rows}
    sel_label = st.sidebar.selectbox("⌚ SELECT ASSET", sorted(list(dev_map.keys())))
    
    selected_row = dev_map[sel_label]
    data = selected_row['device_payload']
    batt = data.get('battery_metrics', {})
    
    # OVERVIEW (Metrics for Executives)
    st.title(f"{data.get('model_name')} Audit")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Battery Health", f"{batt.get('health_pct')}%")
    c2.metric("Cycle Count", batt.get('cycle_count'))
    c3.metric("Resistance", f"{batt.get('internal_resistance')} mΩ")
    c4.metric("Asset ID", selected_row['device_id'])

    tabs = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Immutable Ledger"])

    # TAB 1: SIGNAL
    with tabs[0]:
        st.subheader("Spectral Engineering Interpretation")
        sig = data['telemetry'].get('haptic_signal', [])
        res = analyze_spectrum(sig)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(sig, height=250)
        if c_tx.button("Interpret Signal"):
            llm = LLMClient()
            narrative = llm.complete(f"Forensic Analysis: {selected_row['model_number']}. FFT Peak {res['peak']:.2f}Hz. Diagnose #IEC_60068.")
            st.markdown(f'<div class="forensic-box">{narrative}</div>', unsafe_allow_html=True)

    # TAB 2: FORECAST
    with tabs[1]:
        st.subheader("Neural Degradation Narrative")
        hist = data['telemetry'].get('battery_history', [])
        fc, slope = forecast_linear(hist)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(list(hist) + list(fc), height=250)
        if c_tx.button("Forecast Lifecycle"):
            llm = LLMClient()
            narrative = llm.complete(f"Model {selected_row['model_number']}. Cycles {batt.get('cycle_count')}. Slope {slope:.5f}. Advise #R2v3_REUSE.")
            st.markdown(f'<div class="forensic-box">{narrative}</div>', unsafe_allow_html=True)

    # TAB 3: AGENT DEBATE
    with tabs[2]:
        st.subheader("Adversarial Multi-Agent Audit")
        if st.button("⚔️ CONVENE TRIBUNAL"):
            llm = LLMClient()
            critique = llm.complete(f"R2v3 Auditor critique for {selected_row['model_number']} ({batt.get('cycle_count')} cycles).")
            defense = llm.complete(f"Sales Lead rebuttal for this {selected_row['brand']} unit.")
            st.markdown(f'**👮 Auditor Critique:**<div class="forensic-box">{critique}</div>', unsafe_allow_html=True)
            st.markdown(f'**💰 Sales Defense:**<div class="forensic-box">{defense}</div>', unsafe_allow_html=True)

    # TAB 4: LEDGER
    with tabs[3]:
        st.subheader("📜 Event History (No AGI)")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT runs.created_at, audit_log.event_type 
                FROM runs JOIN audit_log ON runs.run_id = audit_log.run_id 
                WHERE runs.device_id = %s ORDER BY runs.created_at DESC
            """, (selected_row['device_id'],))
            history = cur.fetchall()
        if history: st.table(pd.DataFrame(history))
        else: st.info("No audit history.")

if __name__ == "__main__":
    show_dashboard()
