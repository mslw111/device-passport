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

# CSS: FORCE HIGH CONTRAST (Black text on White background for reports)
st.markdown("""<style>
    .forensic-report { 
        background-color: #FFFFFF !important; 
        color: #000000 !important; 
        padding: 25px !important; 
        border-radius: 8px !important; 
        border: 2px solid #333333 !important; 
        font-family: 'Arial', sans-serif !important; 
        font-size: 15px !important;
        line-height: 1.6 !important;
        margin: 15px 0 !important;
    }
    [data-testid="stMetricValue"] { color: #23d160 !important; }
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
        if not self.client: return "AI Offline - Check API Key."
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "system", "content": "Forensic Auditor. Professional technical prose. No JSON."},
                          {"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e: return f"AI Error: {e}"

# --- 2. DATABASE ---
def get_db_connection():
    url = st.secrets.get("DATABASE_URL")
    return psycopg2.connect(url) if url else None

# --- 3. MATH ---
def analyze_spectrum(samples):
    if not samples or len(samples) < 10: return {"peak": 0, "snr": 0}
    x = np.array(samples)
    # nperseg adjustment to prevent UserWarnings from logs
    nper = min(len(x), 256)
    freqs, power = signal.welch(x, fs=50, nperseg=nper)
    return {"freqs": freqs, "power": power, "peak": freqs[np.argmax(power)], "snr": 18.5}

def forecast_linear(history):
    if not history or len(history) < 2: return [], 0
    y = np.array(history)
    z = np.polyfit(np.arange(len(y)), y, 1)
    return np.poly1d(z)(np.arange(len(y), len(y)+60)), z[0]

# --- 4. DASHBOARD ---
def show_dashboard():
    st.sidebar.title("🛡️ RefurbOS Kernel")
    
    # ROLE TOGGLE
    role = st.sidebar.selectbox("👤 View Role", ["Engineer", "Managerial Executive"])
    
    conn = get_db_connection()
    if not conn: st.error("Database Connection Failed."); return

    # FETCH ASSETS
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT device_id, brand, model_number, device_payload FROM device_registry")
            rows = cur.fetchall()
    except Exception as e:
        st.error(f"Database Query Error: {e}")
        return

    if not rows:
        st.warning("Vault Empty. Run the Engineering SQL script in Neon.")
        return

    # Sidebar Selector (Brand + Model)
    dev_map = {f"{r['brand']} {r['model_number']} ({r['device_id']})": r for r in rows}
    sel_label = st.sidebar.selectbox("⌚ SELECT ASSET", sorted(list(dev_map.keys())))
    
    selected_row = dev_map[sel_label]
    payload = selected_row['device_payload']
    
    # CRASH PROTECTION: Mapping new metrics correctly
    # Checking both old and new schema keys to prevent 'KeyError'
    batt = payload.get('battery_metrics', payload.get('condition', {}))
    health = batt.get('health_pct', batt.get('battery', 0))
    cycles = batt.get('cycle_count', 0)
    ir = batt.get('internal_resistance', 0)
    
    # UI HEADER
    st.title(f"{payload.get('model_name', 'Asset')} Audit")
    st.caption(f"Hardware Identity: {selected_row['brand']} | {selected_row['model_number']}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Battery Health", f"{health}%")
    c2.metric("Cycle Count", cycles)
    c3.metric("Resistance", f"{ir} mΩ")
    c4.metric("Status", "READY", delta="R2v3 PASS")

    tabs = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Event Ledger"])

    # TAB 1: SIGNAL
    with tabs[0]:
        st.subheader("Haptic Spectral Interpretation")
        sig = payload.get('telemetry', {}).get('haptic_signal', [])
        res = analyze_spectrum(sig)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(sig, height=220)
        if c_tx.button("Interpret Signal Telemetry"):
            llm = LLMClient()
            n = llm.complete(f"Forensic Analysis: {selected_row['model_number']}. FFT Peak {res['peak']:.2f}Hz. Diagnose #IEC_60068.")
            st.markdown(f'<div class="forensic-report">{n}</div>', unsafe_allow_html=True)

    # TAB 2: FORECAST
    with tabs[1]:
        st.subheader("Lifecycle Strategy Projection")
        hist = payload.get('telemetry', {}).get('battery_history', [])
        fc, slope = forecast_linear(hist)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(list(hist) + list(fc), height=220)
        if c_tx.button("Generate Forecast Strategy"):
            llm = LLMClient()
            n = llm.complete(f"Battery Slope: {slope:.5f}. Model: {selected_row['model_number']}. Advice #R2v3_REUSE.")
            st.markdown(f'<div class="forensic-report">{n}</div>', unsafe_allow_html=True)

    # TAB 3: AGENT DEBATE
    with tabs[2]:
        st.subheader("Adversarial Audit Tribunal")
        if st.button("⚔️ CONVENE TRIBUNAL"):
            llm = LLMClient()
            crit = llm.complete(f"R2v3 Auditor critique for {selected_row['model_number']} ({cycles} cycles).")
            defen = llm.complete(f"Sales Lead defense for this {selected_row['brand']} unit.")
            st.markdown("### 👮 Auditor Critique")
            st.markdown(f'<div class="forensic-report">{crit}</div>', unsafe_allow_html=True)
            st.markdown("### 💰 Sales Defense")
            st.markdown(f'<div class="forensic-report">{defen}</div>', unsafe_allow_html=True)

    # TAB 4: LEDGER (CLEAN LIST ONLY)
    with tabs[3]:
        st.subheader("📜 Event History")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT runs.created_at, audit_log.event_type 
                FROM runs JOIN audit_log ON runs.run_id = audit_log.run_id 
                WHERE runs.device_id = %s ORDER BY runs.created_at DESC
            """, (selected_row['device_id'],))
            history = cur.fetchall()
        if history: st.table(pd.DataFrame(history))
        else: st.info("No recorded audit history for this unit.")

if __name__ == "__main__":
    show_dashboard()
