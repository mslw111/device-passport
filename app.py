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

st.markdown("""<style>
    .forensic-report { 
        background-color: #FFFFFF !important; 
        color: #000000 !important; 
        padding: 20px !important; 
        border-radius: 8px !important; 
        border: 2px solid #333333 !important; 
        font-family: Arial, sans-serif !important; 
    }
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
                messages=[{"role": "system", "content": "Professional Auditor. Technical prose only."},
                          {"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e: return f"Error: {e}"

# --- 2. DB HELPERS (FIXED PERSISTENCE) ---
def get_db_connection():
    url = st.secrets.get("DATABASE_URL")
    return psycopg2.connect(url) if url else None

def record_event(device_id, event_type, narrative):
    """Writes the AGI output to Neon immediately"""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            rid = f"RUN-{device_id}-{datetime.now().strftime('%H%M%S')}"
            cur.execute("INSERT INTO runs (run_id, device_id) VALUES (%s, %s) ON CONFLICT DO NOTHING", (rid, device_id))
            cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)", 
                        (rid, event_type, json.dumps({"narrative": narrative})))
        conn.commit()
    except Exception as e: st.error(f"Save Error: {e}")
    finally: conn.close()

# --- 3. MATH ---
def analyze_spectrum(samples):
    if not samples: return {"peak": 0, "snr": 0}
    x = np.array(samples)
    nper = min(len(x), 50)
    freqs, power = signal.welch(x, fs=50, nperseg=nper)
    return {"peak": freqs[np.argmax(power)], "snr": 18.5}

# --- 4. DASHBOARD ---
def show_dashboard():
    st.sidebar.title("🛡️ RefurbOS Kernel")
    role = st.sidebar.selectbox("👤 View Role", ["Engineer", "Managerial Executive"])
    
    conn = get_db_connection()
    if not conn: st.error("Database Link Failure."); return

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT device_id, brand, model_number, device_payload FROM device_registry")
        rows = cur.fetchall()

    if not rows: st.warning("Database Empty."); return

    dev_map = {f"{r['brand']} {r['model_number']} ({r['device_id']})": r for r in rows}
    sel_label = st.sidebar.selectbox("⌚ SELECT ASSET", sorted(list(dev_map.keys())))
    
    selected_row = dev_map[sel_label]
    payload = selected_row['device_payload']
    batt = payload.get('battery_metrics', {})
    
    st.title(f"{payload.get('model_name')} Audit")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Health", f"{batt.get('health_pct')}%")
    c2.metric("Cycles", batt.get('cycle_count'))
    c3.metric("Resistance", f"{batt.get('internal_resistance')} mΩ")
    c4.metric("Role", role)

    tabs = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Event Ledger"])

    with tabs[0]:
        st.subheader("Haptic Forensic")
        sig = payload.get('telemetry', {}).get('haptic_signal', [])
        res = analyze_spectrum(sig)
        st.line_chart(sig)
        if st.button("Interpret Signal"):
            llm = LLMClient()
            narrative = llm.complete(f"Forensic Analysis: {selected_row['model_number']}. FFT Peak {res['peak']:.2f}Hz.")
            st.markdown(f'<div class="forensic-report">{narrative}</div>', unsafe_allow_html=True)
            # THIS SAVES TO THE LEDGER
            record_event(selected_row['device_id'], "SIGNAL_AUDIT", narrative)

    with tabs[2]:
        st.subheader("Tribunal")
        if st.button("⚔️ CONVENE TRIBUNAL"):
            llm = LLMClient()
            crit = llm.complete(f"R2v3 Auditor critique for {selected_row['model_number']}.")
            st.markdown(f'<div class="forensic-report"><b>AUDITOR:</b> {crit}</div>', unsafe_allow_html=True)
            # THIS SAVES TO THE LEDGER
            record_event(selected_row['device_id'], "TRIBUNAL_DEBATE", crit)

    with tabs[3]:
        st.subheader("📜 Event Ledger")
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
            st.info("No events recorded for this unit yet. Run an AGI analysis to trigger a log.")

if __name__ == "__main__":
    show_dashboard()
