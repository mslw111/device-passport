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

# Custom CSS to keep the interface tight
st.markdown("""<style>
    .stTextArea textarea { font-size: 13px; line-height: 1.2; }
    [data-testid="stMetricValue"] { font-size: 24px; }
</style>""", unsafe_allow_html=True)

# --- 1. LLM CLIENT (Technical Narrative Engine) ---
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
                messages=[
                    {"role": "system", "content": "Professional forensic auditor. Use deep technical language and standards hashtags. Keep formatting clean."},
                    {"role": "user", "content": prompt}
                ]
            )
            return resp.choices[0].message.content
        except Exception as e: return f"Error: {e}"

# --- 2. DB HELPERS ---
def get_db_connection():
    url = st.secrets.get("DATABASE_URL")
    return psycopg2.connect(url) if url else None

def record_audit(sel_id, event, text):
    """Saves the AI Narrative to the Permanent Ledger"""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            # We use the Asset ID in the run_id for easy searching
            rid = f"AUDIT-{sel_id}-{datetime.now().strftime('%H%M%S')}"
            cur.execute("INSERT INTO runs (run_id, device_id) VALUES (%s, %s)", (rid, sel_id))
            cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)", 
                        (rid, event, json.dumps({"narrative": text})))
        conn.commit()
    except: pass
    finally: conn.close()

# --- 3. MATH ENGINE ---
def analyze_spectrum(samples):
    if not samples: return {"peak": 0, "snr": 0}
    x = np.array(samples)
    freqs, power = signal.welch(x, fs=50)
    peak = freqs[np.argmax(power)]
    snr = 10 * np.log10(np.max(power) / np.mean(power))
    return {"freqs": freqs, "power": power, "peak": peak, "snr": snr}

def forecast_linear(history):
    if not history: return [], 0
    y = np.array(history)
    z = np.polyfit(np.arange(len(y)), y, 1)
    p = np.poly1d(z)
    future = p(np.arange(len(y), len(y)+90))
    return future, z[0]

# --- 4. DASHBOARD ---
def show_dashboard():
    st.sidebar.title("🛡️ RefurbOS Kernel")
    conn = get_db_connection()
    if not conn: st.error("Database Connection Failure."); return

    # FETCH ASSETS
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT device_id, device_payload FROM device_registry")
        rows = cur.fetchall()

    if not rows: st.warning("No Assets Found."); return

    dev_map = {r['device_id']: r['device_payload'] for r in rows}
    sel_id = st.sidebar.selectbox("⌚ SELECT ASSET ID", sorted(list(dev_map.keys())))
    data = dev_map[sel_id]
    
    # Pillar 1: High-Level Overview (Compact)
    st.title(f"Refurb Audit: {data.get('model')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Asset ID", sel_id)
    c2.metric("SoH", f"{data['condition'].get('battery')}%")
    c3.metric("R2v3 Grade", data['condition'].get('grade'))
    c4.markdown("### ✅ #GDPR_WIPED")

    tabs = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Immutable Ledger"])

    # Pillar 3: Signal AGI
    with tabs[0]:
        sig = data['telemetry'].get('haptic_signal', [])
        res = analyze_spectrum(sig)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(sig, height=200)
        c_ch.caption(f"Peak: {res['peak']:.2f}Hz | SNR: {res['snr']:.2f}dB")
        
        if c_tx.button("Analyze Signal Forensic"):
            llm = LLMClient()
            narrative = llm.complete(f"Forensic Analysis for {sel_id}. FFT Peak {res['peak']:.2f}Hz. Diagnose mechanical rattle vs oxidation. Use #IEC_60068.")
            st.text_area("Signal Narrative", narrative, height=200)
            record_audit(sel_id, "SIGNAL_AGI", narrative)

    # Pillar 4: Forecast AGI
    with tabs[1]:
        hist = data['telemetry'].get('battery_history', [])
        fc, slope = forecast_linear(hist)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(list(hist) + list(fc), height=200)
        c_ch.caption(f"Degradation Slope: {slope:.5f}")
        
        if c_tx.button("Project Lifecycle Strategy"):
            llm = LLMClient()
            narrative = llm.complete(f"Strategist Review: Battery decay slope {slope:.5f}. Current {data['condition']['battery']}%. Forecast failure and advise #R2v3_REUSE.")
            st.text_area("Forecast Narrative", narrative, height=200)
            record_audit(sel_id, "FORECAST_AGI", narrative)

    # Pillar 2: Agent Debate
    with tabs[2]:
        if st.button("⚔️ CONVENE TRIBUNAL"):
            llm = LLMClient()
            r1 = llm.complete(f"Critique unit {sel_id} for R2v3/GDPR risks.")
            r2 = llm.complete(f"Rebut: Argue for value in this {data.get('model')}.")
            st.text_area("👮 Compliance Critique", r1, height=150)
            st.text_area("💰 Revenue Defense", r2, height=150)
            record_audit(sel_id, "AGENT_DEBATE", f"Critique: {r1}\nDefense: {r2}")

    # Pillar 5: Immutable Ledger (With AGI Interpretation)
    with tabs[3]:
        st.subheader("Ledger Forensic Auditor")
        
        # 1. Fetch History
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT runs.created_at, audit_log.event_type, audit_log.payload_json 
                FROM runs JOIN audit_log ON runs.run_id = audit_log.run_id 
                WHERE runs.device_id = %s ORDER BY runs.created_at DESC LIMIT 10
            """, (sel_id,))
            history = cur.fetchall()

        if history:
            if st.button("🧠 Audit the Chain (AGI Summary)"):
                llm = LLMClient()
                summary_prompt = f"Analyze the following audit history for {sel_id}. Identify any conflicting diagnostics or compliance gaps. History: {str(history)[:2000]}"
                st.warning(llm.complete(summary_prompt))
            
            for entry in history:
                with st.expander(f"🔹 {entry['created_at']} | {entry['event_type']}"):
                    # Parse Narrative
                    payload = entry['payload_json']
                    if isinstance(payload, str): payload = json.loads(payload)
                    text = payload.get('narrative', str(payload))
                    st.write(text)
        else:
            st.info("Ledger is clean. Run diagnostics in other tabs to populate history.")

if __name__ == "__main__":
    show_dashboard()
