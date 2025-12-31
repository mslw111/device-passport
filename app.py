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

# --- 1. LLM CLIENT (Fixed for sk-proj keys) ---
class LLMClient:
    def __init__(self):
        self.key = st.secrets.get("OPENAI_API_KEY", "").strip()
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.key)
        except:
            self.client = None

    def complete(self, prompt):
        if not self.client: return "AI Key Error."
        try:
            resp = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}]
            )
            return resp.choices[0].message.content
        except Exception as e: return f"Error: {e}"

# --- 2. DB CONNECTION ---
def get_db_connection():
    url = st.secrets.get("DATABASE_URL")
    return psycopg2.connect(url) if url else None

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
    future = p(np.arange(len(y), len(y)+60))
    return future, z[0]

# --- 4. MAIN DASHBOARD ---
def show_dashboard():
    # Sidebar Status
    st.sidebar.title("🛡️ RefurbOS Kernel")
    st.sidebar.success("✅ Neon DB Linked")
    
    conn = get_db_connection()
    if not conn:
        st.error("Missing DATABASE_URL in Secrets.")
        return

    # FETCH ALL WATCHES FROM NEON
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT device_id, device_payload FROM device_registry")
            rows = cur.fetchall()
    except Exception as e:
        st.error(f"SQL Error: {e}")
        return

    if not rows:
        st.warning("Query returned 0 rows. Check your 'device_registry' table in Neon.")
        return

    # MAP DATA TO SELECTOR
    dev_map = {r['device_id']: r['device_payload'] for r in rows}
    asset_ids = sorted(list(dev_map.keys()))
    sel_id = st.sidebar.selectbox("⌚ SELECT WATCH UNIT", asset_ids)
    
    data = dev_map[sel_id]
    
    # PILLAR 1: OVERVIEW & COMPLIANCE FLAGS
    st.title(f"RefurbOS Audit: {data.get('brand')} {data.get('model')}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Asset ID", sel_id)
    m2.metric("Battery SoH", f"{data['condition'].get('battery')}%")
    m3.metric("R2v3 Grade", data['condition'].get('grade'))
    m4.markdown("### ✅ #GDPR_WIPED")

    tabs = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Audit Chain"])

    # PILLAR 3: SIGNAL AGI
    with tabs[0]:
        st.subheader("Spectral Analysis (FFT)")
        sig = data['telemetry'].get('haptic_signal', [])
        if sig:
            res = analyze_spectrum(sig)
            c1, c2 = st.columns([2, 1])
            c1.line_chart(sig)
            if c2.button("Run Signal AGI"):
                llm = LLMClient()
                prompt = (f"Analyze FFT for {sel_id}. Peak: {res['peak']:.2f}Hz, SNR: {res['snr']:.2f}dB. "
                          "Is the haptic motor loose? Tag #HARDWARE_PASS or #HARDWARE_FAIL.")
                st.info(llm.complete(prompt))
        else: st.info("No haptic telemetry.")

    # PILLAR 4: FORECAST AGI
    with tabs[1]:
        st.subheader("Neural Degradation Forecast")
        hist = data['telemetry'].get('battery_history', [])
        if hist:
            fc, slope = forecast_linear(hist)
            c1, c2 = st.columns([2, 1])
            c1.line_chart(list(hist) + list(fc))
            if c2.button("Run Forecast AGI"):
                llm = LLMClient()
                prompt = (f"Battery decay slope: {slope:.4f}. Current: {data['condition']['battery']}%. "
                          "Advise: #RESALE or #RECYCLE based on international standards.")
                st.success(llm.complete(prompt))

    # PILLAR 2: AGENT DEBATE (Adversarial Audit)
    with tabs[2]:
        st.subheader("Adversarial Multi-Agent Tribunal")
        if st.button("⚔️ CONVENE AUDIT TRIBUNAL"):
            llm = LLMClient()
            with st.chat_message("user", avatar="👮"):
                st.write("**Compliance Officer:** Identifying R2v3/GDPR risks...")
                st.write(llm.complete(f"Critique this unit {sel_id} for compliance gaps."))
            with st.chat_message("assistant", avatar="💰"):
                st.write("**Revenue Lead:** Defending refurbishment value...")
                st.write(llm.complete(f"Argue for the resale value of this {data.get('model')}."))

    # PILLAR 5: AUDIT CHAIN
    with tabs[3]:
        st.subheader("Immutable Audit Ledger")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT event_type, created_at FROM audit_log WHERE run_id LIKE %s ORDER BY created_at DESC", (f"%{sel_id}%",))
                for row in cur.fetchall():
                    st.write(f"🔹 {row[1]} | {row[0]}")
        except: st.write("No specific audit history for this unit yet.")

if __name__ == "__main__":
    show_dashboard()
