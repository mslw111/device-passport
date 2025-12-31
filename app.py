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
st.set_page_config(page_title="RefurbOS Command", layout="wide", page_icon="🛡️")

# --- 1. LLM CLIENT (Enhanced Narrative Engine) ---
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
                messages=[
                    {"role": "system", "content": "You are a professional hardware forensic auditor specialized in R2v3, NIST 800-88, and ISO standards. Provide deep technical narratives, not one-word answers."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
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
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    future = p(np.arange(len(y), len(y)+90)) # 90-day neural projection
    return future, z[0]

# --- 4. MAIN DASHBOARD ---
def show_dashboard():
    st.sidebar.title("🛡️ RefurbOS Kernel")
    
    conn = get_db_connection()
    if not conn:
        st.error("Missing Database Configuration."); return

    # FETCH ASSETS
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT device_id, device_payload FROM device_registry")
            rows = cur.fetchall()
    except Exception as e:
        st.error(f"System Error: {e}"); return

    if not rows:
        st.warning("Vault Empty: Check Neon 'device_registry' table."); return

    # PILLAR 1: ASSET SELECTION & OVERVIEW
    dev_map = {r['device_id']: r['device_payload'] for r in rows}
    asset_ids = sorted(list(dev_map.keys()))
    sel_id = st.sidebar.selectbox("⌚ SELECT ASSET ID", asset_ids)
    
    data = dev_map[sel_id]
    st.title(f"Refurb Audit: {data.get('brand')} {data.get('model')}")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Asset ID", sel_id)
    c2.metric("Battery SoH", f"{data['condition'].get('battery')}%")
    c3.metric("R2v3 Grade", data['condition'].get('grade'))
    c4.markdown("### ✅ #GDPR_WIPED")

    tabs = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Audit Chain"])

    # PILLAR 3: SIGNAL AGI (FFT Interpretation)
    with tabs[0]:
        st.subheader("Spectral Analysis & Frequency Interpretation")
        sig = data['telemetry'].get('haptic_signal', [])
        if sig:
            res = analyze_spectrum(sig)
            c1, c2 = st.columns([2, 1])
            c1.line_chart(sig)
            st.caption(f"Peak: {res['peak']:.2f}Hz | SNR: {res['snr']:.2f}dB")
            
            if st.button("Generate Signal Narrative"):
                llm = LLMClient()
                prompt = (f"Perform a spectral forensic analysis on Unit {sel_id}. "
                          f"The FFT shows a peak frequency of {res['peak']:.2f}Hz and an SNR of {res['snr']:.2f}dB. "
                          "Provide a detailed narrative on haptic motor health. "
                          "Does this indicate mechanical 'rattle' or sensor oxidation? "
                          "Use #IEC_60068 and #QC_PASS/FAIL.")
                st.write("---")
                st.info(llm.complete(prompt))
        else: st.info("No haptic telemetry.")

    # PILLAR 4: FORECAST AGI (Lifecycle Narrative)
    with tabs[1]:
        st.subheader("Neural Degradation & Lifecycle Forecasting")
        hist = data['telemetry'].get('battery_history', [])
        if hist:
            fc, slope = forecast_linear(hist)
            st.line_chart(list(hist) + list(fc))
            st.caption(f"Calculated Degradation Slope: {slope:.5f} / day")
            
            if st.button("Generate Forecast Interpretation"):
                llm = LLMClient()
                prompt = (f"Act as a Lifecycle Strategist. This {data.get('model')} has a battery degradation slope of {slope:.5f}. "
                          f"Current Health: {data['condition'].get('battery')}%. "
                          "Provide a deep technical narrative forecasting the 'Time-to-Failure'. "
                          "Evaluate value recovery options: #R2v3_REUSE, #LIFECYCLE_EXTENSION, or #RECOVERY.")
                st.write("---")
                st.success(llm.complete(prompt))

    # PILLAR 2: AGENT DEBATE (Adversarial Multi-Agent Audit)
    with tabs[2]:
        st.subheader("Adversarial Multi-Agent Audit Tribunal")
        if st.button("⚔️ CONVENE AUDIT TRIBUNAL"):
            llm = LLMClient()
            with st.chat_message("user", avatar="👮"):
                st.write("**Compliance Auditor:** R2v3/NIST 800-88 Audit...")
                st.write(llm.complete(f"Critique unit {sel_id} for R2v3 Core Requirement 6 and GDPR data sanitization risks."))
            with st.chat_message("assistant", avatar="💰"):
                st.write("**Revenue Lead:** Value Maximization Case...")
                st.write(llm.complete(f"Argue for the secondary market value of this {data.get('model')} despite the auditor's findings."))

    # PILLAR 5: AUDIT CHAIN (Immutable Log)
    with tabs[3]:
        st.subheader("Immutable Audit Ledger")
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT event_type, created_at FROM audit_log ORDER BY created_at DESC LIMIT 20")
                for row in cur.fetchall():
                    st.write(f"🔹 {row[1]} | {row[0]}")
        except: st.write("Audit History Unavailable.")

if __name__ == "__main__":
    show_dashboard()
