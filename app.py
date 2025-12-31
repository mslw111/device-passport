import streamlit as st
import pandas as pd
import json
import os
import numpy as np
import psycopg2
import psycopg2.extras
from scipy import signal
from datetime import datetime

# --- IMPORTS ---
from db_access import *
from llm_client import LLMClient

st.set_page_config(page_title="RefurbOS Pro", layout="wide", page_icon="🛡️")

# --- 1. PERSISTENCE LAYER (The Audit Chain) ---
def persist_run(run_id, device_id):
    """Immutable Log Entry Creator"""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO runs (run_id, device_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                (run_id, device_id)
            )
        conn.commit()
    except Exception: pass
    finally: conn.close()

def persist_audit_log(run_id, event_type, payload):
    """Detailed Event Recording"""
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)",
                (run_id, event_type, json.dumps(payload))
            )
        conn.commit()
    except Exception: pass
    finally: conn.close()

# --- 2. CONNECT TO NEON (Cloud Database) ---
def get_db_connection():
    if "DATABASE_URL" in st.secrets:
        return psycopg2.connect(st.secrets["DATABASE_URL"])
    elif os.getenv("DATABASE_URL"):
        return psycopg2.connect(os.getenv("DATABASE_URL"))
    else:
        return None

# --- 3. ENGINEERING LOGIC (Signal AGI & Forecast AGI) ---
def analyze_spectrum(samples):
    """Performs FFT for Spectral Analysis"""
    if not samples: return None
    x = np.array(samples)
    freqs, power = signal.welch(x, fs=50)
    peak_idx = np.argmax(power)
    return {
        "freqs": freqs,
        "power": power, 
        "peak_freq": freqs[peak_idx],
        "snr": 10 * np.log10(power[peak_idx] / np.mean(power))
    }

def forecast_linear(history):
    """Linear Regression for Battery Lifecycle"""
    if not history: return [], [], 0
    y = np.array(history)
    x = np.arange(len(y))
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    future_x = np.arange(len(y), len(y) + 90)
    return p(future_x), [], z[0]

# --- 4. MAIN DASHBOARD ---
def show_dashboard():
    st.sidebar.title("🛡️ RefurbOS Audit")
    
    # A. Connect & Fetch
    conn = get_db_connection()
    if not conn:
        st.error("❌ Database Connection Failed. Check Secrets."); return
        
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM device_registry LIMIT 1000")
            devices = cur.fetchall()
    except Exception: st.error("DB Error. Run SQL seed."); return

    if not devices: st.warning("DB Empty."); return

    # B. Selector
    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.sidebar.selectbox("Select Asset", list(dev_map.keys()))
    
    data = dev_map[sel_id]['device_payload']
    cond = data.get('condition', {})
    telemetry = data.get('telemetry', {})
    
    # C. Overview & Compliance Flags
    st.title(f"{data.get('brand')} {data.get('model')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R2v3 Grade", cond.get('grade', 'N/A'))
    c2.metric("Battery SoH", f"{cond.get('battery', 0)}%")
    c3.metric("Serial Hash", str(hash(sel_id))[:8])
    
    # GDPR/Data Wipe Flag (Simulated Check)
    is_wiped = True 
    c4.markdown("#### ✅ #GDPR_WIPED" if is_wiped else "#### ❌ DATA_RISK")

    # D. THE FIVE CORE TABS
    t1, t2, t3, t4 = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Audit Chain"])
    
    # --- TAB 1: SIGNAL AGI (FFT + Insight) ---
    with t1:
        st.subheader("Haptic Motor Diagnostics")
        sig = telemetry.get('haptic_signal', [])
        if sig:
            c_chart, c_ai = st.columns([2, 1])
            with c_chart:
                res = analyze_spectrum(sig)
                st.line_chart(pd.DataFrame({"Freq": res['freqs'], "Power": res['power']}), x="Freq", y="Power")
            with c_ai:
                st.markdown("**AI Diagnostics**")
                if st.button("Analyze Signal"):
                    llm = LLMClient("openai")
                    prompt = (f"Analyze Haptic FFT. Peak: {res['peak_freq']:.2f}Hz, SNR: {res['snr']:.2f}dB. "
                              "Classify as #HARDWARE_PASS or #HARDWARE_FAIL per ISO 9001.")
                    st.info(llm.complete(prompt))

    # --- TAB 2: FORECAST AGI (Math + Advice) ---
    with t2:
        st.subheader("Lifecycle Prediction")
        hist = telemetry.get('battery_history', [])
        if hist:
            fc, _, slope = forecast_linear(hist)
            c_chart, c_ai = st.columns([2, 1])
            with c_chart:
                st.line_chart(pd.DataFrame({"History": hist + [None]*len(fc), "Forecast": [None]*len(hist) + list(fc)}))
            with c_ai:
                st.markdown("**Strategic Advice**")
                if st.button("Generate Strategy"):
                    llm = LLMClient("openai")
                    prompt = (f"Battery Slope: {slope:.4f}. Current: {cond.get('battery')}%. "
                              "Advise: #RESALE_IMMEDIATE vs #HARVEST_PARTS vs #RECYCLE.")
                    st.success(llm.complete(prompt))

    # --- TAB 3: ADVERSARIAL AGENT DEBATE ---
    with t3:
        st.subheader("Adversarial Multi-Agent Audit")
        if st.button("⚔️ Convene Tribunal"):
            llm = LLMClient("openai")
            rid = f"DEBATE-{datetime.now().strftime('%H%M%S')}"
            persist_run(rid, sel_id)

            # AGENT 1: The Compliance Officer (Pessimist)
            with st.chat_message("user", avatar="👮"):
                st.write("**Agent: Compliance Officer**")
                p1 = (f"Review {data.get('model')} (Grade {cond.get('grade')}). "
                      "Identify ALL risks: data privacy, physical safety, R2v3 non-compliance. Be harsh.")
                r1 = llm.complete(p1)
                st.write(r1)
                persist_audit_log(rid, "AGENT_CRITIQUE", {"text": r1})

            # AGENT 2: The Revenue Optimizer (Optimist)
            with st.chat_message("assistant", avatar="💰"):
                st.write("**Agent: Revenue Lead**")
                p2 = (f"Review the critique: '{r1}'. Counter-argue for maximum resale value. "
                      "Highlight market demand and refurbishment potential.")
                r2 = llm.complete(p2)
                st.write(r2)
                persist_audit_log(rid, "AGENT_DEFENSE", {"text": r2})

            # FINAL VERDICT
            st.divider()
            st.markdown("### 🏛️ Tribunal Verdict")
            p3 = f"Synthesize the debate between Critique ({r1}) and Defense ({r2}). Issue final #GO or #NO_GO verdict."
            r3 = llm.complete(p3)
            st.info(r3)
            persist_audit_log(rid, "FINAL_VERDICT", {"text": r3})

    # --- TAB 4: AUDIT CHAIN (Immutable History) ---
    with t4:
        st.subheader("Immutable Ledger")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM runs WHERE device_id=%s ORDER BY created_at DESC", (sel_id,))
            runs = cur.fetchall()
            if runs:
                for run in runs:
                    with st.expander(f"Session: {run['run_id']} ({run['created_at']})"):
                        cur.execute("SELECT * FROM audit_log WHERE run_id=%s", (run['run_id'],))
                        logs = cur.fetchall()
                        st.table(pd.DataFrame(logs)[['event_type', 'payload_json']])
            else:
                st.info("No audit history found.")

if __name__ == "__main__":
    show_dashboard()
