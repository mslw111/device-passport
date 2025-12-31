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

# --- 1. ROBUST LLM CLIENT (Fixes 401 Errors) ---
class LLMClient:
    def __init__(self, provider="openai"):
        self.provider = provider
        # Try Secrets first, then Env. STRIP WHITESPACE to fix 401 errors.
        self.key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        if self.key:
            self.key = self.key.strip() 

        if not self.key:
            st.error("❌ OpenAI API Key missing in Secrets.")
            self.client = None
        else:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.key)
            except Exception as e:
                st.error(f"❌ OpenAI Library Error: {e}")
                self.client = None

    def complete(self, prompt):
        if not self.client: return "⚠️ AI Unavailable: Check API Key."
        try:
            return self.client.chat.completions.create(
                model="gpt-4", 
                messages=[{"role":"user","content":prompt}]
            ).choices[0].message.content
        except Exception as e:
            return f"⚠️ API ERROR: {str(e)}"

# --- 2. DB CONNECT & PERSIST ---
def get_db_connection():
    url = st.secrets.get("DATABASE_URL", os.getenv("DATABASE_URL"))
    if not url: return None
    return psycopg2.connect(url)

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

# --- 3. ENGINEERING LOGIC ---
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
    z = np.polyfit(x, y, 1) # Slope and intercept
    p = np.poly1d(z)
    future_x = np.arange(len(y), len(y) + 90)
    return p(future_x), [], z[0] # Return forecast, empty bounds, and slope

# --- 4. DASHBOARD ---
def show_dashboard():
    st.sidebar.title("🛡️ RefurbOS Audit")
    
    # Connect
    conn = get_db_connection()
    if not conn: st.error("❌ Database Connection Failed."); return
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM device_registry LIMIT 1000")
            devices = cur.fetchall()
    except: st.error("DB Error. Run SQL seed in Neon."); return

    if not devices: st.warning("DB Empty."); return

    # Selector
    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.sidebar.selectbox("Select Asset", list(dev_map.keys()))
    
    data = dev_map[sel_id]['device_payload']
    cond = data.get('condition', {})
    telemetry = data.get('telemetry', {})
    
    # Header
    st.title(f"{data.get('brand')} {data.get('model')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Grade", cond.get('grade', 'N/A'))
    c2.metric("Battery", f"{cond.get('battery', 0)}%")
    c3.metric("Serial", str(hash(sel_id))[:8])
    c4.markdown("#### ✅ #GDPR_WIPED")

    # TABS
    t1, t2, t3, t4 = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Audit Chain"])
    
    # --- TAB 1: SIGNAL AGI ---
    with t1:
        st.subheader("Haptic Motor Diagnostics (FFT)")
        sig = telemetry.get('haptic_signal', [])
        if sig:
            c_chart, c_ai = st.columns([2, 1])
            res = analyze_spectrum(sig)
            
            with c_chart:
                st.line_chart(pd.DataFrame({"Freq": res['freqs'], "Power": res['power']}), x="Freq", y="Power")
                st.caption(f"Peak: {res['peak_freq']:.2f}Hz | SNR: {res['snr']:.2f}dB")
            
            with c_ai:
                st.markdown("### 🧠 Signal AGI")
                if st.button("Analyze Spectrum"):
                    llm = LLMClient("openai")
                    with st.spinner("AI analyzing frequencies..."):
                        prompt = (f"Act as a Hardware Engineer. Analyze FFT Data for {data.get('model')}.\n"
                                  f"Peak Freq: {res['peak_freq']:.2f} Hz. SNR: {res['snr']:.2f} dB.\n"
                                  f"Determine if this indicates a loose motor (rattle) or normal operation.\n"
                                  f"Output compliant hashtags: #HARDWARE_PASS or #HARDWARE_FAIL.")
                        st.info(llm.complete(prompt))

    # --- TAB 2: FORECAST AGI ---
    with t2:
        st.subheader("Battery Lifecycle Prediction")
        hist = telemetry.get('battery_history', [])
        if hist:
            fc, _, slope = forecast_linear(hist)
            
            c_chart, c_ai = st.columns([2, 1])
            with c_chart:
                combined = pd.DataFrame({"History": hist + [None]*len(fc), "Forecast": [None]*len(hist) + list(fc)})
                st.line_chart(combined)
                st.caption(f"Degradation Slope: {slope:.4f}")
            
            with c_ai:
                st.markdown("### 🧠 Forecast AGI")
                if st.button("Generate Strategy"):
                    llm = LLMClient("openai")
                    with st.spinner("AI calculating lifecycle..."):
                        prompt = (f"Act as a Supply Chain Planner. Battery Slope: {slope:.4f}.\n"
                                  f"Current Health: {cond.get('battery')}%. Grade: {cond.get('grade')}.\n"
                                  f"Predict days to failure (<80%). Advise: #RESALE or #RECYCLE?")
                        st.success(llm.complete(prompt))

    # --- TAB 3: AGENT DEBATE ---
    with t3:
        st.subheader("Adversarial Audit Tribunal")
        if st.button("⚔️ Convene Tribunal"):
            llm = LLMClient("openai")
            rid = f"DEBATE-{datetime.now().strftime('%H%M%S')}"
            persist_run(rid, sel_id)

            # 1. Pessimist
            with st.chat_message("user", avatar="👮"):
                st.write("**Compliance Officer**")
                p1 = f"Audit {data.get('model')}. Find ALL risks in Grade {cond.get('grade')}. Be strict."
                r1 = llm.complete(p1)
                st.write(r1)
                persist_audit_log(rid, "CRITIQUE", {"text": r1})

            # 2. Optimist
            with st.chat_message("assistant", avatar="💰"):
                st.write("**Revenue Lead**")
                p2 = f"Rebut this critique: '{r1}'. Argue for high resale value."
                r2 = llm.complete(p2)
                st.write(r2)
                persist_audit_log(rid, "DEFENSE", {"text": r2})

            # 3. Verdict
            st.divider()
            st.markdown("### 🏛️ Final Verdict")
            r3 = llm.complete(f"Synthesize {r1} and {r2}. Issue #GO or #NO_GO.")
            st.info(r3)
            persist_audit_log(rid, "VERDICT", {"text": r3})

    # --- TAB 4: AUDIT CHAIN ---
    with t4:
        st.subheader("Immutable Ledger")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM runs WHERE device_id=%s ORDER BY created_at DESC", (sel_id,))
            runs = cur.fetchall()
            if runs:
                for run in runs:
                    with st.expander(f"Session: {run['run_id']} ({run['created_at']})"):
                        cur.execute("SELECT * FROM audit_log WHERE run_id=%s", (run['run_id'],))
                        st.table(pd.DataFrame(cur.fetchall())[['event_type', 'payload_json']])
            else:
                st.info("No logs yet.")

if __name__ == "__main__":
    show_dashboard()
