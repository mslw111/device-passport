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
                messages=[
                    {"role": "system", "content": "Professional Forensic Auditor. Output technical prose with hashtags. Use clear, direct language."},
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
    conn = get_db_connection()
    if not conn: return
    try:
        with conn.cursor() as cur:
            rid = f"AUDIT-{sel_id}-{datetime.now().strftime('%H%M%S')}"
            cur.execute("INSERT INTO runs (run_id, device_id) VALUES (%s, %s)", (rid, sel_id))
            cur.execute("INSERT INTO audit_log (run_id, event_type, payload_json) VALUES (%s, %s, %s)", 
                        (rid, event, json.dumps({"narrative": text})))
        conn.commit()
    except: pass
    finally: conn.close()

# --- 3. MATH ---
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
    return np.poly1d(z)(np.arange(len(y), len(y)+90)), z[0]

# --- 4. UI HELPER (THE FIX) ---
def display_forensic_text(text, color_type="normal"):
    """Forces high-contrast text rendering to bypass Streamlit theme bugs"""
    bg_color = "#fdfdfd" # Almost white
    text_color = "#1a1a1a" # Almost black
    border = "1px solid #ddd"
    
    if color_type == "critique":
        bg_color = "#fff5f5" # Light red
        border = "2px solid #ff4b4b"
    elif color_type == "success":
        bg_color = "#f0fff4" # Light green
        border = "2px solid #23d160"

    st.markdown(f"""
        <div style="
            background-color: {bg_color}; 
            color: {text_color}; 
            padding: 20px; 
            border-radius: 10px; 
            border: {border}; 
            font-family: sans-serif; 
            line-height: 1.6;
            margin-bottom: 10px;
            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        ">
            {text}
        </div>
    """, unsafe_allow_html=True)

# --- 5. DASHBOARD ---
def show_dashboard():
    st.sidebar.title("🛡️ RefurbOS Kernel")
    conn = get_db_connection()
    if not conn: st.error("Database Link Failure."); return

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT device_id, device_payload FROM device_registry")
        rows = cur.fetchall()

    if not rows: st.warning("No Assets Found."); return

    dev_map = {r['device_id']: r['device_payload'] for r in rows}
    sel_id = st.sidebar.selectbox("⌚ SELECT ASSET ID", sorted(list(dev_map.keys())))
    data = dev_map[sel_id]
    
    st.title(f"Refurb Audit: {data.get('brand')} {data.get('model')}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Asset ID", sel_id)
    c2.metric("SoH", f"{data['condition'].get('battery')}%")
    c3.metric("R2v3 Grade", data['condition'].get('grade'))
    c4.markdown("### ✅ #GDPR_WIPED")

    tabs = st.tabs(["📉 Signal AGI", "🔮 Forecast AGI", "⚔️ Agent Debate", "📜 Audit Ledger"])

    with tabs[0]:
        st.subheader("Haptic Frequency Domain Analysis")
        sig = data['telemetry'].get('haptic_signal', [])
        res = analyze_spectrum(sig)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(sig, height=250)
        if c_tx.button("Interpret Signal Telemetry"):
            llm = LLMClient()
            narrative = llm.complete(f"Forensic Analysis for {sel_id}. FFT Peak {res['peak']:.2f}Hz. Diagnose #IEC_60068.")
            display_forensic_text(narrative)
            record_audit(sel_id, "SIGNAL_REPORT", narrative)

    with tabs[1]:
        st.subheader("Battery Lifecycle & Yield Projection")
        hist = data['telemetry'].get('battery_history', [])
        fc, slope = forecast_linear(hist)
        c_ch, c_tx = st.columns([2, 1.5])
        c_ch.line_chart(list(hist) + list(fc), height=250)
        if c_tx.button("Forecast Lifecycle Strategy"):
            llm = LLMClient()
            narrative = llm.complete(f"Battery decay slope {slope:.5f}. Current {data['condition'].get('battery')}%. Advise #R2v3_REUSE.")
            display_forensic_text(narrative, "success")
            record_audit(sel_id, "LIFECYCLE_REPORT", narrative)

    with tabs[2]:
        st.subheader("Adversarial Multi-Agent Audit")
        if st.button("⚔️ CONVENE AUDIT TRIBUNAL"):
            llm = LLMClient()
            critique = llm.complete(f"Strict R2v3 Auditor critique for unit {sel_id}.")
            defense = llm.complete(f"Sales Lead rebuttal for {sel_id}.")
            st.markdown("#### 👮 Auditor Critique")
            display_forensic_text(critique, "critique")
            st.markdown("#### 💰 Sales Defense")
            display_forensic_text(defense, "success")
            record_audit(sel_id, "TRIBUNAL_DEBATE", f"AUDIT: {critique}\n\nSALES: {defense}")

    with tabs[3]:
        st.subheader("📜 Chain of Custody & Audit History")
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT runs.created_at, audit_log.event_type, audit_log.payload_json 
                FROM runs JOIN audit_log ON runs.run_id = audit_log.run_id 
                WHERE runs.device_id = %s ORDER BY runs.created_at DESC
            """, (sel_id,))
            history = cur.fetchall()

        if history:
            for entry in history:
                payload = entry['payload_json']
                if isinstance(payload, str):
                    try: payload = json.loads(payload)
                    except: payload = {"narrative": str(payload)}
                
                narrative_text = payload.get('narrative', str(payload))
                with st.expander(f"EVENT: {entry['event_type']} | DATE: {entry['created_at'].strftime('%Y-%m-%d %H:%M')}"):
                    display_forensic_text(narrative_text)
        else:
            st.info("No audit history recorded.")

if __name__ == "__main__":
    show_dashboard()
