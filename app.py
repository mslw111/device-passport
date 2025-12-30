import streamlit as st
import pandas as pd
import json
import numpy as np
import os
from db_access import *
from db_persist import persist_run, persist_audit_log
from dsp_utils import run_fft_welch
from forecasting import forecast_with_uncertainty
from llm_client import LLMClient

st.set_page_config(page_title="Device Passport", layout="wide")

# --- UTILS ---
def render_clean(data):
    if isinstance(data, str):
        try: data = json.loads(data)
        except: st.write(data); return
    if isinstance(data, dict):
        if "text" in data: st.info(data["text"])
        else:
            items = []
            for k,v in data.items():
                if not isinstance(v, (list,dict)):
                    items.append({"Parameter": k.replace('_',' ').title(), "Status": v})
            if items:
                st.table(pd.DataFrame(items))
    elif isinstance(data, list):
        for i in data: st.write(f"- {i}")
    else: st.write(data)

def find_signal_events(data):
    arr = np.array(data)
    diffs = np.diff(arr)
    drop_idx = np.argmin(diffs) if len(diffs) > 0 else 0
    return {
        "current": arr[-1],
        "steepest_drop_idx": drop_idx,
        "steepest_drop_val": np.min(diffs) if len(diffs) > 0 else 0,
        "variance": np.var(arr)
    }

def show_device_dashboard(role_name):
    st.title(f"🎛️ {role_name} Console v4.0 (Persona-Aware)")
    
    # --- DB PROTECTION ---
    try:
        devices = list_devices()
    except Exception:
        st.error("Database connection failed. Please check Neon URL.")
        st.stop()
    
    if not devices:
        st.warning("Database is empty. Please run the SQL Seed Script in Neon.")
        st.stop()

    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.selectbox("Select Asset ID", list(dev_map.keys()))
    payload = get_device_payload(sel_id)
    
    brand = payload.get("brand", "OEM")
    model = payload.get("model", "Hardware")
    
    # --- DEFINE ROLE PERSONAS ---
    if role_name == "Engineer":
        persona_role = "Senior Failure Analysis Engineer"
        focus_area = "root cause physics, component failure modes, circuit integrity, and repair feasibility"
        tone = "Technical, Precise, Jargon-Heavy (e.g., Impedance, Harmonic Distortion)"
    else: # Management
        persona_role = "Product Operations Director"
        focus_area = "financial risk, warranty liability, brand reputation, and resale grading (Grade A/B/C)"
        tone = "Executive Summary, Financial Focus, Strategic Decision Making"

    # TABS
    tabs = st.tabs(["📋 Specs", "🤖 Risk Audit", "📈 Spectral Analysis", "🧠 Lifecycle Prediction", "🔒 Audit Chain"])

    with tabs[0]: 
        c1, c2 = st.columns(2)
        with c1: st.subheader("Sanitization Specs"); render_clean(payload.get("gdpr_flags"))
        with c2: st.subheader("Functional Grading"); render_clean(payload.get("r2v3_flags"))

    with tabs[1]: # RISK AUDIT
        st.info(f"Target: {brand} {model}")
        if st.button("🚀 Initialize Protocol"):
            llm = LLMClient("gemini")
            with st.spinner(f"{role_name} Agents Analyzing..."):
                rid = f"RUN-{pd.Timestamp.now().strftime('%H%M%S')}"
                persist_run(rid, sel_id)
                
                # Dynamic Prompt
                p1 = f"""
                Act as a {persona_role}.
                Review this device: {brand} {model}.
                Data: {payload}.
                Task: Identify critical issues focusing strictly on {focus_area}.
                Tone: {tone}.
                Be concise.
                """
                r1 = llm.complete(p1)
                persist_audit_log(rid, "AGENT_ANALYSIS", {"text": r1})
                st.session_state['last_run'] = rid
                st.success("Analysis Complete")
        
        if 'last_run' in st.session_state:
            chain = fetch_audit_chain(st.session_state['last_run'])
            for _, row in chain.iterrows():
                with st.chat_message("assistant"):
                    st.write(f"**{row['event_type']}**")
                    render_clean(row['payload_json'])

    with tabs[2]: # SPECTRAL (FFT)
        st.subheader(f"{role_name} Signal Analysis")
        samples = payload.get("fft_samples")
        if samples:
            res = run_fft_welch(samples, 100)
            st.metric("Signal-to-Noise Ratio (SNR)", f"{res['snr_db']:.1f} dB")
            st.line_chart(samples[:60])
            
            btn_label = "🧠 Generate Engineering Report" if role_name == "Engineer" else "🧠 Assess Quality Risks"
            
            if st.button(btn_label):
                llm = LLMClient("gemini")
                
                # SPLIT BRAIN PROMPT
                if role_name == "Engineer":
                    prompt = f"""
                    Act as a Senior Vibration Analyst.
                    Subject: {brand} {model}.
                    Telemetry: SNR is {res['snr_db']:.1f} dB. Peak Frequency: {res['peak_freq_hz']:.1f} Hz.
                    
                    Task: Write a Failure Hypothesis.
                    1. Analyze the noise floor. Use terms like 'Harmonic Distortion' or 'Mechanical Resonance'.
                    2. Speculate on the exact component failure (e.g. 'Microphone membrane detachment').
                    3. Recommend a repair (e.g. 'Reflow solder').
                    """
                else:
                    prompt = f"""
                    Act as a Quality Assurance Director.
                    Subject: {brand} {model}.
                    Telemetry: SNR is {res['snr_db']:.1f} dB (Industry Standard for this price point is >20dB).
                    
                    Task: Write a Business Risk Assessment.
                    1. Explain what low SNR means for the customer (e.g., "Noise Cancellation will fail").
                    2. Estimate the return rate risk (High/Medium/Low).
                    3. Decision: Should we ship this batch or issue a Recall?
                    """
                
                with st.spinner("Processing..."):
                    st.markdown(f"### 📝 {role_name} Assessment")
                    st.write(llm.complete(prompt))

    with tabs[3]: # LIFECYCLE (FORECAST)
        st.subheader(f"{role_name} Reliability Projection")
        horizon = st.slider("Horizon (Days)", 10, 90, 30)
        
        if st.button("Run Neural Projection"):
            f_res = forecast_with_uncertainty(payload, horizon)
            hist = f_res["history_used"]
            fc = f_res["mean_forecast"]
            events = find_signal_events(hist)
            
            df = pd.DataFrame({"Forecast": [None]*len(hist) + fc, "Historical": hist + [None]*len(fc)})
            st.line_chart(df)
            
            llm = LLMClient("gemini")
            
            # SPLIT BRAIN PROMPT
            if role_name == "Engineer":
                prompt = f"""
                Act as a Battery Chemist. Subject: {brand} {model}.
                Data: A steep drop of {events['steepest_drop_val']:.2f}% was detected at Index {events['steepest_drop_idx']}.
                Task:
                1. Analyze the degradation curve. Is this 'Lithium Plating' or 'SEI Layer Decomposition'?
                2. Assess thermal risks.
                """
            else:
                prompt = f"""
                Act as a Warranty Manager. Subject: {brand} {model}.
                Data: Predicted health drops to {fc[-1]:.1f}% in {horizon} days.
                Task:
                1. Will this device survive the 12-month warranty period?
                2. Financial Risk: If we resell this, what is the likelihood of a warranty claim (C:\Users\msell\OneDrive\AIAlchemy\devicepassport\clean$)?
                3. Recommendation: Sell as 'Refurbished Grade B' or 'Scrap for Parts'?
                """
            
            with st.spinner("Calculating..."):
                st.markdown(f"### 📝 {role_name} Assessment")
                st.write(llm.complete(prompt))

    with tabs[4]: # AUDIT
        runs = fetch_runs_for_device(sel_id)
        if not runs.empty:
            rid = st.selectbox("Run ID", runs['run_id'])
            chain = fetch_audit_chain(rid)
            st.dataframe(chain[["created_at", "event_type"]], use_container_width=True)
            for _, row in chain.iterrows():
                with st.expander(f"{row['event_type']} ({row['created_at']})"):
                    render_clean(row['payload_json'])

    if role_name == "Engineer":
        st.markdown("---")
        with st.expander("🔧 Data Dictionary"): st.write(payload)

st.sidebar.title("✅ NEURAL AUDIT")
role = st.sidebar.selectbox("Role", ["Management", "Engineer"])
show_device_dashboard(role)
