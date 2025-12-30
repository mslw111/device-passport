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
    st.title(f"🎛️ {role_name} Console v6.1 (Sanitized)")
    
    # --- DB PROTECTION ---
    try: devices = list_devices()
    except Exception: st.error("Database connection failed."); st.stop()
    if not devices: st.warning("Database empty."); st.stop()

    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.selectbox("Select Asset ID", list(dev_map.keys()))
    payload = get_device_payload(sel_id)
    brand = payload.get("brand", "OEM")
    model = payload.get("model", "Hardware")
    
    # --- TABS ---
    tabs = st.tabs(["📋 Specs", "⚖️ Agentic Tribunal", "📈 Spectral Analysis", "🧠 Lifecycle Prediction", "🔒 Audit Chain"])

    with tabs[0]: 
        c1, c2 = st.columns(2)
        with c1: st.subheader("Sanitization Specs"); render_clean(payload.get("gdpr_flags"))
        with c2: st.subheader("Functional Grading"); render_clean(payload.get("r2v3_flags"))

    # --- TAB 1: AGENTIC DEBATE ---
    with tabs[1]: 
        st.subheader("Multi-Agent Adversarial Audit")
        st.info(f"Subject: {brand} {model}")
        
        if st.button("🚀 Launch Tribunal Protocol"):
            llm = LLMClient("gemini")
            rid = f"RUN-{pd.Timestamp.now().strftime('%H%M%S')}"
            persist_run(rid, sel_id)

            # 1. AGENT A: PROSECUTOR (Using 'rf' for Raw F-String to prevent unicode errors)
            with st.chat_message("user", avatar="🛑"):
                st.write("**Agent A (Risk Auditor):** Analyzing liabilities...")
                p_risk = rf"""
                Act as a STRICT Risk Auditor. Analyze {brand} {model}.
                Data: {payload}.
                Goal: Find every reason to REJECT this device (Safety, GDPR, Cosmetic).
                Be aggressive. Max 3 sentences.
                """
                r_risk = llm.complete(p_risk)
                st.write(r_risk)
                persist_audit_log(rid, "DEBATE_PROSECUTOR", {"text": r_risk})

            # 2. AGENT B: DEFENDER
            with st.chat_message("assistant", avatar="🛡️"):
                st.write("**Agent B (Value Recovery):** Analyzing potential...")
                p_value = rf"""
                Act as a Value Recovery Specialist. Defend {brand} {model}.
                Data: {payload}.
                Prosecutor Argument: "{r_risk}"
                Goal: Find value (spare parts, rare metals, repairable).
                Counter-argue the risks. Max 3 sentences.
                """
                r_value = llm.complete(p_value)
                st.write(r_value)
                persist_audit_log(rid, "DEBATE_DEFENDER", {"text": r_value})

            # 3. ARBITER
            with st.chat_message("assistant", avatar="⚖️"):
                st.write("**The Arbiter (Final Verdict):** Weighing evidence...")
                p_arbiter = rf"""
                Act as a Supreme Arbiter.
                Device: {brand} {model}.
                Prosecutor says: "{r_risk}"
                Defender says: "{r_value}"
                
                TASK: Issue a binding verdict.
                1. Who won? (Risk vs Value).
                2. Final Disposition: (Resell, Refurbish, or Recycle).
                3. Rationale.
                """
                r_arbiter = llm.complete(p_arbiter)
                st.markdown(f"### 🏛️ VERDICT: \n{r_arbiter}")
                persist_audit_log(rid, "DEBATE_VERDICT", {"text": r_arbiter})
            
            st.session_state['last_run'] = rid
            st.success("Tribunal Adjourned")

        if 'last_run' in st.session_state:
            st.divider()
            chain = fetch_audit_chain(st.session_state['last_run'])
            for _, row in chain.iterrows():
                icon = "⚖️" if "VERDICT" in row['event_type'] else "🛑" if "PROSECUTOR" in row['event_type'] else "🛡️"
                with st.chat_message("assistant", avatar=icon):
                    st.write(f"**{row['event_type']}**")
                    render_clean(row['payload_json'])

    # --- TAB 2: SPECTRAL ---
    with tabs[2]: 
        st.subheader(f"{role_name} Signal Analysis")
        samples = payload.get("fft_samples")
        if samples:
            res = run_fft_welch(samples, 100)
            st.metric("Signal-to-Noise Ratio (SNR)", f"{res['snr_db']:.1f} dB")
            st.line_chart(samples[:60])
            
            btn_label = "🧠 Generate Engineering Report" if role_name == "Engineer" else "🧠 Assess Quality Risks"
            
            if st.button(btn_label):
                llm = LLMClient("gemini")
                
                if role_name == "Engineer":
                    prompt = rf"""
                    Act as a Senior Vibration Analyst. Subject: {brand} {model}.
                    Telemetry: SNR is {res['snr_db']:.1f} dB. Peak Frequency: {res['peak_freq_hz']:.1f} Hz.
                    Task: Write a Failure Hypothesis.
                    1. Analyze the noise floor. Use terms like 'Harmonic Distortion' or 'Mechanical Resonance'.
                    2. Speculate on the exact component failure (e.g. 'Microphone membrane detachment').
                    3. Recommend a repair (e.g. 'Reflow solder').
                    """
                else:
                    prompt = rf"""
                    Act as a Quality Assurance Director. Subject: {brand} {model}.
                    Telemetry: SNR is {res['snr_db']:.1f} dB (Standard >20dB).
                    Task: Write a Business Risk Assessment.
                    1. Explain what low SNR means for the customer (e.g., "Noise Cancellation will fail").
                    2. Estimate Return Rate Risk (High/Low).
                    3. Decision: Quarantine or Ship?
                    """
                
                with st.spinner("Processing..."):
                    st.markdown(f"### 📝 {role_name} Assessment")
                    st.write(llm.complete(prompt))

    # --- TAB 3: LIFECYCLE ---
    with tabs[3]: 
        st.subheader(f"{role_name} Reliability Projection")
        horizon = st.slider("Horizon", 10, 90, 30)
        
        if st.button("Run Neural Projection"):
            f_res = forecast_with_uncertainty(payload, horizon)
            hist = f_res["history_used"]
            fc = f_res["mean_forecast"]
            events = find_signal_events(hist)
            df = pd.DataFrame({"Forecast": [None]*len(hist) + fc, "Historical": hist + [None]*len(fc)})
            st.line_chart(df)
            
            llm = LLMClient("gemini")
            
            if role_name == "Engineer":
                prompt = rf"""
                Act as a Battery Chemist. Subject: {brand} {model}.
                Data: A steep drop of {events['steepest_drop_val']:.2f}% was detected at Index {events['steepest_drop_idx']}.
                Task:
                1. Analyze the degradation curve. Is this 'Lithium Plating' or 'SEI Layer Decomposition'?
                2. Assess thermal risks.
                """
            else:
                prompt = rf"""
                Act as a Warranty Manager. Subject: {brand} {model}.
                Data: Predicted health drops to {fc[-1]:.1f}% in {horizon} days.
                Task:
                1. Will this device survive the 12-month warranty period?
                2. Financial Risk: Likelihood of warranty claim (C:\Users\msell\OneDrive\AIAlchemy\devicepassport\clean$)?
                3. Recommendation: Sell as 'Refurbished Grade B' or 'Scrap'?
                """
            
            st.markdown(f"### 📝 {role_name} Assessment")
            st.write(llm.complete(prompt))

    with tabs[4]: 
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
