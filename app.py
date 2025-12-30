import streamlit as st
import pandas as pd
import json
import numpy as np
import os
from db_access import *
from db_persist import persist_run, persist_audit_log
# --- CHANGED IMPORT (Fixes KeyError) ---
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
                    items.append({"Spec": k.replace('_',' ').title(), "Value": v})
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
    # TITLE CHANGED TO VERIFY FIX
    st.title(f"🎛️ {role_name} Console v3.0 (Fixed)")
    
    # --- CRASH PROTECTION ---
    try:
        devices = list_devices()
    except Exception as e:
        st.error(f"Database Error: {e}")
        st.stop()
    
    if not devices:
        st.warning("Database is empty.")
        st.info("The app is connected but needs data. Please run the SQL seed script in Neon if you haven't already.")
        st.stop()

    dev_map = {d['device_id']: d for d in devices}
    sel_id = st.selectbox("Select Asset ID", list(dev_map.keys()))
    payload = get_device_payload(sel_id)
    
    brand = payload.get("brand", "OEM")
    model = payload.get("model", "Hardware")
    
    tabs = st.tabs(["📋 Specs", "🤖 Risk Audit", "📈 Spectral Engineering", "🧠 Reliability", "🔒 Chain of Custody"])

    with tabs[0]: 
        c1, c2 = st.columns(2)
        with c1: st.subheader("Sanitization Specs"); render_clean(payload.get("gdpr_flags"))
        with c2: st.subheader("Functional Grading"); render_clean(payload.get("r2v3_flags"))

    with tabs[1]:
        st.info(f"Target: {brand} {model}")
        if st.button("🚀 Initialize Protocol"):
            llm = LLMClient("gemini")
            with st.spinner("Agents Analyzing Vectors..."):
                rid = f"RUN-{pd.Timestamp.now().strftime('%H%M%S')}"
                persist_run(rid, sel_id)
                p1 = f"Act as Lead Auditor. Review {brand} {model}. Data: {payload}. Cite specific failures. Be terse."
                r1 = llm.complete(p1)
                persist_audit_log(rid, "AGENT_ANALYSIS", {"text": r1})
                st.session_state['last_run'] = rid
                st.success("Protocol Complete")
        if 'last_run' in st.session_state:
            chain = fetch_audit_chain(st.session_state['last_run'])
            for _, row in chain.iterrows():
                with st.chat_message("assistant"):
                    st.write(f"**{row['event_type']}**")
                    render_clean(row['payload_json'])

    with tabs[2]:
        samples = payload.get("fft_samples")
        if samples:
            res = run_fft_welch(samples, 100)
            st.metric("SNR", f"{res['snr_db']:.1f} dB")
            st.line_chart(samples[:60])
            if st.button("🧠 Generate Engineering Report"):
                llm = LLMClient("gemini")
                prompt = f"Act as Vibration Analyst. Subject: {brand} {model}. SNR {res['snr_db']:.1f} dB. Write technical failure hypothesis."
                st.write(llm.complete(prompt))

    with tabs[3]:
        horizon = st.slider("Projection Horizon (Days)", 10, 90, 30)
        if st.button("Run Neural Projection"):
            f_res = forecast_with_uncertainty(payload, horizon)
            hist = f_res["history_used"]
            fc = f_res["mean_forecast"]
            events = find_signal_events(hist)
            df = pd.DataFrame({"Forecast": [None]*len(hist) + fc, "Historical": hist + [None]*len(fc)})
            st.line_chart(df)
            llm = LLMClient("gemini")
            prompt = f"Act as Reliability Engineer. Subject: {brand} {model}. Drop at Index {events['steepest_drop_idx']}. Analyze degradation."
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
