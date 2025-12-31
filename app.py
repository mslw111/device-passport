# app.py
import streamlit as st
import pandas as pd
import json
import numpy as np
from db_access import *
from db_persist import persist_run, persist_audit_log
from dsp_utils import run_fft_welch
from llm_client import LLMClient

st.set_page_config(page_title="Wearable Passport", layout="wide", page_icon="⌚")

def render_clean(data):
    if isinstance(data, dict):
        # Flatten dict for table view
        df = pd.DataFrame([{"Metric": k.replace("_", " ").title(), "Value": v} for k,v in data.items()])
        st.table(df)
    else:
        st.write(data)

def show_device_dashboard(role_name):
    st.title(f"⌚ Wearable Refurb Console ({role_name})")
    
    # DB Load
    try: devices = list_devices()
    except: st.error("DB Connection Error. Run seed_smartwatch.py first."); st.stop()
    
    if not devices: st.warning("No devices found."); st.stop()
    
    # Sidebar Selection
    dev_map = {d["device_id"]: d for d in devices}
    sel_id = st.sidebar.selectbox("Select Unit ID", list(dev_map.keys()))
    payload = get_device_payload(sel_id)
    
    brand = payload.get("brand", "Generic")
    model = payload.get("model", "Watch")
    specs = payload.get("specs", {})
    r2 = payload.get("r2v3_flags", {})
    tel = payload.get("telemetry", {})

    # Header Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Model", f"{brand} {model}")
    c2.metric("Case Size", f"{specs.get('case_size_mm', 'N/A')} mm")
    c3.metric("Battery Health", f"{r2.get('battery_health_percent', 0)}%")
    c4.metric("Cosmetic Grade", r2.get("cosmetic_grade", "N/A"))

    # Tabs
    tabs = st.tabs(["🔍 Inspection", "⚖️ Tribunal", "💓 Sensor Health", "📝 Audit Log"])

    # TAB 1: INSPECTION DATA
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1: 
            st.subheader("Physical Condition")
            render_clean(r2)
        with c2: 
            st.subheader("Internal Telemetry")
            render_clean(tel)

    # TAB 2: AI TRIBUNAL (UPDATED PROMPTS)
    with tabs[1]:
        st.subheader("Refurbishment Viability Tribunal")
        if st.button("🚀 Convene Tribunal"):
            llm = LLMClient("gemini") # Or openai
            rid = f"RUN-{pd.Timestamp.now().strftime('%H%M%S')}"
            persist_run(rid, sel_id)
            
            # --- PROSECUTOR (RISK) ---
            with st.chat_message("user", avatar="🛑"):
                st.write("**Agent: Risk Auditor**")
                p1 = (f"You are a strict QA Auditor for Smartwatches. Analyzing {brand} {model}. "
                      f"Defects noted: {r2.get('primary_defect')}. Telemetry: {tel}. "
                      "Highlight risks regarding sweat corrosion, sensor accuracy, and waterproofing. Be aggressive.")
                r1 = llm.complete(p1)
                st.write(r1)
                persist_audit_log(rid, "RISK_AUDIT", {"text": r1})

            # --- DEFENDER (VALUE) ---
            with st.chat_message("assistant", avatar="🛡️"):
                st.write("**Agent: Value Recovery**")
                p2 = (f"Act as Refurb Engineer. Defend this {brand} {model}. "
                      f"Counter-argue the risks: {r1}. Focus on the value of the {specs.get('housing_material')} case "
                      "and modularity of the screen/battery. Argue for repair over recycling.")
                r2 = llm.complete(p2)
                st.write(r2)
                persist_audit_log(rid, "VALUE_DEFENSE", {"text": r2})

            # --- ARBITER (VERDICT) ---
            with st.chat_message("assistant", avatar="⚖️"):
                st.write("**The Arbiter**")
                p3 = (f"Act as Final Arbiter. Review arguments. Risk: {r1}. Value: {r2}. "
                      "Decide fate: 'Resell as Grade B', 'Refurbish (New Screen)', or 'Harvest for Parts'. "
                      "Provide a final dollar valuation estimate.")
                r3 = llm.complete(p3)
                st.markdown(f"### VERDICT\n{r3}")
                persist_audit_log(rid, "FINAL_VERDICT", {"text": r3})
                st.session_state['last_run'] = rid

    # TAB 3: SENSOR FFT
    with tabs[2]:
        st.subheader("Heart Rate / Taptic Engine Signal")
        samples = payload.get("fft_samples", [])
        if samples:
            st.line_chart(samples)
            if st.button("Analyze Signal Noise"):
                llm = LLMClient("gemini")
                prompt = (f"Act as a Signal Processing Engineer. Analyze this variance data for a {brand} smartwatch. "
                          "Is this sensor drift indicative of moisture damage or normal operation?")
                st.info(llm.complete(prompt))

    # TAB 4: LOGS
    with tabs[3]:
        runs = fetch_runs_for_device(sel_id)
        if not runs.empty:
            rid = st.selectbox("Select Audit Run", runs['run_id'])
            chain = fetch_audit_chain(rid)
            for _, row in chain.iterrows():
                with st.expander(f"{row['event_type']} @ {row['created_at']}"):
                    st.write(row['payload_json'])

role = st.sidebar.selectbox("User Role", ["Technician", "Manager"])
show_device_dashboard(role)
